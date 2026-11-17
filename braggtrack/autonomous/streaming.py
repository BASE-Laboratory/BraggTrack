"""Strictly causal online scan-by-scan tracker for live beamline streaming.

Processes incoming 3D rocking-curve scans ``t = 0, 1, 2, ...`` without looking ahead:
- Causal trailing-window median Otsu thresholding (using only scans ``<= t``).
- 3D LoG + seeded watershed segmentation and Poisson-uncertainty feature extraction.
- Batched multi-view ``DINOv2`` embedding and cached ROI patch-dispersion scoring.
- Incremental Hungarian + gap-closing + ``1 -> 2+`` split/merge trajectory DAG update.
- Autonomous DFXM "Scout-and-Zoom" target selection at the end of every scan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from braggtrack.io.models import AxisCalibration
from braggtrack.io.nexus import extract_axis_calibration, extract_scan_metadata, load_primary_volume
from braggtrack.segmentation import (
    extract_instance_table,
    otsu_threshold,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
)
from braggtrack.semantic import DEFAULT_DINO_MODEL, make_multiview_encoder, make_patch_encoder
from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips
from braggtrack.tracking.cost import GeometrySemanticCost, PositionShapeCost
from braggtrack.tracking.events import LineageEvent, SplitMergeConfig, events_to_table
from braggtrack.tracking.gap_closing import GapClosingConfig
from braggtrack.tracking.kinematics import GrainKinematics, compute_grain_kinematics
from braggtrack.tracking.lifecycle import build_tracks, tracks_to_table
from braggtrack.tracking.precursors import (
    ObservationPrecursorScore,
    PrecursorCandidate,
    PrecursorConfig,
    compute_fracture_precursors,
)

from .trigger import ZoomTarget, ZoomTriggerPolicy, select_zoom_targets

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.semantic.dino import BackendName


@dataclass(frozen=True)
class StreamingStepResult:
    """Snapshot of online causal tracking and autonomous DFXM triggers after scan ``scan_idx``."""

    scan_idx: int
    scan_name: str
    threshold_raw: float
    threshold_causal: float
    n_spots: int
    n_active_tracks: int
    n_lineage_events_total: int
    new_lineage_events: list[LineageEvent]
    precursor_candidates: list[PrecursorCandidate]
    zoom_targets: list[ZoomTarget]
    step_latency_sec: float
    stimulus: ScanStimulusPoint | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_idx": int(self.scan_idx),
            "scan_name": self.scan_name,
            "threshold_raw": round(float(self.threshold_raw), 5),
            "threshold_causal": round(float(self.threshold_causal), 5),
            "n_spots": int(self.n_spots),
            "n_active_tracks": int(self.n_active_tracks),
            "n_lineage_events_total": int(self.n_lineage_events_total),
            "new_lineage_events": events_to_table(self.new_lineage_events),
            "n_precursor_candidates": len(self.precursor_candidates),
            "zoom_targets": [t.to_bliss_command() for t in self.zoom_targets],
            "step_latency_sec": round(float(self.step_latency_sec), 4),
            "stimulus": self.stimulus.to_dict() if self.stimulus is not None else None,
        }


class CausalStreamingTracker:
    """Stateful strictly causal scan-by-scan 4D reflection tracker and zoom trigger engine."""

    def __init__(
        self,
        *,
        backend: BackendName | None = "auto",
        model_name: str = DEFAULT_DINO_MODEL,
        smooth_window: int = 5,
        seed_peak_fraction: float = 0.08,
        seed_response_percentile: float = 0.0,
        min_spot_voxels: int = 4,
        calibration: AxisCalibration | None = None,
        detector: DetectorGeometry | None = None,
        cost_alpha: float = 1.0,
        cost_beta: float = 0.25,
        max_cost: float = float("inf"),
        max_gap: int = 0,
        split_max_distance: float = 25.0,
        precursor_config: PrecursorConfig | None = None,
        trigger_policy: ZoomTriggerPolicy | None = None,
        batch_size: int = 32,
        use_patch_encoder: bool = True,
    ) -> None:
        self.smooth_window = max(1, int(smooth_window))
        self.seed_peak_fraction = float(seed_peak_fraction)
        self.seed_response_percentile = float(seed_response_percentile)
        self.min_spot_voxels = max(1, int(min_spot_voxels))
        self.calibration = calibration
        self.detector = detector
        self.cost_alpha = float(cost_alpha)
        self.cost_beta = float(cost_beta)
        self.max_cost = float(max_cost)
        self.batch_size = max(1, int(batch_size))

        self.gap_config = GapClosingConfig(max_gap=max(0, int(max_gap))) if max_gap > 0 else None
        self.split_merge_config = SplitMergeConfig(max_distance=float(split_max_distance))
        self.precursor_config = precursor_config or PrecursorConfig()
        self.trigger_policy = trigger_policy or ZoomTriggerPolicy()

        self._multiview_encoder = (
            make_multiview_encoder(backend, model_name=model_name) if self.cost_beta > 0.0 else None
        )
        self._patch_encoder = make_patch_encoder(backend, model_name=model_name) if use_patch_encoder else None
        self._cost_fn = GeometrySemanticCost(
            PositionShapeCost(),
            cost_alpha=self.cost_alpha,
            cost_beta=self.cost_beta,
        )

        # Strictly causal state accumulated across scans 0 .. t
        self._raw_thresholds: list[float] = []
        self._causal_thresholds: list[float] = []
        self._scan_names: list[str] = []
        self._scan_timestamps: dict[int, str | None] = {}
        self._scan_tables: list[list[dict[str, Any]]] = []
        self._stimuli: dict[int, ScanStimulusPoint] = {}
        self._roi_cache: dict[tuple[int, int], tuple[int, float]] = {}
        self._cooldown_by_lineage: dict[int, int] = {}

        # Latest DAG & derived tables
        self._graph: nx.DiGraph = nx.DiGraph()
        self._track_table: list[dict[str, Any]] = []
        self._kinematics: list[GrainKinematics] = []
        self._obs_precursor_scores: list[ObservationPrecursorScore] = []
        self._precursor_candidates: list[PrecursorCandidate] = []
        self._all_zoom_targets: list[ZoomTarget] = []

    @property
    def n_scans_processed(self) -> int:
        return len(self._scan_tables)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def track_table(self) -> list[dict[str, Any]]:
        return list(self._track_table)

    @property
    def kinematics(self) -> list[GrainKinematics]:
        return list(self._kinematics)

    @property
    def obs_precursor_scores(self) -> list[ObservationPrecursorScore]:
        return list(self._obs_precursor_scores)

    @property
    def all_zoom_targets(self) -> list[ZoomTarget]:
        return list(self._all_zoom_targets)

    def _resolve_volume_and_meta(
        self,
        volume_or_path: np.ndarray | str | Path,
        scan_name: str | None,
        timestamp: str | None,
    ) -> tuple[np.ndarray, str, str | None, AxisCalibration]:
        if isinstance(volume_or_path, (str, Path)):
            p = Path(volume_or_path)
            if p.is_dir():
                h5s = sorted([*p.glob("*.h5"), *p.glob("*.hdf5"), *p.glob("*.nxs")])
                if not h5s:
                    raise FileNotFoundError(f"No HDF5/NeXus file found in scan directory {p}")
                file_path = h5s[0]
                default_name = p.name
            else:
                file_path = p
                default_name = p.stem
            vol = load_primary_volume(file_path)
            meta = extract_scan_metadata(file_path)
            ts = timestamp if timestamp is not None else meta.get("start_time")
            cal = self.calibration if self.calibration is not None else extract_axis_calibration(file_path)
            if self.calibration is None and cal.calibrated:
                self.calibration = cal
            return vol, (scan_name or default_name), ts, cal

        vol = np.asarray(volume_or_path, dtype=np.float64)
        cal = self.calibration if self.calibration is not None else AxisCalibration()
        name = scan_name or f"scan_{len(self._scan_tables) + 1:04d}"
        return vol, name, timestamp, cal

    def step(
        self,
        volume_or_path: np.ndarray | str | Path,
        *,
        scan_name: str | None = None,
        timestamp: str | None = None,
        stimulus: ScanStimulusPoint | None = None,
    ) -> StreamingStepResult:
        """Ingest one new 3D scan ``t`` causally and emit tracking + autonomous zoom targets."""
        t0 = time.perf_counter()
        scan_idx = len(self._scan_tables)
        vol, resolved_name, resolved_ts, active_cal = self._resolve_volume_and_meta(
            volume_or_path, scan_name, timestamp
        )
        if stimulus is not None and resolved_ts is None and stimulus.timestamp is not None:
            resolved_ts = stimulus.timestamp

        # 1. Strictly causal trailing-window Otsu threshold (scans max(0, t - W + 1) .. t only!)
        raw_thr = float(otsu_threshold(vol))
        self._raw_thresholds.append(raw_thr)
        win_lo = max(0, len(self._raw_thresholds) - self.smooth_window)
        causal_thr = float(np.median(self._raw_thresholds[win_lo:]))
        self._causal_thresholds.append(causal_thr)

        # 2. Classical 3D LoG + watershed segmentation & calibrated feature + uncertainty extraction
        seg = segment_classical(
            vol,
            threshold=causal_thr,
            seed_peak_fraction=self.seed_peak_fraction,
            seed_response_percentile=self.seed_response_percentile,
        )
        labels = relabel_sequential(remove_small_objects(seg.labeled_volume, min_size=self.min_spot_voxels))
        spots = extract_instance_table(labels, vol, calibration=active_cal)

        # 3. Batched DINOv2 multiview embeddings for all spots in scan t
        if self._multiview_encoder is not None and spots:
            triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for row in spots:
                lbl = int(row["label"])
                masked_cube, _ = crop_spot_cube(
                    vol,
                    labels,
                    lbl,
                    row,
                    margin=self.precursor_config.roi_margin,
                )
                triplets.append(orthogonal_mips(masked_cube))
            embs = self._multiview_encoder.embed_batch(triplets, batch_size=self.batch_size)
            for row, vec in zip(spots, embs, strict=True):
                row["embedding"] = vec

        self._scan_names.append(resolved_name)
        self._scan_timestamps[scan_idx] = resolved_ts
        self._scan_tables.append(spots)
        if stimulus is not None:
            self._stimuli[scan_idx] = stimulus

        # 4. Build causal trajectory DAG over scans 0 .. t
        G = build_tracks(
            self._scan_tables,
            cost_fn=self._cost_fn,
            max_cost=self.max_cost,
            gap_closing_config=self.gap_config,
            split_merge_config=self.split_merge_config,
            scan_timestamps=self._scan_timestamps,
        )
        t_table = tracks_to_table(G)
        lineage_events: list[LineageEvent] = list(G.graph.get("lineage_events", []))
        new_events = [ev for ev in lineage_events if ev.scan_idx_to == scan_idx]

        # 5. Compute causal kinematics & incremental ROI-cached fracture precursors (only current vol in RAM!)
        kin_list = compute_grain_kinematics(t_table)
        prec_candidates, obs_scores = compute_fracture_precursors(
            t_table,
            self.precursor_config,
            volumes={scan_idx: vol},
            label_volumes={scan_idx: labels},
            patch_encoder=self._patch_encoder,
            lineage_events=lineage_events,
            roi_cache=self._roi_cache,
            batch_size=self.batch_size,
        )

        # 6. Select and extrapolate autonomous DFXM zoom targets at scan t
        zoom_targets = select_zoom_targets(
            current_scan_idx=scan_idx,
            scan_name=resolved_name,
            track_table=t_table,
            kinematics=kin_list,
            obs_precursor_scores=obs_scores,
            policy=self.trigger_policy,
            calibration=active_cal,
            detector=self.detector,
            stimulus=stimulus,
            last_triggered_scan_by_lineage=self._cooldown_by_lineage,
        )

        self._graph = G
        self._track_table = t_table
        self._kinematics = kin_list
        self._obs_precursor_scores = obs_scores
        self._precursor_candidates = prec_candidates
        self._all_zoom_targets.extend(zoom_targets)

        n_active = sum(1 for row in t_table if int(row["scan_idx"]) == scan_idx)
        dt = time.perf_counter() - t0
        return StreamingStepResult(
            scan_idx=scan_idx,
            scan_name=resolved_name,
            threshold_raw=raw_thr,
            threshold_causal=causal_thr,
            n_spots=len(spots),
            n_active_tracks=n_active,
            n_lineage_events_total=len(lineage_events),
            new_lineage_events=new_events,
            precursor_candidates=prec_candidates,
            zoom_targets=zoom_targets,
            step_latency_sec=dt,
            stimulus=stimulus,
        )
