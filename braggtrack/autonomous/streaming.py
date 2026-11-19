"""Strictly causal online scan-by-scan tracker with ``tell / ask / report`` agent protocol.

Processes incoming 3D rocking-curve scans ``t = 0, 1, 2, ...`` without future lookahead,
adopting the ``tell()`` / ``ask()`` / ``report()`` agent grammar standardized in synchrotron
autonomous frameworks (``Bluesky-Adaptive`` — Maffettone et al., 2023; ``gpCAM`` — Noack et al., 2021):

- ``tell(scan, ...)``: ingests one new 3D scan with causal trailing-window Otsu thresholding,
  ``float32`` LoG + watershed segmentation, outer-ring pedestal subtraction, batched ``DINOv2``
  encoding, and bounded $O(1)$ sliding-window trajectory DAG updates.
- ``ask(n, acquisition=...)``: queries the Gaussian Process trajectory forecaster and Bayesian
  acquisition policy (``ucb``, ``ei``, ``max_variance``) for high-resolution DFXM zoom setpoints.
- ``tell_zoom_outcome(track_id, outcome)``: optional multi-fidelity hook feeding back measured
  DFXM zoom metrics (e.g. ``darfix`` mosaicity spread) into the ensemble GP.
- ``step(scan, ...)``: single-call convenience wrapper executing ``tell()`` followed by ``ask()``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np

from braggtrack.io.lazy import detector_foreground_threshold_chunked, open_lazy_volume
from braggtrack.io.models import AxisCalibration
from braggtrack.io.nexus import extract_axis_calibration, extract_scan_metadata
from braggtrack.segmentation import (
    detector_foreground_threshold,
    extract_instance_table,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
)
from braggtrack.semantic import DEFAULT_DINO_MODEL, make_multiview_encoder, make_patch_encoder
from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips
from braggtrack.tracking.cost import CostFunction, GeometrySemanticCost, PositionShapeCost, WassersteinBuresCost
from braggtrack.tracking.drift import DriftMode, estimate_slab_shifts_fft
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

from .gp import AcquisitionKind
from .trigger import ZoomTarget, ZoomTriggerPolicy, select_zoom_targets

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.semantic.dino import BackendName


@dataclass(frozen=True)
class StreamingStepResult:
    """Snapshot of online causal tracking and autonomous DFXM zoom targets after scan ``scan_idx``."""

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
    beam_ok: bool = True

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
            "beam_ok": bool(self.beam_ok),
            "stimulus": self.stimulus.to_dict() if self.stimulus is not None else None,
        }


class CausalStreamingTracker:
    """Stateful causal 4D reflection tracker and GP-driven ``tell / ask / report`` DFXM agent."""

    def __init__(
        self,
        *,
        backend: BackendName | None = "auto",
        model_name: str = DEFAULT_DINO_MODEL,
        smooth_window: int = 5,
        bg_sigma: float = 5.0,
        seed_peak_fraction: float = 0.08,
        seed_response_percentile: float = 95.0,
        min_spot_voxels: int = 6,
        anisotropic_segmentation: bool = False,
        calibration: AxisCalibration | None = None,
        detector: DetectorGeometry | None = None,
        cost_model: Literal["position_shape", "wasserstein_bures"] = "position_shape",
        drift_mode: DriftMode = "none",
        cost_alpha: float = 1.0,
        cost_beta: float = 0.0,
        max_cost: float = 900.0,
        max_step_mu: float = 12.0,
        max_step_chi: float = 40.0,
        max_step_d: float = 40.0,
        max_gap: int = 0,
        split_max_distance: float = 25.0,
        precursor_config: PrecursorConfig | None = None,
        trigger_policy: ZoomTriggerPolicy | None = None,
        sliding_window_scans: int | None = 24,
        swmr: bool = False,
        batch_size: int = 32,
        use_patch_encoder: bool = False,
    ) -> None:
        self.smooth_window = max(1, int(smooth_window))
        self.bg_sigma = max(0.0, float(bg_sigma))
        self.seed_peak_fraction = float(seed_peak_fraction)
        self.seed_response_percentile = float(seed_response_percentile)
        self.min_spot_voxels = max(1, int(min_spot_voxels))
        self.anisotropic_segmentation = bool(anisotropic_segmentation)
        self.calibration = calibration
        self.detector = detector
        self.cost_model = cost_model
        self.drift_mode: DriftMode = drift_mode
        self.cost_alpha = float(cost_alpha)
        self.cost_beta = float(cost_beta)
        self.max_cost = float(max_cost)
        self.max_step_mu = float(max_step_mu)
        self.max_step_chi = float(max_step_chi)
        self.max_step_d = float(max_step_d)
        self.batch_size = max(1, int(batch_size))
        self.swmr = bool(swmr)
        self.sliding_window_scans = max(6, int(sliding_window_scans)) if sliding_window_scans is not None else None

        self.gap_config = GapClosingConfig(max_gap=max(0, int(max_gap))) if max_gap > 0 else None
        self.split_merge_config = SplitMergeConfig(max_distance=float(split_max_distance))
        self.precursor_config = precursor_config or PrecursorConfig()
        self.trigger_policy = trigger_policy or ZoomTriggerPolicy()

        self._multiview_encoder = (
            make_multiview_encoder(backend, model_name=model_name) if self.cost_beta > 0.0 else None
        )
        self._patch_encoder = make_patch_encoder(backend, model_name=model_name) if use_patch_encoder else None
        base_cost: CostFunction
        if self.cost_model == "wasserstein_bures":
            base_cost = WassersteinBuresCost(
                bures_weight=1.0,
                intensity_weight=25.0,
                max_step_mu=self.max_step_mu,
                max_step_chi=self.max_step_chi,
                max_step_d=self.max_step_d,
            )
        else:
            base_cost = PositionShapeCost(
                shape_weight=2.0,
                intensity_weight=25.0,
                max_step_mu=self.max_step_mu,
                max_step_chi=self.max_step_chi,
                max_step_d=self.max_step_d,
            )
        self._cost_fn = GeometrySemanticCost(
            base_cost,
            cost_alpha=self.cost_alpha,
            cost_beta=self.cost_beta,
        )

        # Strictly causal state accumulated across scans 0 .. t
        self._raw_thresholds: list[float] = []
        self._causal_thresholds: list[float] = []
        self._nominal_flux_history: list[float] = []
        self._scan_names: list[str] = []
        self._scan_timestamps: dict[int, str | None] = {}
        self._scan_elapsed_s: dict[int, float] = {}
        self._scan_tables: list[list[dict[str, Any]]] = []
        self._prev_vol: np.ndarray | None = None
        self._pair_shifts: dict[int, list[tuple[float, float, float]]] = {}
        self._stimuli: dict[int, ScanStimulusPoint] = {}
        self._roi_cache: dict[tuple[int, int], Any] = {}
        self._cooldown_by_lineage: dict[int, int] = {}
        self._zoom_counts_by_lineage: dict[int, int] = {}
        self._zoom_outcomes: dict[int, tuple[np.ndarray, float]] = {}
        self._step_latencies: list[float] = []
        self._last_beam_ok: bool = True
        self._last_active_cal: AxisCalibration = calibration or AxisCalibration()

        # Sliding-window state preserving global track_ids and root strain anchors in O(1)/step
        self._next_global_tid: int = 1
        self._spot_to_global_tid: dict[tuple[int, int], int] = {}
        self._birth_anchor_by_tid: dict[int, dict[str, Any]] = {}
        self._committed_rows: list[dict[str, Any]] = []
        self._committed_events: list[LineageEvent] = []
        self._committed_up_to_scan: int = -1

        # Latest DAG & derived tables
        self._graph: nx.DiGraph = nx.DiGraph()
        self._window_rows: list[dict[str, Any]] = []
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
        return [*self._committed_rows, *self._window_rows]

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
    ) -> tuple[np.ndarray, float, str, str | None, AxisCalibration]:
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
            with open_lazy_volume(file_path, swmr=self.swmr) as lazy_vol:
                raw_thr = float(detector_foreground_threshold_chunked(lazy_vol, bg_sigma=self.bg_sigma))
                vol = np.asarray(lazy_vol._dataset[()], dtype=np.float32)
            meta = extract_scan_metadata(file_path, swmr=self.swmr)
            ts = timestamp if timestamp is not None else meta.get("start_time")
            cal = (
                self.calibration
                if self.calibration is not None
                else extract_axis_calibration(file_path, swmr=self.swmr)
            )
            if self.calibration is None and cal.calibrated:
                self.calibration = cal
            return vol, raw_thr, (scan_name or default_name), ts, cal

        arr = np.asarray(volume_or_path)
        vol = np.asarray(arr, dtype=np.float32 if arr.dtype.itemsize <= 4 else np.float64)
        raw_thr = float(detector_foreground_threshold(vol, bg_sigma=self.bg_sigma))
        cal = self.calibration if self.calibration is not None else AxisCalibration()
        name = scan_name or f"scan_{len(self._scan_tables) + 1:04d}"
        return vol, raw_thr, name, timestamp, cal

    def _check_beam_flux(self, vol: np.ndarray) -> bool:
        """Return False during synchrotron beam dumps or closed-shutter scans."""
        flux = max(0.0, float(np.sum(vol, dtype=np.float64)))
        if not self._nominal_flux_history:
            if flux > 0.0:
                self._nominal_flux_history.append(flux)
            return flux > 0.0
        baseline = float(np.median(self._nominal_flux_history[-self.smooth_window :]))
        if baseline <= 0.0:
            return True
        ok = (flux / baseline) >= float(self.trigger_policy.min_beam_fraction)
        if ok and flux > 0.0:
            self._nominal_flux_history.append(flux)
        return ok

    def _build_window_dag(self, scan_idx: int) -> tuple[nx.DiGraph, list[dict[str, Any]], list[LineageEvent]]:
        """Build causal DAG over active window ``[win_start, scan_idx]`` with immutable global track IDs."""
        n_total = scan_idx + 1
        win_start = (
            0
            if self.sliding_window_scans is None or n_total <= self.sliding_window_scans
            else n_total - self.sliding_window_scans
        )

        if win_start > 0 and self._committed_up_to_scan < win_start - 1:
            for row in self._window_rows:
                s_i = int(row["scan_idx"])
                if self._committed_up_to_scan < s_i < win_start:
                    self._committed_rows.append(row)
            prev_events: list[LineageEvent] = list(self._graph.graph.get("window_lineage_events", []))
            for ev in prev_events:
                if self._committed_up_to_scan < ev.scan_idx_from < win_start:
                    self._committed_events.append(ev)
            self._committed_up_to_scan = win_start - 1
            stale_keys = [k for k in self._roi_cache if k[0] < win_start - 1]
            for k in stale_keys:
                del self._roi_cache[k]
            stale_shifts = [k for k in self._pair_shifts if k < win_start - 1]
            for k in stale_shifts:
                del self._pair_shifts[k]

        if win_start == 0:
            G = build_tracks(
                self._scan_tables,
                cost_fn=self._cost_fn,
                max_cost=self.max_cost,
                gap_closing_config=self.gap_config,
                split_merge_config=self.split_merge_config,
                scan_timestamps=self._scan_timestamps,
                drift_mode=self.drift_mode,
                precomputed_pair_shifts=self._pair_shifts if self._pair_shifts else None,
                scan_elapsed_s=self._scan_elapsed_s if self._scan_elapsed_s else None,
            )
            w_rows = tracks_to_table(G)
            for row in w_rows:
                s_i = int(row["scan_idx"])
                sp_i = int(row["spot_idx"])
                tid = int(row["track_id"])
                self._spot_to_global_tid[(s_i, sp_i)] = tid
                if tid >= self._next_global_tid:
                    self._next_global_tid = tid + 1
                if tid not in self._birth_anchor_by_tid or s_i < int(self._birth_anchor_by_tid[tid]["scan_idx"]):
                    self._birth_anchor_by_tid[tid] = dict(row)
            w_events = list(G.graph.get("lineage_events", []))
            G.graph["window_lineage_events"] = w_events
            return G, w_rows, w_events

        sub_tables = self._scan_tables[win_start : scan_idx + 1]
        sub_ts = {k - win_start: v for k, v in self._scan_timestamps.items() if k >= win_start}
        sub_shifts = {k - win_start: v for k, v in self._pair_shifts.items() if win_start <= k < scan_idx}
        sub_elapsed = {k - win_start: v for k, v in self._scan_elapsed_s.items() if k >= win_start}
        G_local = build_tracks(
            sub_tables,
            cost_fn=self._cost_fn,
            max_cost=self.max_cost,
            gap_closing_config=self.gap_config,
            split_merge_config=self.split_merge_config,
            scan_timestamps=sub_ts,
            drift_mode=self.drift_mode,
            precomputed_pair_shifts=sub_shifts if sub_shifts else None,
            scan_elapsed_s=sub_elapsed if sub_elapsed else None,
        )

        nodes_by_local_tid: dict[int, list[tuple[int, int, str]]] = {}
        for nid, data in G_local.nodes(data=True):
            ltid = int(data["track_id"])
            nodes_by_local_tid.setdefault(ltid, []).append((int(data["scan_idx"]), int(data["spot_idx"]), nid))

        local_to_global: dict[int, int] = {}
        used_globals: set[int] = set()
        for ltid in sorted(nodes_by_local_tid):
            obs_list = sorted(nodes_by_local_tid[ltid])
            s0_loc, sp0, _ = obs_list[0]
            s0_glob = s0_loc + win_start
            cand = self._spot_to_global_tid.get((s0_glob, sp0))
            if cand is not None and cand not in used_globals:
                gtid = cand
            else:
                gtid = self._next_global_tid
                self._next_global_tid += 1
            used_globals.add(gtid)
            local_to_global[ltid] = gtid

        G_global = nx.DiGraph()
        nid_map: dict[str, str] = {}
        for nid, data in G_local.nodes(data=True):
            d = dict(data)
            s_glob = int(d["scan_idx"]) + win_start
            sp_i = int(d["spot_idx"])
            ltid = int(d["track_id"])
            gtid = local_to_global[ltid]
            gnid = f"s{s_glob}_i{sp_i}"
            nid_map[nid] = gnid
            d["scan_idx"] = s_glob
            d["track_id"] = gtid
            if "parent_track_ids" in d and isinstance(d["parent_track_ids"], list):
                d["parent_track_ids"] = [local_to_global.get(int(p), int(p)) for p in d["parent_track_ids"]]
            elif gtid in self._birth_anchor_by_tid:
                anchor = self._birth_anchor_by_tid[gtid]
                if anchor.get("parent_track_ids"):
                    raw_p = anchor["parent_track_ids"]
                    if isinstance(raw_p, str):
                        d["parent_track_ids"] = [int(x) for x in raw_p.split(";") if x.strip()]
                    elif isinstance(raw_p, list):
                        d["parent_track_ids"] = list(raw_p)
                if anchor.get("lineage_role") and not d.get("lineage_role"):
                    d["lineage_role"] = anchor["lineage_role"]
            G_global.add_node(gnid, **d)
            self._spot_to_global_tid[(s_glob, sp_i)] = gtid

        for u, v, edata in G_local.edges(data=True):
            ed = dict(edata)
            if "parent_track_id" in ed:
                ed["parent_track_id"] = local_to_global.get(int(ed["parent_track_id"]), int(ed["parent_track_id"]))
            if "child_track_id" in ed:
                ed["child_track_id"] = local_to_global.get(int(ed["child_track_id"]), int(ed["child_track_id"]))
            G_global.add_edge(nid_map[u], nid_map[v], **ed)

        remapped_win_events: list[LineageEvent] = []
        for ev in G_local.graph.get("lineage_events", []):
            remapped_win_events.append(
                replace(
                    ev,
                    scan_idx_from=ev.scan_idx_from + win_start,
                    scan_idx_to=ev.scan_idx_to + win_start,
                    parent_track_ids=tuple(local_to_global.get(int(p), int(p)) for p in ev.parent_track_ids),
                    child_track_ids=tuple(local_to_global.get(int(c), int(c)) for c in ev.child_track_ids),
                    parent_nodes=tuple(nid_map.get(n, n) for n in ev.parent_nodes),
                    child_nodes=tuple(nid_map.get(n, n) for n in ev.child_nodes),
                )
            )
        all_events = [*self._committed_events, *remapped_win_events]
        G_global.graph["window_lineage_events"] = remapped_win_events
        G_global.graph["lineage_events"] = all_events
        G_global.graph["gap_closed_count"] = G_local.graph.get("gap_closed_count", 0)

        w_rows = tracks_to_table(G_global)
        for row in w_rows:
            gtid = int(row["track_id"])
            s_i = int(row["scan_idx"])
            if gtid not in self._birth_anchor_by_tid or s_i < int(self._birth_anchor_by_tid[gtid]["scan_idx"]):
                self._birth_anchor_by_tid[gtid] = dict(row)
        return G_global, w_rows, all_events

    def tell(
        self,
        volume_or_path: np.ndarray | str | Path,
        *,
        scan_name: str | None = None,
        timestamp: str | None = None,
        stimulus: ScanStimulusPoint | None = None,
    ) -> StreamingStepResult:
        """Ingest one new 3D scan ``t`` and update causal tracking, kinematics, and precursors."""
        t0 = time.perf_counter()
        scan_idx = len(self._scan_tables)

        vol, raw_thr, resolved_name, resolved_ts, active_cal = self._resolve_volume_and_meta(
            volume_or_path, scan_name, timestamp
        )
        if stimulus is not None and resolved_ts is None and stimulus.timestamp is not None:
            resolved_ts = stimulus.timestamp

        beam_ok = self._check_beam_flux(vol)
        self._last_beam_ok = beam_ok
        self._last_active_cal = active_cal

        if beam_ok or not self._raw_thresholds:
            self._raw_thresholds.append(raw_thr)
        win_lo = max(0, len(self._raw_thresholds) - self.smooth_window)
        causal_thr = float(np.median(self._raw_thresholds[win_lo:]))
        self._causal_thresholds.append(causal_thr)

        seg = segment_classical(
            vol,
            threshold=causal_thr,
            seed_peak_fraction=self.seed_peak_fraction,
            seed_response_percentile=self.seed_response_percentile,
            anisotropic_auto=self.anisotropic_segmentation,
        )
        labels = relabel_sequential(remove_small_objects(seg.labeled_volume, min_size=self.min_spot_voxels))
        spots = extract_instance_table(labels, vol, calibration=active_cal) if beam_ok else []

        if self.drift_mode in ("fft", "fft_kalman"):
            if scan_idx > 0 and self._prev_vol is not None and self._scan_tables[scan_idx - 1]:
                self._pair_shifts[scan_idx - 1], _ = estimate_slab_shifts_fft(
                    self._prev_vol,
                    vol,
                    self._scan_tables[scan_idx - 1],
                    max_step_chi=self.max_step_chi,
                    max_step_d=self.max_step_d,
                    calibration=active_cal,
                )
            self._prev_vol = vol
        else:
            self._prev_vol = None

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
            if stimulus.elapsed_s is not None and math.isfinite(stimulus.elapsed_s):
                self._scan_elapsed_s[scan_idx] = float(stimulus.elapsed_s)

        G, w_rows, lineage_events = self._build_window_dag(scan_idx)
        new_events = [ev for ev in lineage_events if ev.scan_idx_to == scan_idx]

        win_min_scan = min((int(r["scan_idx"]) for r in w_rows), default=scan_idx)
        anchor_rows: list[dict[str, Any]] = []
        if win_min_scan > 0:
            queue = list({int(r["track_id"]) for r in w_rows})
            visited_anc: set[int] = set()
            while queue:
                cur_t = queue.pop()
                if cur_t in visited_anc:
                    continue
                visited_anc.add(cur_t)
                anc = self._birth_anchor_by_tid.get(cur_t)
                if anc is not None:
                    if int(anc["scan_idx"]) < win_min_scan:
                        anchor_rows.append(anc)
                    raw_p = anc.get("parent_track_ids", "")
                    ptids = (
                        [int(x) for x in raw_p.split(";") if x.strip()]
                        if isinstance(raw_p, str)
                        else [int(x) for x in raw_p]
                        if isinstance(raw_p, list)
                        else []
                    )
                    for pt in ptids:
                        if pt not in visited_anc:
                            queue.append(pt)

        kin_list = compute_grain_kinematics(
            [*anchor_rows, *w_rows],
            scan_elapsed_s=self._scan_elapsed_s if self._scan_elapsed_s else None,
        )

        prec_candidates, obs_scores = compute_fracture_precursors(
            w_rows,
            self.precursor_config,
            volumes={scan_idx: vol},
            label_volumes={scan_idx: labels},
            patch_encoder=self._patch_encoder,
            lineage_events=lineage_events,
            roi_cache=self._roi_cache,
            batch_size=self.batch_size,
        )

        self._graph = G
        self._window_rows = w_rows
        self._kinematics = kin_list
        self._obs_precursor_scores = obs_scores
        self._precursor_candidates = prec_candidates

        n_active = sum(1 for row in w_rows if int(row["scan_idx"]) == scan_idx)
        dt = time.perf_counter() - t0
        self._step_latencies.append(dt)

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
            zoom_targets=[],
            step_latency_sec=dt,
            stimulus=stimulus,
            beam_ok=beam_ok,
        )

    def ask(
        self,
        n: int | None = None,
        *,
        acquisition: AcquisitionKind | None = None,
        commit: bool = True,
    ) -> list[ZoomTarget]:
        """Query the GP surrogate and Bayesian acquisition policy for top ``n`` DFXM zoom targets."""
        if not self._scan_tables:
            return []
        scan_idx = len(self._scan_tables) - 1
        targets = select_zoom_targets(
            current_scan_idx=scan_idx,
            scan_name=self._scan_names[scan_idx],
            track_table=self._window_rows,
            kinematics=self._kinematics,
            obs_precursor_scores=self._obs_precursor_scores,
            policy=self.trigger_policy,
            acquisition_override=acquisition,
            max_targets_override=n,
            calibration=self._last_active_cal,
            detector=self.detector,
            stimulus=self._stimuli.get(scan_idx),
            stimuli_by_scan=self._stimuli if self._stimuli else None,
            scan_elapsed_s=self._scan_elapsed_s if self._scan_elapsed_s else None,
            last_triggered_scan_by_lineage=self._cooldown_by_lineage,
            zoom_counts_by_lineage=self._zoom_counts_by_lineage,
            zoom_outcomes=self._zoom_outcomes,
            beam_ok=self._last_beam_ok,
            commit=commit,
        )
        if commit and targets:
            self._all_zoom_targets.extend(targets)
        return targets

    def tell_zoom_outcome(self, track_id: int, outcome: float) -> None:
        """Record a measured high-resolution DFXM zoom scalar (e.g. ``darfix`` mosaicity) for GP conditioning."""
        kin_by_tid = {k.track_id: k for k in self._kinematics}
        kin = kin_by_tid.get(int(track_id))
        if kin is None or not kin.scan_indices:
            return
        pos = len(kin.scan_indices) - 1
        root_tid = int(kin.root_track_id) if kin.root_track_id > 0 else int(track_id)
        l_strain = float(kin.lineage_strain[pos]) if len(kin.lineage_strain) > pos else float(kin.strain[pos])
        s_rate = float(kin.strain_rate[pos]) if len(kin.strain_rate) > pos else 0.0
        aniso = float(kin.anisotropy[pos]) if len(kin.anisotropy) > pos else 1.0

        sc_matching = [s for s in self._obs_precursor_scores if s.track_id == int(track_id)]
        prec_score = float(sc_matching[-1].precursor_score) if sc_matching else 0.0
        subpeaks = int(sc_matching[-1].roi_subpeak_count) if sc_matching else 1

        f_vec = np.array(
            [
                abs(l_strain),
                abs(s_rate),
                math.log1p(max(0.0, aniso - 1.0)),
                float(max(0, subpeaks - 1)),
                prec_score,
            ],
            dtype=np.float64,
        )
        self._zoom_outcomes[root_tid] = (f_vec, float(outcome))

    def report(self) -> dict[str, Any]:
        """Return a concise ``Bluesky-Adaptive`` style status summary of the streaming agent."""
        scan_idx = len(self._scan_tables) - 1
        n_active = sum(1 for row in self._window_rows if int(row["scan_idx"]) == scan_idx) if scan_idx >= 0 else 0
        return {
            "n_scans_processed": len(self._scan_tables),
            "n_active_tracks": n_active,
            "n_zoom_targets_emitted": len(self._all_zoom_targets),
            "n_zoom_outcomes_recorded": len(self._zoom_outcomes),
            "acquisition": self.trigger_policy.acquisition,
            "execution_mode": self.trigger_policy.execution_mode,
            "beam_ok": self._last_beam_ok,
            "mean_step_latency_sec": round(float(np.mean(self._step_latencies)), 4) if self._step_latencies else 0.0,
        }

    def step(
        self,
        volume_or_path: np.ndarray | str | Path,
        *,
        scan_name: str | None = None,
        timestamp: str | None = None,
        stimulus: ScanStimulusPoint | None = None,
    ) -> StreamingStepResult:
        """Convenience wrapper executing ``tell(volume_or_path, ...)`` followed by ``ask()``."""
        t0 = time.perf_counter()
        tell_res = self.tell(volume_or_path, scan_name=scan_name, timestamp=timestamp, stimulus=stimulus)
        targets = self.ask(commit=True)
        total_dt = time.perf_counter() - t0
        self._step_latencies[-1] = total_dt
        return replace(tell_res, zoom_targets=targets, step_latency_sec=total_dt)
