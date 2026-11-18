"""Strictly causal online scan-by-scan tracker for live beamline streaming.

Processes incoming 3D rocking-curve scans ``t = 0, 1, 2, ...`` without looking ahead:
- Causal trailing-window median Otsu thresholding (using only scans ``<= t``), with chunked z-slab
  streaming on HDF5 inputs and ``float32`` working volumes.
- Upstream instrument-health weighting $\\mathcal{W}(\\mathbf{s}_t)$ (suppressing beam-dump false positives)
  and Operational Design Domain ($\\mathcal{O}_{\\text{ODD}}$) verification.
- 3D LoG + seeded watershed segmentation, background-pedestal subtraction, Poisson centroid error,
  and standardized rocking-curve kurtosis $g_2$ extraction.
- $O(1)$ bounded sliding-window incremental trajectory DAG updates for ``1,000+`` scan operando runs.
- Automatic next-scan Sim-to-Real Gap ($\\Delta_{\\text{gap}}$, FAL Eq. 4) trajectory auditing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from braggtrack.io.lazy import open_lazy_volume, otsu_threshold_chunked
from braggtrack.io.models import AxisCalibration
from braggtrack.io.nexus import extract_axis_calibration, extract_scan_metadata
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

from .odd import (
    BulkReferencePrior,
    FALActiveLevel,
    SimToRealAuditRecord,
    compute_sim_to_real_gap,
)
from .trigger import ZoomTarget, ZoomTriggerPolicy, select_zoom_targets

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.semantic.dino import BackendName


@dataclass(frozen=True)
class StreamingStepResult:
    """Snapshot of online causal tracking, FAL telemetry, and autonomous DFXM triggers after scan ``scan_idx``."""

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
    instrument_health: float = 1.0
    fal_level_active: FALActiveLevel = "L3"
    within_latency_budget: bool = True
    odd_violations: tuple[str, ...] = ()
    audit_records: list[SimToRealAuditRecord] = field(default_factory=list)

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
            "within_latency_budget": bool(self.within_latency_budget),
            "instrument_health": round(float(self.instrument_health), 4),
            "fal_level_active": self.fal_level_active,
            "odd_violations": list(self.odd_violations),
            "audit_records": [a.to_dict() for a in self.audit_records],
            "stimulus": self.stimulus.to_dict() if self.stimulus is not None else None,
        }


class CausalStreamingTracker:
    """Stateful strictly causal scan-by-scan 4D reflection tracker and FAL Level 3 zoom engine."""

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
        reference_prior: BulkReferencePrior | None = None,
        sliding_window_scans: int | None = 24,
        swmr: bool = False,
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
        self.swmr = bool(swmr)
        self.sliding_window_scans = max(6, int(sliding_window_scans)) if sliding_window_scans is not None else None

        self.gap_config = GapClosingConfig(max_gap=max(0, int(max_gap))) if max_gap > 0 else None
        self.split_merge_config = SplitMergeConfig(max_distance=float(split_max_distance))
        self.precursor_config = precursor_config or PrecursorConfig()
        self.trigger_policy = trigger_policy or ZoomTriggerPolicy()
        self._reference_prior = reference_prior

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
        self._nominal_flux_history: list[float] = []
        self._scan_names: list[str] = []
        self._scan_timestamps: dict[int, str | None] = {}
        self._scan_elapsed_s: dict[int, float] = {}
        self._scan_tables: list[list[dict[str, Any]]] = []
        self._stimuli: dict[int, ScanStimulusPoint] = {}
        self._roi_cache: dict[tuple[int, int], tuple[int, float]] = {}
        self._cooldown_by_lineage: dict[int, int] = {}
        self._zoom_counts_by_lineage: dict[int, int] = {}
        self._pending_zoom_targets: dict[int, list[ZoomTarget]] = {}
        self._audit_records: list[SimToRealAuditRecord] = []
        self._last_step_latency_sec: float = 0.0

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

    @property
    def audit_records(self) -> list[SimToRealAuditRecord]:
        return list(self._audit_records)

    @property
    def reference_prior(self) -> BulkReferencePrior | None:
        return self._reference_prior

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
                raw_thr = float(otsu_threshold_chunked(lazy_vol))
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
        raw_thr = float(otsu_threshold(vol))
        cal = self.calibration if self.calibration is not None else AxisCalibration()
        name = scan_name or f"scan_{len(self._scan_tables) + 1:04d}"
        return vol, raw_thr, name, timestamp, cal

    def _compute_instrument_health(self, vol: np.ndarray, min_health: float) -> float:
        """Compute upstream synchrotron flux/shutter health $\\mathcal{W}(\\mathbf{s}_t) \\in [0, 1]$."""
        flux = max(0.0, float(np.sum(vol, dtype=np.float64)))
        if not self._nominal_flux_history:
            if flux > 0.0:
                self._nominal_flux_history.append(flux)
            return 1.0 if flux > 0.0 else 0.0
        win = self._nominal_flux_history[-self.smooth_window :]
        baseline = float(np.median(win))
        if baseline <= 0.0:
            return 1.0
        ratio = flux / baseline
        health = 1.0 if ratio >= 0.50 else float(np.clip(ratio / 0.50, 0.0, 1.0))
        if health >= min_health and flux > 0.0:
            self._nominal_flux_history.append(flux)
        return health

    def _build_window_dag(self, scan_idx: int) -> tuple[nx.DiGraph, list[dict[str, Any]], list[LineageEvent]]:
        """Build causal DAG over active window ``[win_start, scan_idx]`` with immutable global track IDs."""
        n_total = scan_idx + 1
        win_start = (
            0
            if self.sliding_window_scans is None or n_total <= self.sliding_window_scans
            else n_total - self.sliding_window_scans
        )

        # Commit outgoing scans strictly before win_start from the previous step's window rows/events
        if win_start > 0 and self._committed_up_to_scan < win_start - 1:
            for row in self._window_rows:
                s_i = int(row["scan_idx"])
                if self._committed_up_to_scan < s_i < win_start:
                    self._committed_rows.append(row)
            prev_events: list[LineageEvent] = list(self._graph.graph.get("window_lineage_events", []))
            for ev in prev_events:
                if self._committed_up_to_scan < ev.scan_idx_to < win_start:
                    self._committed_events.append(ev)
            self._committed_up_to_scan = win_start - 1
            # Prune ROI cache older than win_start - 1 so RAM stays O(W)
            stale_keys = [k for k in self._roi_cache if k[0] < win_start - 1]
            for k in stale_keys:
                del self._roi_cache[k]

        if win_start == 0:
            G = build_tracks(
                self._scan_tables,
                cost_fn=self._cost_fn,
                max_cost=self.max_cost,
                gap_closing_config=self.gap_config,
                split_merge_config=self.split_merge_config,
                scan_timestamps=self._scan_timestamps,
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

        # Sliding-window solve on scans [win_start .. scan_idx]
        sub_tables = self._scan_tables[win_start : scan_idx + 1]
        sub_ts = {k - win_start: v for k, v in self._scan_timestamps.items() if k >= win_start}
        G_local = build_tracks(
            sub_tables,
            cost_fn=self._cost_fn,
            max_cost=self.max_cost,
            gap_closing_config=self.gap_config,
            split_merge_config=self.split_merge_config,
            scan_timestamps=sub_ts,
        )

        # Group local nodes by local_tid and map to persistent global_tid
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

        # Rebuild relabeled window graph with global scan_idx and global track_id
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
                LineageEvent(
                    event_type=ev.event_type,
                    scan_idx_from=ev.scan_idx_from + win_start,
                    scan_idx_to=ev.scan_idx_to + win_start,
                    parent_track_ids=[local_to_global.get(int(p), int(p)) for p in ev.parent_track_ids],
                    child_track_ids=[local_to_global.get(int(c), int(c)) for c in ev.child_track_ids],
                    intensity_ratio=ev.intensity_ratio,
                    mean_distance=ev.mean_distance,
                    confidence=ev.confidence,
                    timestamp_from=ev.timestamp_from,
                    timestamp_to=ev.timestamp_to,
                    split_subtype=ev.split_subtype,
                    delta_orientation_norm=ev.delta_orientation_norm,
                    delta_d_norm=ev.delta_d_norm,
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

    def _evaluate_pending_audits(
        self,
        scan_idx: int,
        window_rows: list[dict[str, Any]],
        kin_by_tid: dict[int, GrainKinematics],
        validity_epsilon: float,
    ) -> list[SimToRealAuditRecord]:
        """Evaluate FAL Eq. (4) $\\Delta_{\\text{gap}}$ for zoom predictions targeting ``scan_idx``."""
        pending = self._pending_zoom_targets.pop(scan_idx, [])
        if not pending:
            return []

        obs_at_scan = [r for r in window_rows if int(r["scan_idx"]) == scan_idx]
        if not obs_at_scan:
            return []

        by_tid: dict[int, dict[str, Any]] = {int(r["track_id"]): r for r in obs_at_scan}
        by_root: dict[int, list[dict[str, Any]]] = {}
        for r in obs_at_scan:
            tid = int(r["track_id"])
            gk = kin_by_tid.get(tid)
            rtid = int(gk.root_track_id) if gk is not None and gk.root_track_id > 0 else tid
            by_root.setdefault(rtid, []).append(r)

        records: list[SimToRealAuditRecord] = []
        for tgt in pending:
            matched_obs = by_tid.get(tgt.track_id)
            if matched_obs is None:
                cands = by_root.get(tgt.root_track_id, [])
                if cands:
                    pm, pc, pd = tgt.predicted_setpoint
                    matched_obs = min(
                        cands,
                        key=lambda r: (
                            (float(r["centroid_mu"]) - pm) ** 2
                            + (float(r["centroid_chi"]) - pc) ** 2
                            + (float(r["centroid_d"]) - pd) ** 2
                        ),
                    )
            if matched_obs is None:
                continue
            obs_c = (
                float(matched_obs["centroid_mu"]),
                float(matched_obs["centroid_chi"]),
                float(matched_obs["centroid_d"]),
            )
            d_gap, m2, in_95, in_valid = compute_sim_to_real_gap(
                tgt.predicted_setpoint,
                tgt.predictive_std,
                obs_c,
                validity_epsilon=validity_epsilon,
            )
            rec = SimToRealAuditRecord(
                source_scan_idx=tgt.scan_idx,
                eval_scan_idx=scan_idx,
                track_id=tgt.track_id,
                root_track_id=tgt.root_track_id,
                execution_mode=tgt.execution_mode,
                actuated=tgt.actuate,
                hypothesis=tgt.hypothesis,
                predicted_setpoint=tgt.predicted_setpoint,
                predictive_std=tgt.predictive_std,
                observed_centroid=obs_c,
                scan_half_widths=tgt.scan_half_widths,
                setpoint_hit=tgt.contains_point(*obs_c),
                delta_gap=d_gap,
                mahalanobis_sq=m2,
                within_95pct_ci=in_95,
                within_validity_domain=in_valid,
            )
            records.append(rec)
            self._audit_records.append(rec)
        return records

    def step(
        self,
        volume_or_path: np.ndarray | str | Path,
        *,
        scan_name: str | None = None,
        timestamp: str | None = None,
        stimulus: ScanStimulusPoint | None = None,
    ) -> StreamingStepResult:
        """Ingest one new 3D scan ``t`` causally and emit tracking + FAL Level 3 zoom targets."""
        t0 = time.perf_counter()
        scan_idx = len(self._scan_tables)
        odd = self.trigger_policy.resolved_odd()

        vol, raw_thr, resolved_name, resolved_ts, active_cal = self._resolve_volume_and_meta(
            volume_or_path, scan_name, timestamp
        )
        if stimulus is not None and resolved_ts is None and stimulus.timestamp is not None:
            resolved_ts = stimulus.timestamp

        # Upstream instrument health W(s_t) & ODD environment verification
        health = self._compute_instrument_health(vol, min_health=odd.min_instrument_health)
        env_ok, violations = odd.check_sample_environment(stimulus)
        if health < odd.min_instrument_health:
            violations.append(f"instrument_health_low={health:.3f}<{odd.min_instrument_health:.3f}")

        prev_lat_ok, _ = odd.evaluate_latency_budget(self._last_step_latency_sec)
        if not env_ok or health < odd.min_instrument_health:
            fal_active: FALActiveLevel = "L1_halt"
        elif not prev_lat_ok:
            fal_active = "L2_fallback"
            violations.append(f"prev_step_latency_exceeded={self._last_step_latency_sec:.3f}s")
        else:
            fal_active = "L3"

        # 1. Strictly causal trailing-window Otsu threshold (scans max(0, t - W + 1) .. t only!)
        if health >= odd.min_instrument_health or not self._raw_thresholds:
            self._raw_thresholds.append(raw_thr)
        win_lo = max(0, len(self._raw_thresholds) - self.smooth_window)
        causal_thr = float(np.median(self._raw_thresholds[win_lo:]))
        self._causal_thresholds.append(causal_thr)

        # 2. Classical 3D LoG + watershed segmentation & calibrated feature + kurtosis extraction
        seg = segment_classical(
            vol,
            threshold=causal_thr,
            seed_peak_fraction=self.seed_peak_fraction,
            seed_response_percentile=self.seed_response_percentile,
        )
        labels = relabel_sequential(remove_small_objects(seg.labeled_volume, min_size=self.min_spot_voxels))
        spots = (
            extract_instance_table(labels, vol, calibration=active_cal) if health >= odd.min_instrument_health else []
        )

        # Calibrate BulkReferencePrior on first nominal scan if not explicitly supplied
        if self._reference_prior is None and spots and health >= odd.min_instrument_health:
            self._reference_prior = BulkReferencePrior.from_spots(spots)

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
            if stimulus.elapsed_s is not None and math.isfinite(stimulus.elapsed_s):
                self._scan_elapsed_s[scan_idx] = float(stimulus.elapsed_s)

        # 4. Build bounded O(1) sliding-window causal trajectory DAG
        G, w_rows, lineage_events = self._build_window_dag(scan_idx)
        new_events = [ev for ev in lineage_events if ev.scan_idx_to == scan_idx]

        # Prepend lightweight historical birth/ancestor anchors for exact root-referenced strain in O(W)
        win_min_scan = min((int(r["scan_idx"]) for r in w_rows), default=scan_idx)
        anchor_rows: list[dict[str, Any]] = []
        if win_min_scan > 0:
            needed_tids = {int(r["track_id"]) for r in w_rows}
            # Walk parent links in _birth_anchor_by_tid to include root ancestors
            queue = list(needed_tids)
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

        kin_input = [*anchor_rows, *w_rows]
        kin_list = compute_grain_kinematics(
            kin_input,
            scan_elapsed_s=self._scan_elapsed_s if self._scan_elapsed_s else None,
        )
        kin_by_tid = {k.track_id: k for k in kin_list}

        # Evaluate FAL Eq. (4) Sim-to-Real Gap for previous zoom targets scheduled for scan_idx
        step_audits = self._evaluate_pending_audits(
            scan_idx,
            w_rows,
            kin_by_tid,
            validity_epsilon=odd.validity_epsilon,
        )

        # 5. Compute incremental ROI-cached fracture precursors over active window (O(W))
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

        # 6. Select and extrapolate autonomous DFXM zoom targets at scan t
        zoom_targets = select_zoom_targets(
            current_scan_idx=scan_idx,
            scan_name=resolved_name,
            track_table=w_rows,
            kinematics=kin_list,
            obs_precursor_scores=obs_scores,
            policy=self.trigger_policy,
            calibration=active_cal,
            detector=self.detector,
            stimulus=stimulus,
            last_triggered_scan_by_lineage=self._cooldown_by_lineage,
            zoom_counts_by_lineage=self._zoom_counts_by_lineage,
            reference_prior=self._reference_prior,
            instrument_health=health,
            fal_level=fal_active,
        )

        if zoom_targets:
            eval_offset = max(1, round(float(self.trigger_policy.lead_scans)))
            self._pending_zoom_targets.setdefault(scan_idx + eval_offset, []).extend(zoom_targets)

        self._graph = G
        self._window_rows = w_rows
        self._kinematics = kin_list
        self._obs_precursor_scores = obs_scores
        self._precursor_candidates = prec_candidates
        self._all_zoom_targets.extend(zoom_targets)

        n_active = sum(1 for row in w_rows if int(row["scan_idx"]) == scan_idx)
        dt = time.perf_counter() - t0
        self._last_step_latency_sec = dt
        within_lat, _ = odd.evaluate_latency_budget(dt)

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
            instrument_health=health,
            fal_level_active=fal_active,
            within_latency_budget=within_lat,
            odd_violations=tuple(violations),
            audit_records=step_audits,
        )
