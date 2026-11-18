"""Closed-loop retrospective causal replay benchmark for autonomous DFXM zoom.

Replays a recorded 4D operando diffraction dataset strictly scan-by-scan through
:class:`CausalStreamingTracker` and evaluates whether GP-guided zoom triggers fire
*before* actual ``1 -> 2+`` grain fracture events, whether GP-extrapolated motor
setpoints accurately center the grain at ``t + 1`` (reporting physical setpoint MAE
alongside hit rate), and how ``BraggTrack`` compares against naive beamline trigger
baselines (``random``, ``brightest_intensity``, and ``strain_only``) under an
identical zoom acquisition budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from braggtrack.tracking.kinematics import compute_grain_kinematics
from braggtrack.tracking.precursors import PrecursorConfig

from .streaming import CausalStreamingTracker, StreamingStepResult
from .trigger import ZoomTarget, ZoomTriggerPolicy, clamp_setpoint_bounds, compute_target_geometry

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.models import AxisCalibration
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.semantic.dino import BackendName


@dataclass(frozen=True)
class PolicyReplayMetrics:
    """Quantitative closed-loop zoom performance for one trigger policy."""

    policy_name: str
    n_zooms_fired: int
    n_fracture_lineages_total: int
    n_fracture_lineages_captured_pre_split: int
    fracture_capture_recall: float
    fracture_capture_precision: float
    mean_lead_scans_before_split: float
    mean_lead_voltage_mv: float | None
    setpoint_hit_rate_next_scan: float
    static_centroid_hit_rate_next_scan: float
    mean_setpoint_mae: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "n_zooms_fired": int(self.n_zooms_fired),
            "n_fracture_lineages_total": int(self.n_fracture_lineages_total),
            "n_fracture_lineages_captured_pre_split": int(self.n_fracture_lineages_captured_pre_split),
            "fracture_capture_recall": round(float(self.fracture_capture_recall), 4),
            "fracture_capture_precision": round(float(self.fracture_capture_precision), 4),
            "mean_lead_scans_before_split": round(float(self.mean_lead_scans_before_split), 3),
            "mean_lead_voltage_mv": (
                round(float(self.mean_lead_voltage_mv), 2) if self.mean_lead_voltage_mv is not None else None
            ),
            "setpoint_hit_rate_next_scan": round(float(self.setpoint_hit_rate_next_scan), 4),
            "static_centroid_hit_rate_next_scan": round(float(self.static_centroid_hit_rate_next_scan), 4),
            "mean_setpoint_mae": (
                [round(float(x), 6) for x in self.mean_setpoint_mae] if self.mean_setpoint_mae is not None else None
            ),
        }


@dataclass(frozen=True)
class ReplayBenchmarkResult:
    """Complete causal replay trajectory and baseline comparison table."""

    n_scans: int
    mean_step_latency_sec: float
    max_step_latency_sec: float
    steps: list[StreamingStepResult]
    zoom_targets: list[ZoomTarget]
    policy_comparison: dict[str, PolicyReplayMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scans": int(self.n_scans),
            "mean_step_latency_sec": round(float(self.mean_step_latency_sec), 4),
            "max_step_latency_sec": round(float(self.max_step_latency_sec), 4),
            "n_zoom_targets": len(self.zoom_targets),
            "zoom_targets": [t.to_bliss_command() for t in self.zoom_targets],
            "policy_comparison": {k: v.to_dict() for k, v in self.policy_comparison.items()},
            "steps": [s.to_dict() for s in self.steps],
        }


def _score_trigger_events(
    policy_name: str,
    triggers: list[tuple[int, int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    *,
    split_scan_by_root: dict[int, int],
    obs_by_scan_and_root: dict[tuple[int, int], list[tuple[float, float, float]]],
    voltage_by_scan: dict[int, float],
) -> PolicyReplayMetrics:
    """Evaluate a list of ``(scan_idx, root_tid, cur_c, pred_c, half_w)`` zoom actions."""
    n_frac_total = len(split_scan_by_root)
    captured_roots: dict[int, int] = {}
    pre_split_hits = 0

    next_scan_eval_count = 0
    extrap_hits = 0
    static_hits = 0
    abs_errs: list[tuple[float, float, float]] = []

    for trig_scan, root_tid, cur_c, pred_c, half_w in triggers:
        sp_scan = split_scan_by_root.get(root_tid)
        if sp_scan is not None and trig_scan < sp_scan:
            pre_split_hits += 1
            if root_tid not in captured_roots or trig_scan < captured_roots[root_tid]:
                captured_roots[root_tid] = trig_scan

        next_obs = obs_by_scan_and_root.get((trig_scan + 1, root_tid), [])
        if next_obs:
            next_scan_eval_count += 1
            hm, hc, hd = half_w
            pm, pc, pd = pred_c
            cm, cc, cd = cur_c
            if any(abs(m - pm) <= hm and abs(c - pc) <= hc and abs(d - pd) <= hd for m, c, d in next_obs):
                extrap_hits += 1
            if any(abs(m - cm) <= hm and abs(c - cc) <= hc and abs(d - cd) <= hd for m, c, d in next_obs):
                static_hits += 1
            closest = min(next_obs, key=lambda o: (o[0] - pm) ** 2 + (o[1] - pc) ** 2 + (o[2] - pd) ** 2)
            abs_errs.append((abs(closest[0] - pm), abs(closest[1] - pc), abs(closest[2] - pd)))

    lead_scans = [split_scan_by_root[r] - s for r, s in captured_roots.items()]
    lead_mv: list[float] = []
    for r, s in captured_roots.items():
        sp = split_scan_by_root[r]
        if s in voltage_by_scan and sp in voltage_by_scan:
            lead_mv.append(abs(voltage_by_scan[sp] - voltage_by_scan[s]) * 1000.0)

    n_fired = len(triggers)
    recall = (len(captured_roots) / n_frac_total) if n_frac_total > 0 else 0.0
    precision = (pre_split_hits / n_fired) if n_fired > 0 else 0.0
    mean_lead = float(np.mean(lead_scans)) if lead_scans else 0.0
    mean_mv = float(np.mean(lead_mv)) if lead_mv else None
    extrap_rate = (extrap_hits / next_scan_eval_count) if next_scan_eval_count > 0 else 0.0
    static_rate = (static_hits / next_scan_eval_count) if next_scan_eval_count > 0 else 0.0
    mae = (
        (
            float(np.mean([e[0] for e in abs_errs])),
            float(np.mean([e[1] for e in abs_errs])),
            float(np.mean([e[2] for e in abs_errs])),
        )
        if abs_errs
        else None
    )

    return PolicyReplayMetrics(
        policy_name=policy_name,
        n_zooms_fired=n_fired,
        n_fracture_lineages_total=n_frac_total,
        n_fracture_lineages_captured_pre_split=len(captured_roots),
        fracture_capture_recall=recall,
        fracture_capture_precision=precision,
        mean_lead_scans_before_split=mean_lead,
        mean_lead_voltage_mv=mean_mv,
        setpoint_hit_rate_next_scan=extrap_rate,
        static_centroid_hit_rate_next_scan=static_rate,
        mean_setpoint_mae=mae,
    )


def evaluate_causal_replay(
    scans: Sequence[np.ndarray | str | Path],
    *,
    scan_names: Sequence[str] | None = None,
    stimuli: Sequence[ScanStimulusPoint | None] | None = None,
    calibration: AxisCalibration | None = None,
    detector: DetectorGeometry | None = None,
    backend: BackendName | None = "mock",
    precursor_config: PrecursorConfig | None = None,
    trigger_policy: ZoomTriggerPolicy | None = None,
    zoom_budget: int | None = None,
    seed: int = 0,
) -> ReplayBenchmarkResult:
    """Run strictly causal scan-by-scan replay and benchmark against naive trigger baselines."""
    pol = trigger_policy or ZoomTriggerPolicy()
    tracker = CausalStreamingTracker(
        backend=backend,
        calibration=calibration,
        detector=detector,
        precursor_config=precursor_config,
        trigger_policy=pol,
    )

    steps: list[StreamingStepResult] = []
    causal_pools: list[
        list[
            tuple[
                int,
                float,
                float,
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ]
    ] = []

    for idx, item in enumerate(scans):
        s_name = scan_names[idx] if scan_names is not None and idx < len(scan_names) else None
        stim = stimuli[idx] if stimuli is not None and idx < len(stimuli) else None
        res = tracker.step(item, scan_name=s_name, stimulus=stim)
        steps.append(res)

        kin_by_tid = {k.track_id: k for k in tracker.kinematics}
        pool_t: list[
            tuple[
                int,
                float,
                float,
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ] = []
        for obs in tracker._window_rows:
            if int(obs["scan_idx"]) != idx:
                continue
            role = str(obs.get("lineage_role", ""))
            if pol.exclude_post_split_daughters and role in ("split_child", "split_daughter"):
                continue
            tid = int(obs["track_id"])
            kin = kin_by_tid.get(tid)
            if kin is None or idx not in kin.scan_indices:
                continue
            pos = kin.scan_indices.index(idx)
            if (pos + 1) < pol.min_track_age:
                continue
            root_tid = int(kin.root_track_id) if kin.root_track_id > 0 else tid
            inten = float(obs.get("integrated_intensity", 0.0))
            if inten < pol.min_intensity:
                continue
            l_strain = abs(float(kin.lineage_strain[pos])) if len(kin.lineage_strain) > pos else 0.0
            cur_c, pred_c, half_w, _, _, _ = compute_target_geometry(
                obs,
                kin,
                pos,
                lead_scans=pol.lead_scans,
                window_sigma=pol.window_sigma,
                calibration=tracker.calibration,
                detector=detector,
                max_extrapolation_steps=pol.max_extrapolation_steps,
                scan_elapsed_s=tracker._scan_elapsed_s if tracker._scan_elapsed_s else None,
                use_gp=pol.use_gp_extrapolation,
            )
            pred_c, half_w, _ = clamp_setpoint_bounds(
                pred_c,
                half_w,
                mu_bounds=pol.mu_bounds,
                chi_bounds=pol.chi_bounds,
                d_bounds=pol.d_bounds,
            )
            pool_t.append((root_tid, inten, l_strain, cur_c, pred_c, half_w))
        causal_pools.append(pool_t)

    full_table = tracker.track_table
    final_kin = compute_grain_kinematics(full_table) if full_table else []
    final_kin_by_tid = {k.track_id: k for k in final_kin}
    split_scan_by_root: dict[int, int] = {}
    for ev in tracker.graph.graph.get("lineage_events", []):
        if ev.event_type == "split":
            for ptid in ev.parent_track_ids:
                pk = final_kin_by_tid.get(int(ptid))
                rtid = int(pk.root_track_id) if pk is not None and pk.root_track_id > 0 else int(ptid)
                if rtid not in split_scan_by_root or ev.scan_idx_to < split_scan_by_root[rtid]:
                    split_scan_by_root[rtid] = int(ev.scan_idx_to)

    obs_by_scan_and_root: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for obs in full_table:
        s_i = int(obs["scan_idx"])
        tid = int(obs["track_id"])
        pk = final_kin_by_tid.get(tid)
        rtid = int(pk.root_track_id) if pk is not None and pk.root_track_id > 0 else tid
        obs_by_scan_and_root.setdefault((s_i, rtid), []).append(
            (float(obs["centroid_mu"]), float(obs["centroid_chi"]), float(obs["centroid_d"]))
        )

    voltage_by_scan: dict[int, float] = {}
    if stimuli is not None:
        for idx, stim in enumerate(stimuli):
            if stim is not None and stim.voltage_v is not None:
                voltage_by_scan[idx] = float(stim.voltage_v)

    bt_targets = list(tracker.all_zoom_targets)
    if zoom_budget is not None and zoom_budget >= 0:
        bt_targets = bt_targets[:zoom_budget]

    effective_budget = len(bt_targets) if zoom_budget is None else max(0, int(zoom_budget))

    bt_tuples = [
        (t.scan_idx, t.root_track_id, t.current_centroid, t.predicted_setpoint, t.scan_half_widths) for t in bt_targets
    ]

    rng = np.random.default_rng(seed)

    def _run_baseline(
        mode: str,
    ) -> list[tuple[int, int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
        fired: list[
            tuple[int, int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        ] = []
        cd: dict[int, int] = {}
        for t_idx, pool in enumerate(causal_pools):
            if len(fired) >= effective_budget or not pool:
                continue
            eligible = [p for p in pool if (p[0] not in cd or (t_idx - cd[p[0]]) >= pol.cooldown_scans)]
            if not eligible:
                continue
            if mode == "random":
                choice = eligible[int(rng.integers(0, len(eligible)))]
            elif mode == "brightest_intensity":
                choice = max(eligible, key=lambda p: p[1])
            elif mode == "strain_only":
                choice = max(eligible, key=lambda p: p[2])
            else:
                raise ValueError(f"Unknown baseline mode: {mode}")
            rtid, _, _, cur_c, pred_c, half_w = choice
            cd[rtid] = t_idx
            fired.append((t_idx, rtid, cur_c, pred_c, half_w))
        return fired

    comparison: dict[str, PolicyReplayMetrics] = {
        "braggtrack_autonomous": _score_trigger_events(
            "braggtrack_autonomous",
            bt_tuples,
            split_scan_by_root=split_scan_by_root,
            obs_by_scan_and_root=obs_by_scan_and_root,
            voltage_by_scan=voltage_by_scan,
        ),
    }
    for base_name in ("brightest_intensity", "strain_only", "random"):
        comparison[base_name] = _score_trigger_events(
            base_name,
            _run_baseline(base_name),
            split_scan_by_root=split_scan_by_root,
            obs_by_scan_and_root=obs_by_scan_and_root,
            voltage_by_scan=voltage_by_scan,
        )

    latencies = [s.step_latency_sec for s in steps] if steps else [0.0]
    return ReplayBenchmarkResult(
        n_scans=len(steps),
        mean_step_latency_sec=float(np.mean(latencies)),
        max_step_latency_sec=float(np.max(latencies)),
        steps=steps,
        zoom_targets=bt_targets,
        policy_comparison=comparison,
    )
