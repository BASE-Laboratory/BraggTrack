"""Autonomous DFXM "Scout-and-Zoom" trigger policy, GP setpoint forecasting, and BLISS output.

Evaluates active tracked reflections at scan ``t`` using strictly causal kinematics,
heteroscedastic Gaussian Process (GP) trajectory forecasting (:mod:`braggtrack.autonomous.gp`),
and standard Bayesian acquisition functions (``ucb``, ``ei``, ``max_variance``) to select
high-priority candidate grains for zoomed Dark-Field X-ray Microscopy (DFXM) acquisitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .gp import AcquisitionKind, forecast_grain_trajectory_gp, score_candidates_gp

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.models import AxisCalibration
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.tracking.kinematics import GrainKinematics
    from braggtrack.tracking.precursors import ObservationPrecursorScore

ExecutionMode = Literal["replay", "shadow", "watch", "active"]


def clamp_setpoint_bounds(
    pred: tuple[float, float, float],
    half_widths: tuple[float, float, float],
    *,
    mu_bounds: tuple[float, float] | None = None,
    chi_bounds: tuple[float, float] | None = None,
    d_bounds: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], bool]:
    """Clamp ``(mu, chi, d)`` setpoints and scan intervals to diffractometer soft limits."""
    pm, pc, pd = pred
    hm, hc, hd = half_widths
    clamped = False

    def _clamp_axis(val: float, hw: float, bounds: tuple[float, float] | None) -> tuple[float, float]:
        nonlocal clamped
        if bounds is None:
            return val, hw
        lo, hi = min(bounds[0], bounds[1]), max(bounds[0], bounds[1])
        new_val = min(max(val, lo), hi)
        if abs(new_val - val) > 1e-12:
            clamped = True
        max_span_half = 0.5 * (hi - lo)
        new_hw = min(hw, max_span_half)
        if new_val - new_hw < lo - 1e-12:
            new_val = lo + new_hw
            clamped = True
        if new_val + new_hw > hi + 1e-12:
            new_val = hi - new_hw
            clamped = True
        new_val = min(max(new_val, lo), hi)
        return new_val, max(0.0, new_hw)

    pm, hm = _clamp_axis(pm, hm, mu_bounds)
    pc, hc = _clamp_axis(pc, hc, chi_bounds)
    pd, hd = _clamp_axis(pd, hd, d_bounds)
    return (pm, pc, pd), (hm, hc, hd), clamped


@dataclass(frozen=True)
class ZoomTriggerPolicy:
    """Bayesian acquisition settings, thresholds, and practical safety bounds for DFXM zoom."""

    min_precursor_score: float = 0.40
    min_persistence: int = 2
    min_subpeaks: int = 1
    min_strain_rate: float = 0.0
    min_intensity: float = 20.0
    min_track_age: int = 2
    min_assignment_margin: float = 0.0
    cooldown_scans: int = 3
    max_triggers_per_scan: int = 2
    lead_scans: float = 1.0
    window_sigma: float = 3.5
    exclude_post_split_daughters: bool = True
    acquisition: AcquisitionKind = "ucb"
    ucb_beta: float = 1.96
    ei_xi: float = 0.01
    use_gp_extrapolation: bool = True
    execution_mode: ExecutionMode = "replay"
    mu_bounds: tuple[float, float] | None = None
    chi_bounds: tuple[float, float] | None = None
    d_bounds: tuple[float, float] | None = None
    voltage_bounds_v: tuple[float, float] | None = None
    max_zooms_per_lineage: int | None = None
    max_extrapolation_steps: float = 12.0
    min_beam_fraction: float = 0.20
    exclude_border_truncated: bool = False


@dataclass(frozen=True)
class ZoomTarget:
    """Actionable DFXM zoom acquisition target emitted after causal scan ``scan_idx``."""

    scan_idx: int
    scan_name: str
    track_id: int
    root_track_id: int
    priority_rank: int
    priority_score: float
    precursor_score: float
    persistence_scans: int
    roi_subpeak_count: int
    anisotropy: float
    lineage_strain: float
    strain_err: float
    strain_rate: float
    assignment_margin: float
    current_centroid: tuple[float, float, float]
    predicted_setpoint: tuple[float, float, float]
    scan_half_widths: tuple[float, float, float]
    centroid_err: tuple[float, float, float]
    lead_scans: float
    trigger_reasons: tuple[str, ...]
    scattering_angles: dict[str, float] | None = None
    operando_state: dict[str, Any] | None = None
    predictive_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acquisition_kind: AcquisitionKind = "ucb"
    gp_posterior_mean: float = 0.0
    gp_posterior_std: float = 0.0
    execution_mode: ExecutionMode = "replay"
    actuate: bool = False

    def contains_point(self, mu: float, chi: float, d: float) -> bool:
        """Return True if ``(mu, chi, d)`` lies inside ``predicted_setpoint ± scan_half_widths``."""
        pm, pc, pd = self.predicted_setpoint
        hm, hc, hd = self.scan_half_widths
        return abs(mu - pm) <= hm and abs(chi - pc) <= hc and abs(d - pd) <= hd

    def to_bliss_command(self) -> dict[str, Any]:
        """Format as an ESRF BLISS / Ewoks / Bluesky machine-readable DFXM zoom command payload."""
        pm, pc, pd = self.predicted_setpoint
        hm, hc, hd = self.scan_half_widths
        em, ec, ed = self.centroid_err
        sm, sc, sd = self.predictive_std
        cmd: dict[str, Any] = {
            "schema_version": "bliss_dfxm_zoom.v1",
            "action": "dfxm_high_res_zoom",
            "actuate": bool(self.actuate),
            "execution_mode": self.execution_mode,
            "acquisition": self.acquisition_kind,
            "source_scan_idx": int(self.scan_idx),
            "source_scan_name": self.scan_name,
            "track_id": int(self.track_id),
            "root_track_id": int(self.root_track_id),
            "priority_rank": int(self.priority_rank),
            "priority_score": round(float(self.priority_score), 4),
            "lead_scans": round(float(self.lead_scans), 3),
            "target_motors": {
                "mu": round(float(pm), 6),
                "chi": round(float(pc), 6),
                "obpitch_or_2theta": round(float(pd), 6),
            },
            "scan_intervals": {
                "mu": [round(float(pm - hm), 6), round(float(pm + hm), 6)],
                "chi": [round(float(pc - hc), 6), round(float(pc + hc), 6)],
                "obpitch_or_2theta": [round(float(pd - hd), 6), round(float(pd + hd), 6)],
            },
            "scan_half_widths": {
                "mu": round(float(hm), 6),
                "chi": round(float(hc), 6),
                "obpitch_or_2theta": round(float(hd), 6),
            },
            "uncertainty_1sigma": {
                "centroid_mu_err": round(float(em), 6),
                "centroid_chi_err": round(float(ec), 6),
                "centroid_d_err": round(float(ed), 6),
                "gp_predictive_mu_std": round(float(sm), 6),
                "gp_predictive_chi_std": round(float(sc), 6),
                "gp_predictive_d_std": round(float(sd), 6),
                "strain_err": round(float(self.strain_err), 7),
            },
            "diagnostics": {
                "current_centroid": [round(float(x), 6) for x in self.current_centroid],
                "gp_posterior_mean": round(float(self.gp_posterior_mean), 4),
                "gp_posterior_std": round(float(self.gp_posterior_std), 4),
                "precursor_score": round(float(self.precursor_score), 4),
                "persistence_scans": int(self.persistence_scans),
                "roi_subpeak_count": int(self.roi_subpeak_count),
                "anisotropy": round(float(self.anisotropy), 4),
                "lineage_strain": round(float(self.lineage_strain), 6),
                "strain_rate": round(float(self.strain_rate), 6),
                "assignment_margin": (
                    round(float(self.assignment_margin), 4) if math.isfinite(self.assignment_margin) else "inf"
                ),
            },
            "trigger_reasons": list(self.trigger_reasons),
        }
        if self.scattering_angles is not None:
            cmd["crystallographic_setpoint"] = dict(self.scattering_angles)
        if self.operando_state is not None:
            cmd["operando_state"] = dict(self.operando_state)
        return cmd

    def to_dict(self) -> dict[str, Any]:
        return self.to_bliss_command()


def _trailing_persistence(
    track_obs_scores: list[ObservationPrecursorScore],
    current_scan_idx: int,
    threshold: float,
) -> int:
    """Return contiguous causal run length ending at ``current_scan_idx`` with score >= threshold."""
    relevant = sorted(
        (s for s in track_obs_scores if s.scan_idx <= current_scan_idx),
        key=lambda s: s.scan_idx,
    )
    if not relevant or relevant[-1].scan_idx != current_scan_idx:
        return 0
    run = 0
    expected_scan = current_scan_idx
    for sc in reversed(relevant):
        if sc.scan_idx != expected_scan or sc.precursor_score < threshold:
            break
        run += 1
        expected_scan -= 1
    return run


def compute_target_geometry(
    obs: dict[str, Any],
    kin: GrainKinematics | None,
    obs_pos_in_kin: int,
    *,
    lead_scans: float = 1.0,
    window_sigma: float = 3.5,
    calibration: AxisCalibration | None = None,
    detector: DetectorGeometry | None = None,
    max_extrapolation_steps: float | None = None,
    scan_elapsed_s: dict[int, float] | None = None,
    use_gp: bool = True,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    dict[str, float] | None,
]:
    """Forecast ``(mu, chi, d)`` setpoint, scan half-widths, centroid errors, GP stds, and angles."""
    cur_mu = float(obs["centroid_mu"])
    cur_chi = float(obs["centroid_chi"])
    cur_d = float(obs["centroid_d"])

    step_mu = abs(float(calibration.mu_step)) if calibration is not None else float(obs.get("step_mu", 1.0))
    step_chi = abs(float(calibration.chi_step)) if calibration is not None else float(obs.get("step_chi", 1.0))
    step_d = abs(float(calibration.d_step)) if calibration is not None else float(obs.get("step_d", 1.0))

    if kin is not None and 0 <= obs_pos_in_kin < len(kin.scan_indices):
        end = obs_pos_in_kin + 1
        s_indices = kin.scan_indices[:end]
        err_mu = float(kin.centroid_mu_err[obs_pos_in_kin]) if len(kin.centroid_mu_err) > obs_pos_in_kin else 0.0
        err_chi = float(kin.centroid_chi_err[obs_pos_in_kin]) if len(kin.centroid_chi_err) > obs_pos_in_kin else 0.0
        err_d = float(kin.centroid_d_err[obs_pos_in_kin]) if len(kin.centroid_d_err) > obs_pos_in_kin else 0.0

        if use_gp and len(s_indices) >= 2:
            if scan_elapsed_s and all(int(si) in scan_elapsed_s for si in s_indices):
                t_hist = np.asarray([float(scan_elapsed_s[int(si)]) for si in s_indices], dtype=np.float64)
                diffs = np.diff(t_hist)
                dt_step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
            else:
                t_hist = np.asarray(s_indices, dtype=np.float64)
                dt_step = 1.0

            target_t = float(t_hist[-1]) + float(lead_scans) * dt_step
            e_mu = (
                np.asarray(kin.centroid_mu_err[:end], dtype=np.float64)
                if len(kin.centroid_mu_err) >= end
                else np.zeros(end, dtype=np.float64)
            )
            e_chi = (
                np.asarray(kin.centroid_chi_err[:end], dtype=np.float64)
                if len(kin.centroid_chi_err) >= end
                else np.zeros(end, dtype=np.float64)
            )
            e_d = (
                np.asarray(kin.centroid_d_err[:end], dtype=np.float64)
                if len(kin.centroid_d_err) >= end
                else np.zeros(end, dtype=np.float64)
            )
            gp_out = forecast_grain_trajectory_gp(
                t_hist,
                np.asarray(kin.centroid_mu[:end], dtype=np.float64),
                np.asarray(kin.centroid_chi[:end], dtype=np.float64),
                np.asarray(kin.centroid_d[:end], dtype=np.float64),
                e_mu,
                e_chi,
                e_d,
                target_t,
                step_floors=(step_mu, step_chi, step_d),
            )
            dm = gp_out.mean[0] - cur_mu
            dc = gp_out.mean[1] - cur_chi
            dd = gp_out.mean[2] - cur_d
            gp_std = gp_out.std
        else:
            v_mu = float(kin.rate_mu[obs_pos_in_kin]) if len(kin.rate_mu) > obs_pos_in_kin else 0.0
            v_chi = float(kin.rate_chi[obs_pos_in_kin]) if len(kin.rate_chi) > obs_pos_in_kin else 0.0
            v_d = float(kin.rate_d[obs_pos_in_kin]) if len(kin.rate_d) > obs_pos_in_kin else 0.0
            dm = float(lead_scans) * v_mu
            dc = float(lead_scans) * v_chi
            dd = float(lead_scans) * v_d
            gp_std = (
                max(err_mu, 0.35 * step_mu),
                max(err_chi, 0.35 * step_chi),
                max(err_d, 0.35 * step_d),
            )
    else:
        err_mu = float(obs.get("centroid_mu_err", 0.0))
        err_chi = float(obs.get("centroid_chi_err", 0.0))
        err_d = float(obs.get("centroid_d_err", 0.0))
        dm = dc = dd = 0.0
        gp_std = (
            max(err_mu, 0.35 * step_mu),
            max(err_chi, 0.35 * step_chi),
            max(err_d, 0.35 * step_d),
        )

    if max_extrapolation_steps is not None and max_extrapolation_steps > 0.0:
        cap_m = float(max_extrapolation_steps) * max(step_mu, 1e-12)
        cap_c = float(max_extrapolation_steps) * max(step_chi, 1e-12)
        cap_d = float(max_extrapolation_steps) * max(step_d, 1e-12)
        dm = min(max(dm, -cap_m), cap_m)
        dc = min(max(dc, -cap_c), cap_c)
        dd = min(max(dd, -cap_d), cap_d)

    pred_mu = cur_mu + dm
    pred_chi = cur_chi + dc
    pred_d = cur_d + dd

    std_mu = math.sqrt(max(float(obs.get("cov_zz", obs.get("cov_mu_mu", 0.0))), 0.0))
    std_chi = math.sqrt(max(float(obs.get("cov_xx", obs.get("cov_chi_chi", 0.0))), 0.0))
    std_d = math.sqrt(max(float(obs.get("cov_yy", obs.get("cov_d_d", 0.0))), 0.0))

    # Rocking window combines intrinsic reflection spread and GP posterior setpoint uncertainty
    half_mu = max(window_sigma * math.sqrt(std_mu * std_mu + gp_std[0] * gp_std[0]), 1.5 * step_mu)
    half_chi = max(window_sigma * math.sqrt(std_chi * std_chi + gp_std[1] * gp_std[1]), 1.5 * step_chi)
    half_d = max(window_sigma * math.sqrt(std_d * std_d + gp_std[2] * gp_std[2]), 1.5 * step_d)

    scattering: dict[str, float] | None = None
    if detector is not None:
        if calibration is not None and calibration.calibrated:
            z_idx = (pred_mu - calibration.mu_origin) / (calibration.mu_step if calibration.mu_step != 0 else 1.0)
            x_px = (pred_chi - calibration.chi_origin) / (calibration.chi_step if calibration.chi_step != 0 else 1.0)
            y_px = (pred_d - calibration.d_origin) / (calibration.d_step if calibration.d_step != 0 else 1.0)
        else:
            z_idx, y_px, x_px = pred_mu, pred_d, pred_chi
        scattering = detector.spot_to_scattering(z_idx, y_px, x_px)

    return (
        (cur_mu, cur_chi, cur_d),
        (pred_mu, pred_chi, pred_d),
        (half_mu, half_chi, half_d),
        (err_mu, err_chi, err_d),
        gp_std,
        scattering,
    )


def select_zoom_targets(
    *,
    current_scan_idx: int,
    scan_name: str,
    track_table: list[dict[str, Any]],
    kinematics: list[GrainKinematics],
    obs_precursor_scores: list[ObservationPrecursorScore],
    policy: ZoomTriggerPolicy | None = None,
    acquisition_override: AcquisitionKind | None = None,
    max_targets_override: int | None = None,
    calibration: AxisCalibration | None = None,
    detector: DetectorGeometry | None = None,
    stimulus: ScanStimulusPoint | None = None,
    scan_elapsed_s: dict[int, float] | None = None,
    last_triggered_scan_by_lineage: dict[int, int] | None = None,
    zoom_counts_by_lineage: dict[int, int] | None = None,
    zoom_outcomes: dict[int, tuple[np.ndarray, float]] | None = None,
    beam_ok: bool = True,
    commit: bool = True,
) -> list[ZoomTarget]:
    """Select, rank via Bayesian GP acquisition, and extrapolate DFXM zoom targets."""
    pol = policy or ZoomTriggerPolicy()
    if not beam_ok:
        return []

    if (
        pol.voltage_bounds_v is not None
        and stimulus is not None
        and stimulus.voltage_v is not None
        and math.isfinite(stimulus.voltage_v)
    ):
        v_lo, v_hi = min(pol.voltage_bounds_v), max(pol.voltage_bounds_v)
        if not (v_lo <= float(stimulus.voltage_v) <= v_hi):
            return []

    acq_kind: AcquisitionKind = acquisition_override if acquisition_override is not None else pol.acquisition
    limit = int(max_targets_override) if max_targets_override is not None else max(0, int(pol.max_triggers_per_scan))
    if limit <= 0:
        return []

    kin_by_tid = {k.track_id: k for k in kinematics}
    scores_by_tid: dict[int, list[ObservationPrecursorScore]] = {}
    score_at_current: dict[int, ObservationPrecursorScore] = {}
    for sc in obs_precursor_scores:
        scores_by_tid.setdefault(sc.track_id, []).append(sc)
        if sc.scan_idx == current_scan_idx:
            score_at_current[sc.track_id] = sc

    cooldown_map = last_triggered_scan_by_lineage if last_triggered_scan_by_lineage is not None else {}
    dose_map = zoom_counts_by_lineage if zoom_counts_by_lineage is not None else {}
    operando_dict = stimulus.to_dict() if stimulus is not None else None
    should_actuate = bool(pol.execution_mode == "active")
    has_supervised_outcomes = bool(zoom_outcomes and len(zoom_outcomes) > 0)

    eligible: list[dict[str, Any]] = []
    for obs in track_table:
        if int(obs["scan_idx"]) != current_scan_idx:
            continue
        tid = int(obs["track_id"])
        role = str(obs.get("lineage_role", ""))
        if pol.exclude_post_split_daughters and role in ("split_child", "split_daughter"):
            continue
        if pol.exclude_border_truncated and bool(obs.get("touches_rocking_border", False)):
            continue
        if float(obs.get("integrated_intensity", 0.0)) < pol.min_intensity:
            continue
        margin = float(obs.get("assignment_margin", float("inf")))
        if margin < pol.min_assignment_margin:
            continue

        kin = kin_by_tid.get(tid)
        if kin is None or current_scan_idx not in kin.scan_indices:
            continue
        pos = kin.scan_indices.index(current_scan_idx)
        if (pos + 1) < pol.min_track_age:
            continue

        root_tid = int(kin.root_track_id) if kin.root_track_id > 0 else tid
        prev_trig = cooldown_map.get(root_tid)
        if prev_trig is not None and (current_scan_idx - prev_trig) < pol.cooldown_scans:
            continue
        if pol.max_zooms_per_lineage is not None and dose_map.get(root_tid, 0) >= int(pol.max_zooms_per_lineage):
            continue

        sc_now = score_at_current.get(tid)
        prec_score = float(sc_now.precursor_score) if sc_now is not None else 0.0
        subpeaks = int(sc_now.roi_subpeak_count) if sc_now is not None else 1
        aniso = float(sc_now.anisotropy) if sc_now is not None else float(kin.anisotropy[pos])
        pers = _trailing_persistence(scores_by_tid.get(tid, []), current_scan_idx, pol.min_precursor_score)
        s_rate = float(kin.strain_rate[pos]) if len(kin.strain_rate) > pos else 0.0

        if abs(s_rate) < pol.min_strain_rate or subpeaks < pol.min_subpeaks:
            continue

        # In exploitation-aware modes without external DFXM zoom outcomes, gate on precursor persistence;
        # in pure exploration ('max_variance') or with supervised zoom outcomes, score all eligible tracks.
        if (
            acq_kind in ("ucb", "ei")
            and not has_supervised_outcomes
            and (prec_score < pol.min_precursor_score or pers < pol.min_persistence)
        ):
            continue

        l_strain = float(kin.lineage_strain[pos]) if len(kin.lineage_strain) > pos else float(kin.strain[pos])
        s_err = float(kin.lineage_strain_err[pos]) if len(kin.lineage_strain_err) > pos else 0.0

        track_scores = sorted(
            (s for s in scores_by_tid.get(tid, []) if s.scan_idx <= current_scan_idx),
            key=lambda s: s.scan_idx,
        )
        if scan_elapsed_s and all(int(s.scan_idx) in scan_elapsed_s for s in track_scores):
            s_times = np.asarray([float(scan_elapsed_s[int(s.scan_idx)]) for s in track_scores], dtype=np.float64)
        else:
            s_times = np.asarray([float(s.scan_idx) for s in track_scores], dtype=np.float64)
        s_hist = np.asarray([float(s.precursor_score) for s in track_scores], dtype=np.float64)

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

        eligible.append(
            {
                "obs": obs,
                "kin": kin,
                "pos": pos,
                "tid": tid,
                "root_tid": root_tid,
                "prec_score": prec_score,
                "subpeaks": subpeaks,
                "aniso": aniso,
                "pers": pers,
                "s_rate": s_rate,
                "l_strain": l_strain,
                "s_err": s_err,
                "margin": margin,
                "score_times": s_times,
                "score_history": s_hist,
                "feature_vec": f_vec,
            }
        )

    if not eligible:
        return []

    gp_mu, gp_std, acq_scores = score_candidates_gp(
        eligible,
        acquisition=acq_kind,
        beta=pol.ucb_beta,
        xi=pol.ei_xi,
        lead_scans=pol.lead_scans,
        zoom_outcomes=zoom_outcomes,
    )

    order = sorted(range(len(eligible)), key=lambda i: (-float(acq_scores[i]), int(eligible[i]["tid"])))
    selected: list[ZoomTarget] = []
    seen_roots: set[int] = set()

    for idx in order:
        if len(selected) >= limit:
            break
        rec = eligible[idx]
        root_tid = int(rec["root_tid"])
        if root_tid in seen_roots:
            continue
        seen_roots.add(root_tid)

        cur_c, pred_c, half_w, c_err, pred_std, scat = compute_target_geometry(
            rec["obs"],
            rec["kin"],
            int(rec["pos"]),
            lead_scans=pol.lead_scans,
            window_sigma=pol.window_sigma,
            calibration=calibration,
            detector=detector,
            max_extrapolation_steps=pol.max_extrapolation_steps,
            scan_elapsed_s=scan_elapsed_s,
            use_gp=pol.use_gp_extrapolation,
        )
        pred_c, half_w, was_clamped = clamp_setpoint_bounds(
            pred_c,
            half_w,
            mu_bounds=pol.mu_bounds,
            chi_bounds=pol.chi_bounds,
            d_bounds=pol.d_bounds,
        )

        reasons: list[str] = [
            f"acquisition={acq_kind}({float(acq_scores[idx]):.3f})",
            f"gp_posterior={float(gp_mu[idx]):.3f}+/-{float(gp_std[idx]):.3f}",
        ]
        if float(rec["prec_score"]) >= pol.min_precursor_score:
            reasons.append(f"precursor_score={float(rec['prec_score']):.3f}")
        if int(rec["pers"]) >= 1:
            reasons.append(f"persistence={int(rec['pers'])}_scans")
        if int(rec["subpeaks"]) >= 2:
            reasons.append(f"bimodal_subpeaks={int(rec['subpeaks'])}")
        if float(rec["aniso"]) >= 2.0:
            reasons.append(f"mosaicity_anisotropy={float(rec['aniso']):.2f}")
        if was_clamped:
            reasons.append("clamped_to_motor_bounds")

        if commit:
            if last_triggered_scan_by_lineage is not None:
                last_triggered_scan_by_lineage[root_tid] = current_scan_idx
            if zoom_counts_by_lineage is not None:
                zoom_counts_by_lineage[root_tid] = zoom_counts_by_lineage.get(root_tid, 0) + 1

        selected.append(
            ZoomTarget(
                scan_idx=current_scan_idx,
                scan_name=scan_name,
                track_id=int(rec["tid"]),
                root_track_id=root_tid,
                priority_rank=len(selected) + 1,
                priority_score=float(acq_scores[idx]),
                precursor_score=float(rec["prec_score"]),
                persistence_scans=int(rec["pers"]),
                roi_subpeak_count=int(rec["subpeaks"]),
                anisotropy=float(rec["aniso"]),
                lineage_strain=float(rec["l_strain"]),
                strain_err=float(rec["s_err"]),
                strain_rate=float(rec["s_rate"]),
                assignment_margin=float(rec["margin"]),
                current_centroid=cur_c,
                predicted_setpoint=pred_c,
                scan_half_widths=half_w,
                centroid_err=c_err,
                lead_scans=pol.lead_scans,
                trigger_reasons=tuple(reasons),
                scattering_angles=scat,
                operando_state=operando_dict,
                predictive_std=pred_std,
                acquisition_kind=acq_kind,
                gp_posterior_mean=float(gp_mu[idx]),
                gp_posterior_std=float(gp_std[idx]),
                execution_mode=pol.execution_mode,
                actuate=should_actuate,
            )
        )

    return selected
