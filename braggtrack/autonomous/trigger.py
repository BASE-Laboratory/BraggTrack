"""Autonomous DFXM "Scout-and-Zoom" trigger policy and BLISS/Ewoks setpoint extrusion.

Evaluates active tracked reflections at scan ``t`` using strictly causal kinematics,
analytical centroid uncertainties, and multi-scan fracture precursor persistence to
select high-priority candidate grains for zoomed Dark-Field X-ray Microscopy (DFXM)
rocking/mosaicity acquisitions. Motor setpoints are velocity-extrapolated to
``t + lead_scans`` so fast-shifting reflections (e.g. Ni-rich layered oxide H2 -> H3
lattice collapse) stay centered in the narrow DFXM objective acceptance cone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.models import AxisCalibration
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.tracking.kinematics import GrainKinematics
    from braggtrack.tracking.precursors import ObservationPrecursorScore


@dataclass(frozen=True)
class ZoomTriggerPolicy:
    """Decision thresholds, cooldowns, and scan-window geometry for autonomous DFXM zoom."""

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

    def contains_point(self, mu: float, chi: float, d: float) -> bool:
        """Return True if ``(mu, chi, d)`` lies inside ``predicted_setpoint ± scan_half_widths``."""
        pm, pc, pd = self.predicted_setpoint
        hm, hc, hd = self.scan_half_widths
        return abs(mu - pm) <= hm and abs(chi - pc) <= hc and abs(d - pd) <= hd

    def to_bliss_command(self) -> dict[str, Any]:
        """Format as an ESRF BLISS / Ewoks machine-readable DFXM zoom command payload."""
        pm, pc, pd = self.predicted_setpoint
        hm, hc, hd = self.scan_half_widths
        em, ec, ed = self.centroid_err
        cmd: dict[str, Any] = {
            "schema_version": "bliss_dfxm_zoom.v1",
            "action": "dfxm_high_res_zoom",
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
                "strain_err": round(float(self.strain_err), 7),
            },
            "diagnostics": {
                "current_centroid": [round(float(x), 6) for x in self.current_centroid],
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
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    dict[str, float] | None,
]:
    """Compute current centroid, velocity-extrapolated setpoint, half-widths, errors, and scattering angles."""
    cur_mu = float(obs["centroid_mu"])
    cur_chi = float(obs["centroid_chi"])
    cur_d = float(obs["centroid_d"])

    if kin is not None and 0 <= obs_pos_in_kin < len(kin.scan_indices):
        v_mu = float(kin.rate_mu[obs_pos_in_kin]) if len(kin.rate_mu) > obs_pos_in_kin else 0.0
        v_chi = float(kin.rate_chi[obs_pos_in_kin]) if len(kin.rate_chi) > obs_pos_in_kin else 0.0
        v_d = float(kin.rate_d[obs_pos_in_kin]) if len(kin.rate_d) > obs_pos_in_kin else 0.0
        err_mu = float(kin.centroid_mu_err[obs_pos_in_kin]) if len(kin.centroid_mu_err) > obs_pos_in_kin else 0.0
        err_chi = float(kin.centroid_chi_err[obs_pos_in_kin]) if len(kin.centroid_chi_err) > obs_pos_in_kin else 0.0
        err_d = float(kin.centroid_d_err[obs_pos_in_kin]) if len(kin.centroid_d_err) > obs_pos_in_kin else 0.0
    else:
        v_mu = v_chi = v_d = 0.0
        err_mu = float(obs.get("centroid_mu_err", 0.0))
        err_chi = float(obs.get("centroid_chi_err", 0.0))
        err_d = float(obs.get("centroid_d_err", 0.0))

    pred_mu = cur_mu + lead_scans * v_mu
    pred_chi = cur_chi + lead_scans * v_chi
    pred_d = cur_d + lead_scans * v_d

    step_mu = abs(float(calibration.mu_step)) if calibration is not None else 1.0
    step_chi = abs(float(calibration.chi_step)) if calibration is not None else 1.0
    step_d = abs(float(calibration.d_step)) if calibration is not None else 1.0

    std_mu = math.sqrt(max(float(obs.get("cov_zz", obs.get("cov_mu_mu", 0.0))), 0.0))
    std_chi = math.sqrt(max(float(obs.get("cov_xx", obs.get("cov_chi_chi", 0.0))), 0.0))
    std_d = math.sqrt(max(float(obs.get("cov_yy", obs.get("cov_d_d", 0.0))), 0.0))

    # Half-width covers intragranular spread (window_sigma * std) + centroid + drift uncertainty, floored at 1.5 steps.
    drift_pad_mu = 0.35 * abs(lead_scans * v_mu)
    drift_pad_chi = 0.35 * abs(lead_scans * v_chi)
    drift_pad_d = 0.35 * abs(lead_scans * v_d)

    half_mu = max(window_sigma * (std_mu + err_mu) + drift_pad_mu, 1.5 * step_mu)
    half_chi = max(window_sigma * (std_chi + err_chi) + drift_pad_chi, 1.5 * step_chi)
    half_d = max(window_sigma * (std_d + err_d) + drift_pad_d, 1.5 * step_d)

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
    calibration: AxisCalibration | None = None,
    detector: DetectorGeometry | None = None,
    stimulus: ScanStimulusPoint | None = None,
    last_triggered_scan_by_lineage: dict[int, int] | None = None,
) -> list[ZoomTarget]:
    """Select, rank, and extrapolate autonomous DFXM zoom targets at ``current_scan_idx``."""
    pol = policy or ZoomTriggerPolicy()
    kin_by_tid = {k.track_id: k for k in kinematics}
    scores_by_tid: dict[int, list[ObservationPrecursorScore]] = {}
    score_at_current: dict[int, ObservationPrecursorScore] = {}
    for sc in obs_precursor_scores:
        scores_by_tid.setdefault(sc.track_id, []).append(sc)
        if sc.scan_idx == current_scan_idx:
            score_at_current[sc.track_id] = sc

    cooldown_map = last_triggered_scan_by_lineage if last_triggered_scan_by_lineage is not None else {}
    operando_dict = stimulus.to_dict() if stimulus is not None else None
    dq_dv_boost = 0.0
    if stimulus is not None and stimulus.dq_dv is not None and math.isfinite(stimulus.dq_dv):
        dq_dv_boost = min(0.08, 0.08 * min(1.0, abs(float(stimulus.dq_dv)) / 250.0))

    raw_candidates: list[ZoomTarget] = []
    for obs in track_table:
        if int(obs["scan_idx"]) != current_scan_idx:
            continue
        tid = int(obs["track_id"])
        if pol.exclude_post_split_daughters and str(obs.get("lineage_role", "")) == "split_daughter":
            continue

        intensity = float(obs.get("integrated_intensity", 0.0))
        if intensity < pol.min_intensity:
            continue

        margin = float(obs.get("assignment_margin", float("inf")))
        if margin < pol.min_assignment_margin:
            continue

        kin = kin_by_tid.get(tid)
        if kin is None or current_scan_idx not in kin.scan_indices:
            continue
        pos = kin.scan_indices.index(current_scan_idx)
        track_age = pos + 1
        if track_age < pol.min_track_age:
            continue

        root_tid = int(kin.root_track_id) if kin.root_track_id > 0 else tid
        prev_trig = cooldown_map.get(root_tid)
        if prev_trig is not None and (current_scan_idx - prev_trig) < pol.cooldown_scans:
            continue

        sc_now = score_at_current.get(tid)
        if sc_now is None or sc_now.precursor_score < pol.min_precursor_score:
            continue
        if sc_now.roi_subpeak_count < pol.min_subpeaks:
            continue

        pers = _trailing_persistence(scores_by_tid.get(tid, []), current_scan_idx, pol.min_precursor_score)
        if pers < pol.min_persistence:
            continue

        s_rate = float(kin.strain_rate[pos]) if len(kin.strain_rate) > pos else 0.0
        if abs(s_rate) < pol.min_strain_rate:
            continue

        l_strain = float(kin.lineage_strain[pos]) if len(kin.lineage_strain) > pos else float(kin.strain[pos])
        s_err = float(kin.lineage_strain_err[pos]) if len(kin.lineage_strain_err) > pos else 0.0

        priority = (
            0.55 * float(sc_now.precursor_score)
            + 0.20 * min(1.0, pers / 4.0)
            + 0.15 * min(1.0, max(0, sc_now.roi_subpeak_count - 1) * 0.5)
            + 0.10 * min(1.0, abs(s_rate) / 0.005)
            + dq_dv_boost
        )

        reasons: list[str] = [
            f"precursor_score={sc_now.precursor_score:.3f}>={pol.min_precursor_score:.2f}",
            f"persistence={pers}_scans",
        ]
        if sc_now.roi_subpeak_count >= 2:
            reasons.append(f"bimodal_subpeaks={sc_now.roi_subpeak_count}")
        if sc_now.anisotropy >= 2.0:
            reasons.append(f"mosaicity_anisotropy={sc_now.anisotropy:.2f}")
        if abs(s_rate) >= 0.001:
            reasons.append(f"strain_rate={s_rate:+.5f}/scan")
        if dq_dv_boost > 0.02 and stimulus is not None and stimulus.voltage_v is not None:
            reasons.append(f"redox_active_E={stimulus.voltage_v:.3f}V")

        cur_c, pred_c, half_w, c_err, scat = compute_target_geometry(
            obs,
            kin,
            pos,
            lead_scans=pol.lead_scans,
            window_sigma=pol.window_sigma,
            calibration=calibration,
            detector=detector,
        )

        raw_candidates.append(
            ZoomTarget(
                scan_idx=current_scan_idx,
                scan_name=scan_name,
                track_id=tid,
                root_track_id=root_tid,
                priority_rank=0,
                priority_score=float(priority),
                precursor_score=float(sc_now.precursor_score),
                persistence_scans=pers,
                roi_subpeak_count=int(sc_now.roi_subpeak_count),
                anisotropy=float(sc_now.anisotropy),
                lineage_strain=l_strain,
                strain_err=s_err,
                strain_rate=s_rate,
                assignment_margin=margin,
                current_centroid=cur_c,
                predicted_setpoint=pred_c,
                scan_half_widths=half_w,
                centroid_err=c_err,
                lead_scans=pol.lead_scans,
                trigger_reasons=tuple(reasons),
                scattering_angles=scat,
                operando_state=operando_dict,
            )
        )

    raw_candidates.sort(key=lambda t: (-t.priority_score, -t.precursor_score, t.track_id))
    selected = raw_candidates[: max(0, int(pol.max_triggers_per_scan))]
    ranked: list[ZoomTarget] = []
    for rank_idx, tgt in enumerate(selected, start=1):
        cooldown_map[tgt.root_track_id] = current_scan_idx
        ranked.append(
            ZoomTarget(
                scan_idx=tgt.scan_idx,
                scan_name=tgt.scan_name,
                track_id=tgt.track_id,
                root_track_id=tgt.root_track_id,
                priority_rank=rank_idx,
                priority_score=tgt.priority_score,
                precursor_score=tgt.precursor_score,
                persistence_scans=tgt.persistence_scans,
                roi_subpeak_count=tgt.roi_subpeak_count,
                anisotropy=tgt.anisotropy,
                lineage_strain=tgt.lineage_strain,
                strain_err=tgt.strain_err,
                strain_rate=tgt.strain_rate,
                assignment_margin=tgt.assignment_margin,
                current_centroid=tgt.current_centroid,
                predicted_setpoint=tgt.predicted_setpoint,
                scan_half_widths=tgt.scan_half_widths,
                centroid_err=tgt.centroid_err,
                lead_scans=tgt.lead_scans,
                trigger_reasons=tgt.trigger_reasons,
                scattering_angles=tgt.scattering_angles,
                operando_state=tgt.operando_state,
            )
        )
    return ranked
