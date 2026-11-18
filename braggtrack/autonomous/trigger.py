"""Autonomous DFXM "Scout-and-Zoom" trigger policy and BLISS/Ewoks setpoint extrusion.

Evaluates active tracked reflections at scan ``t`` using strictly causal kinematics,
analytical centroid uncertainties, standardized rocking-curve excess kurtosis $g_2$,
calibrated bulk reference priors, and multi-scan fracture precursor persistence to
select high-priority candidate grains for zoomed Dark-Field X-ray Microscopy (DFXM)
acquisitions at Facility Autonomy Level 3 (FAL L3). Motor setpoints are velocity-extrapolated
to ``t + lead_scans`` and clamped to an explicit :class:`OperationalDesignDomain`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .odd import (
    BulkReferencePrior,
    ExecutionMode,
    FALActiveLevel,
    HypothesisMode,
    OperationalDesignDomain,
    spot_kurtosis_non_gaussianity,
)

if TYPE_CHECKING:
    from braggtrack.io.crystallography import DetectorGeometry
    from braggtrack.io.models import AxisCalibration
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.tracking.kinematics import GrainKinematics
    from braggtrack.tracking.precursors import ObservationPrecursorScore


@dataclass(frozen=True)
class ZoomTriggerPolicy:
    """Decision thresholds, hypothesis mode, ODD bounds, and window geometry for FAL Level 3 zoom."""

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
    hypothesis: HypothesisMode = "mechanical_fracture"
    execution_mode: ExecutionMode = "replay"
    odd: OperationalDesignDomain | None = None
    mu_bounds: tuple[float, float] | None = None
    chi_bounds: tuple[float, float] | None = None
    d_bounds: tuple[float, float] | None = None
    max_extrapolation_steps: float = 12.0
    exclude_border_truncated: bool = False
    border_truncation_penalty: float = 0.65

    def resolved_odd(self) -> OperationalDesignDomain:
        """Return effective :class:`OperationalDesignDomain` merging policy-level motor overrides."""
        if self.odd is not None:
            return OperationalDesignDomain(
                mu_bounds=self.mu_bounds if self.mu_bounds is not None else self.odd.mu_bounds,
                chi_bounds=self.chi_bounds if self.chi_bounds is not None else self.odd.chi_bounds,
                d_bounds=self.d_bounds if self.d_bounds is not None else self.odd.d_bounds,
                max_extrapolation_steps=self.odd.max_extrapolation_steps,
                voltage_bounds_v=self.odd.voltage_bounds_v,
                temperature_bounds_c=self.odd.temperature_bounds_c,
                max_zooms_per_lineage=self.odd.max_zooms_per_lineage,
                scan_cadence_sec=self.odd.scan_cadence_sec,
                latency_safety_factor_kappa=self.odd.latency_safety_factor_kappa,
                min_instrument_health=self.odd.min_instrument_health,
                validity_epsilon=self.odd.validity_epsilon,
            )
        return OperationalDesignDomain(
            mu_bounds=self.mu_bounds,
            chi_bounds=self.chi_bounds,
            d_bounds=self.d_bounds,
            max_extrapolation_steps=self.max_extrapolation_steps,
        )


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
    kurtosis_non_gaussianity: float = 0.0
    discovery_potential: float = 0.0
    instrument_health: float = 1.0
    hypothesis: HypothesisMode = "mechanical_fracture"
    execution_mode: ExecutionMode = "replay"
    actuate: bool = False
    fal_level: FALActiveLevel = "L3"

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
        sm, sc, sd = self.predictive_std
        disc = self.discovery_potential if self.discovery_potential > 0.0 else self.priority_score
        cmd: dict[str, Any] = {
            "schema_version": "bliss_dfxm_zoom.v1",
            "action": "dfxm_high_res_zoom",
            "actuate": bool(self.actuate),
            "execution_mode": self.execution_mode,
            "fal_level": self.fal_level,
            "hypothesis": self.hypothesis,
            "source_scan_idx": int(self.scan_idx),
            "source_scan_name": self.scan_name,
            "track_id": int(self.track_id),
            "root_track_id": int(self.root_track_id),
            "priority_rank": int(self.priority_rank),
            "priority_score": round(float(self.priority_score), 4),
            "discovery_potential": round(float(disc), 4),
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
                "predictive_mu_std": round(float(sm), 6),
                "predictive_chi_std": round(float(sc), 6),
                "predictive_d_std": round(float(sd), 6),
                "strain_err": round(float(self.strain_err), 7),
            },
            "diagnostics": {
                "current_centroid": [round(float(x), 6) for x in self.current_centroid],
                "precursor_score": round(float(self.precursor_score), 4),
                "persistence_scans": int(self.persistence_scans),
                "roi_subpeak_count": int(self.roi_subpeak_count),
                "anisotropy": round(float(self.anisotropy), 4),
                "kurtosis_non_gaussianity": round(float(self.kurtosis_non_gaussianity), 4),
                "instrument_health": round(float(self.instrument_health), 4),
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


def compute_predictive_std(
    obs: dict[str, Any],
    kin: GrainKinematics | None,
    obs_pos_in_kin: int,
    *,
    lead_scans: float = 1.0,
    calibration: AxisCalibration | None = None,
) -> tuple[float, float, float]:
    """Compute 1-sigma predictive transition uncertainty $(\\sigma_{\\text{pred},\\mu}, \\sigma_{\\text{pred},\\chi}, \\sigma_{\\text{pred},d})$."""
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

    step_mu = abs(float(calibration.mu_step)) if calibration is not None else float(obs.get("step_mu", 1.0))
    step_chi = abs(float(calibration.chi_step)) if calibration is not None else float(obs.get("step_chi", 1.0))
    step_d = abs(float(calibration.d_step)) if calibration is not None else float(obs.get("step_d", 1.0))

    std_mu = math.sqrt(max(float(obs.get("cov_zz", obs.get("cov_mu_mu", 0.0))), 0.0))
    std_chi = math.sqrt(max(float(obs.get("cov_xx", obs.get("cov_chi_chi", 0.0))), 0.0))
    std_d = math.sqrt(max(float(obs.get("cov_yy", obs.get("cov_d_d", 0.0))), 0.0))

    # Propagated 1-step linear extrapolation variance + intragranular sub-domain spread + discretization floor
    vel_factor_sq = 1.0 + 2.0 * (float(lead_scans) ** 2)
    sig_mu = max(
        math.sqrt(std_mu * std_mu + vel_factor_sq * err_mu * err_mu + (0.25 * lead_scans * v_mu) ** 2),
        0.35 * max(step_mu, 1e-12),
    )
    sig_chi = max(
        math.sqrt(std_chi * std_chi + vel_factor_sq * err_chi * err_chi + (0.25 * lead_scans * v_chi) ** 2),
        0.35 * max(step_chi, 1e-12),
    )
    sig_d = max(
        math.sqrt(std_d * std_d + vel_factor_sq * err_d * err_d + (0.25 * lead_scans * v_d) ** 2),
        0.35 * max(step_d, 1e-12),
    )
    return (sig_mu, sig_chi, sig_d)


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

    step_mu = abs(float(calibration.mu_step)) if calibration is not None else float(obs.get("step_mu", 1.0))
    step_chi = abs(float(calibration.chi_step)) if calibration is not None else float(obs.get("step_chi", 1.0))
    step_d = abs(float(calibration.d_step)) if calibration is not None else float(obs.get("step_d", 1.0))

    dm = lead_scans * v_mu
    dc = lead_scans * v_chi
    dd = lead_scans * v_d
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

    # Half-width covers intragranular spread (window_sigma * std) + centroid + drift uncertainty, floored at 1.5 steps.
    drift_pad_mu = 0.35 * abs(dm)
    drift_pad_chi = 0.35 * abs(dc)
    drift_pad_d = 0.35 * abs(dd)

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
    zoom_counts_by_lineage: dict[int, int] | None = None,
    reference_prior: BulkReferencePrior | None = None,
    instrument_health: float = 1.0,
    fal_level: FALActiveLevel = "L3",
) -> list[ZoomTarget]:
    """Select, rank, and extrapolate autonomous DFXM zoom targets at ``current_scan_idx``."""
    pol = policy or ZoomTriggerPolicy()
    odd = pol.resolved_odd()

    # FAL Eq. (6) & ODD safety: halt autonomous target emission on beam dumps or ODD sample excursions
    health = min(max(float(instrument_health), 0.0), 1.0)
    env_ok, _ = odd.check_sample_environment(stimulus)
    if not env_ok or fal_level == "L1_halt" or health < odd.min_instrument_health:
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
    dq_dv_boost = 0.0
    if stimulus is not None and stimulus.dq_dv is not None and math.isfinite(stimulus.dq_dv):
        dq_dv_boost = min(0.08, 0.08 * min(1.0, abs(float(stimulus.dq_dv)) / 250.0))

    k_thresh = reference_prior.kurtosis_threshold if reference_prior is not None else 0.75
    should_actuate = bool(pol.execution_mode == "active" and fal_level == "L3")

    raw_candidates: list[ZoomTarget] = []
    for obs in track_table:
        if int(obs["scan_idx"]) != current_scan_idx:
            continue
        tid = int(obs["track_id"])
        role = str(obs.get("lineage_role", ""))
        if pol.exclude_post_split_daughters and role in ("split_child", "split_daughter"):
            continue

        touches_rocking = bool(obs.get("touches_rocking_border", False))
        if pol.exclude_border_truncated and touches_rocking:
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
        if odd.max_zooms_per_lineage is not None and dose_map.get(root_tid, 0) >= int(odd.max_zooms_per_lineage):
            continue

        sc_now = score_at_current.get(tid)
        prec_score = float(sc_now.precursor_score) if sc_now is not None else 0.0
        subpeaks = int(sc_now.roi_subpeak_count) if sc_now is not None else 1
        aniso = float(sc_now.anisotropy) if sc_now is not None else 1.0
        pers = _trailing_persistence(scores_by_tid.get(tid, []), current_scan_idx, pol.min_precursor_score)
        s_rate = float(kin.strain_rate[pos]) if len(kin.strain_rate) > pos else 0.0
        if abs(s_rate) < pol.min_strain_rate:
            continue
        if subpeaks < pol.min_subpeaks:
            continue

        g2_val = spot_kurtosis_non_gaussianity(obs)
        reasons: list[str] = [f"hypothesis={pol.hypothesis}"]

        # FAL Level 3 Condition (iii): hypothesis-conditioned candidate admission & heuristic H(x)
        if pol.hypothesis == "mechanical_fracture":
            if sc_now is None or prec_score < pol.min_precursor_score or pers < pol.min_persistence:
                continue
            priority = (
                0.55 * prec_score
                + 0.20 * min(1.0, pers / 4.0)
                + 0.15 * min(1.0, max(0, subpeaks - 1) * 0.5)
                + 0.10 * min(1.0, abs(s_rate) / 0.005)
                + dq_dv_boost
            )
            reasons.extend(
                [
                    f"precursor_score={prec_score:.3f}>={pol.min_precursor_score:.2f}",
                    f"persistence={pers}_scans",
                ]
            )
        elif pol.hypothesis == "phase_nucleation":
            min_phase_rate = max(pol.min_strain_rate, 0.0015)
            if abs(s_rate) < min_phase_rate:
                continue
            g2_d = abs(float(obs.get("kurtosis_d", 0.0)))
            priority = (
                0.65 * min(1.0, abs(s_rate) / 0.005)
                + 0.15 * min(1.0, g2_d / max(k_thresh, 0.25))
                + 0.10 * prec_score
                + 0.10 * min(1.0, pers / 4.0)
                + 2.0 * dq_dv_boost
            )
            reasons.append(f"phase_strain_rate={s_rate:+.5f}/scan")
        elif pol.hypothesis == "intragranular_disorder":
            if g2_val < k_thresh:
                continue
            priority = (
                0.70 * min(1.0, g2_val / max(2.0 * k_thresh, 0.50))
                + 0.15 * prec_score
                + 0.15 * min(1.0, max(0.0, aniso - 1.0) / 2.0)
                + dq_dv_boost
            )
            reasons.append(f"kurtosis_g2={g2_val:.3f}>={k_thresh:.3f}")
        else:
            raise ValueError(f"Unsupported hypothesis mode: {pol.hypothesis!r}")

        if touches_rocking:
            priority *= float(pol.border_truncation_penalty)
            reasons.append("rocking_border_truncated")

        # FAL Eq. (6): D(x) = H(x) * W(s_t)
        disc_potential = priority * health

        l_strain = float(kin.lineage_strain[pos]) if len(kin.lineage_strain) > pos else float(kin.strain[pos])
        s_err = float(kin.lineage_strain_err[pos]) if len(kin.lineage_strain_err) > pos else 0.0

        if subpeaks >= 2:
            reasons.append(f"bimodal_subpeaks={subpeaks}")
        if aniso >= 2.0:
            reasons.append(f"mosaicity_anisotropy={aniso:.2f}")
        if abs(s_rate) >= 0.001 and pol.hypothesis != "phase_nucleation":
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
            max_extrapolation_steps=odd.max_extrapolation_steps,
        )
        pred_c, half_w, was_clamped = odd.clamp_setpoint(pred_c, half_w)
        if was_clamped:
            reasons.append("clamped_to_motor_bounds")
            if detector is not None:
                if calibration is not None and calibration.calibrated:
                    z_idx = (pred_c[0] - calibration.mu_origin) / (
                        calibration.mu_step if calibration.mu_step != 0 else 1.0
                    )
                    x_px = (pred_c[1] - calibration.chi_origin) / (
                        calibration.chi_step if calibration.chi_step != 0 else 1.0
                    )
                    y_px = (pred_c[2] - calibration.d_origin) / (calibration.d_step if calibration.d_step != 0 else 1.0)
                else:
                    z_idx, y_px, x_px = pred_c[0], pred_c[2], pred_c[1]
                scat = detector.spot_to_scattering(z_idx, y_px, x_px)

        pred_std = compute_predictive_std(
            obs,
            kin,
            pos,
            lead_scans=pol.lead_scans,
            calibration=calibration,
        )

        raw_candidates.append(
            ZoomTarget(
                scan_idx=current_scan_idx,
                scan_name=scan_name,
                track_id=tid,
                root_track_id=root_tid,
                priority_rank=0,
                priority_score=float(disc_potential),
                precursor_score=prec_score,
                persistence_scans=pers,
                roi_subpeak_count=subpeaks,
                anisotropy=aniso,
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
                predictive_std=pred_std,
                kurtosis_non_gaussianity=g2_val,
                discovery_potential=float(disc_potential),
                instrument_health=health,
                hypothesis=pol.hypothesis,
                execution_mode=pol.execution_mode,
                actuate=should_actuate,
                fal_level=fal_level,
            )
        )

    raw_candidates.sort(key=lambda t: (-t.priority_score, -t.precursor_score, t.track_id))
    selected = raw_candidates[: max(0, int(pol.max_triggers_per_scan))]
    ranked: list[ZoomTarget] = []
    for rank_idx, tgt in enumerate(selected, start=1):
        cooldown_map[tgt.root_track_id] = current_scan_idx
        dose_map[tgt.root_track_id] = dose_map.get(tgt.root_track_id, 0) + 1
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
                predictive_std=tgt.predictive_std,
                kurtosis_non_gaussianity=tgt.kurtosis_non_gaussianity,
                discovery_potential=tgt.discovery_potential,
                instrument_health=tgt.instrument_health,
                hypothesis=tgt.hypothesis,
                execution_mode=tgt.execution_mode,
                actuate=tgt.actuate,
                fal_level=tgt.fal_level,
            )
        )
    return ranked
