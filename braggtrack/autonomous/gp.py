"""Gaussian Process trajectory forecasting and Bayesian acquisition for autonomous DFXM.

Implements standard surrogate modelling and active-learning primitives inspired by
synchrotron, neutron, and microscopy autonomous experimentation frameworks
(``gpCAM`` — Noack et al., 2021; ``ANDiE`` — McDannald et al., 2022; Kalinin et al., 2021)
using ``scikit-learn`` and ``scipy`` with zero extra dependencies:

- **Universal Kriging trajectory forecaster** (:func:`forecast_grain_trajectory_gp`):
  combines weighted linear trend estimation with a heteroscedastic ``Matérn(nu=1.5)``
  Gaussian Process over residuals using analytical Poisson centroid variances
  ``alpha = sigma_pos**2`` to forecast smooth ``(mu, chi, d)`` setpoints and posterior
  ``1-sigma`` uncertainties across uniform or non-uniform timestamps.
- **Bayesian acquisition functions** (:func:`evaluate_acquisition`, :func:`score_candidates_gp`):
  ranks candidate grains via Upper Confidence Bound (``ucb``), Expected Improvement (``ei``),
  or pure Kriging variance exploration (``max_variance``), optionally conditioned on
  high-resolution DFXM zoom outcomes supplied through ``tell_zoom_outcome``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from braggtrack.tracking.drift import TrackKalmanState

if TYPE_CHECKING:
    from braggtrack.io.operando_log import ScanStimulusPoint

AcquisitionKind = Literal["ucb", "ei", "max_variance"]
ForecastDomain = Literal["time", "electro_state"]


@dataclass(frozen=True)
class GPForecast:
    """Posterior predictive mean and 1-sigma uncertainty for ``(mu, chi, d)`` at ``target_time``."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def compute_electrochemical_state_coord(
    stimuli: Sequence[ScanStimulusPoint] | Mapping[int, ScanStimulusPoint],
    scan_indices: Sequence[int],
    *,
    lead_scans: float = 1.0,
    voltage_weight: float = 0.35,
) -> tuple[np.ndarray, float] | None:
    """Build a strictly causal cumulative electrochemical state coordinate ``s_0 .. s_t`` and ``s_{t+lead}``.

    Uses only stimulus measurements at ``scan_indices`` (no future lookahead). During open-circuit
    voltage (OCV) rests (``|I_t| < 5%`` of active median ``|I|`` or ``ds_last ~ 0``), ``s_{t+lead}``
    halts extrapolation (``ds_lead = 0``) instead of overshooting linearly in wall-clock time.
    Returns ``None`` when neither ``capacity_mah_g`` nor ``voltage_v`` is available on the history.
    """
    if len(scan_indices) == 0:
        return np.zeros(0, dtype=np.float64), 0.0

    stim_map: dict[int, ScanStimulusPoint] = (
        {int(k): v for k, v in stimuli.items()}
        if isinstance(stimuli, Mapping)
        else {int(st.scan_idx): st for st in stimuli}
    )
    pts: list[ScanStimulusPoint] = []
    for si in scan_indices:
        st = stim_map.get(int(si))
        if st is None:
            return None
        pts.append(st)

    has_q = all(p.capacity_mah_g is not None and math.isfinite(float(p.capacity_mah_g)) for p in pts)
    has_i_t = all(
        p.current_ma is not None and math.isfinite(float(p.current_ma)) and math.isfinite(float(p.elapsed_s))
        for p in pts
    )
    has_v = all(p.voltage_v is not None and math.isfinite(float(p.voltage_v)) for p in pts)
    if not (has_q or has_i_t or has_v):
        return None

    n = len(pts)
    if n == 1:
        return np.array([0.0], dtype=np.float64), 0.0

    dq = np.zeros(n - 1, dtype=np.float64)
    dv = np.zeros(n - 1, dtype=np.float64)
    for k in range(n - 1):
        if has_q:
            dq[k] = abs(float(pts[k + 1].capacity_mah_g) - float(pts[k].capacity_mah_g))  # type: ignore[arg-type]
        elif has_i_t:
            dt_h = max(0.0, float(pts[k + 1].elapsed_s) - float(pts[k].elapsed_s)) / 3600.0
            i_mid = 0.5 * (abs(float(pts[k + 1].current_ma)) + abs(float(pts[k].current_ma)))  # type: ignore[arg-type]
            dq[k] = i_mid * dt_h
        if has_v:
            dv[k] = abs(float(pts[k + 1].voltage_v) - float(pts[k].voltage_v))  # type: ignore[arg-type]

    pos_dq = dq[dq > 1e-9]
    pos_dv = dv[dv > 1e-9]
    q_scale = float(np.median(pos_dq)) if len(pos_dq) > 0 else 1.0
    v_scale = float(np.median(pos_dv)) if len(pos_dv) > 0 else 1.0

    if len(pos_dq) > 0 and len(pos_dv) > 0:
        raw_ds = (dq / q_scale) + float(voltage_weight) * (dv / v_scale)
    elif len(pos_dq) > 0:
        raw_ds = dq / q_scale
    elif len(pos_dv) > 0:
        raw_ds = dv / v_scale
    else:
        s_coords = np.arange(n, dtype=np.float64) * 1e-4
        return s_coords, float(s_coords[-1])

    ds = np.maximum(raw_ds, 1e-4)
    s_coords = np.concatenate([[0.0], np.cumsum(ds)])

    st_now = pts[-1]
    ds_last = float(raw_ds[-1])
    active_ds = raw_ds[raw_ds > 0.05 * max(float(np.median(raw_ds)), 1e-9)]
    med_active_ds = float(np.median(active_ds)) if len(active_ds) > 0 else ds_last

    if has_i_t:
        currents = np.asarray([abs(float(p.current_ma)) for p in pts], dtype=np.float64)  # type: ignore[arg-type]
        active_i = currents[currents > 1e-9]
        med_i = float(np.median(active_i)) if len(active_i) > 0 else 0.0
        i_now = abs(float(st_now.current_ma))  # type: ignore[arg-type]
        if med_i <= 1e-9 or i_now <= 0.05 * med_i:
            ds_lead_per_scan = 0.0
        else:
            i_prev = abs(float(pts[-2].current_ma))  # type: ignore[arg-type]
            if i_prev > 0.05 * med_i and ds_last > 1e-6:
                ds_lead_per_scan = ds_last * (i_now / i_prev)
            else:
                ds_lead_per_scan = med_active_ds * (i_now / med_i)
    else:
        if ds_last <= 0.05 * max(med_active_ds, 1e-9):
            ds_lead_per_scan = 0.0
        else:
            ds_lead_per_scan = ds_last

    target_s = float(s_coords[-1]) + max(0.0, float(lead_scans)) * ds_lead_per_scan
    return s_coords, target_s


def forecast_1d_gp(
    times: np.ndarray,
    values: np.ndarray,
    obs_std: np.ndarray,
    target_time: float,
    *,
    min_std_floor: float = 1e-6,
) -> tuple[float, float]:
    """Forecast scalar trajectory ``y(target_time)`` via weighted linear trend + Matérn-3/2 GP.

    Universal Kriging formulation (Rasmussen & Williams, Sec. 2.7):
    detrends ``values`` with inverse-variance-weighted linear regression so steady
    crystallographic drift (e.g. ``H2 -> H3`` contraction) never mean-reverts toward
    zero during extrapolation, then fits a heteroscedastic ``Matérn(nu=1.5)`` GP with
    inhomogeneous noise ``alpha = obs_std**2`` to capture smooth nonlinear curvature.
    """
    t_arr = np.asarray(times, dtype=np.float64).ravel()
    y_arr = np.asarray(values, dtype=np.float64).ravel()
    s_arr = np.maximum(np.asarray(obs_std, dtype=np.float64).ravel(), float(min_std_floor))
    n = len(t_arr)

    if n == 0:
        return 0.0, float(min_std_floor)
    if n == 1:
        return float(y_arr[0]), float(max(s_arr[0], min_std_floor))

    diffs = np.diff(t_arr)
    pos_diffs = diffs[diffs > 1e-12]
    dt_scale = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0
    t_last = float(t_arr[-1])
    tau = (t_arr - t_last) / dt_scale
    tau_star = (float(target_time) - t_last) / dt_scale

    weights = 1.0 / np.maximum(s_arr * s_arr, 1e-18)
    if abs(tau_star) <= 1e-9:
        return float(y_arr[-1]), float(max(s_arr[-1], min_std_floor))

    H = np.column_stack([np.ones_like(tau), tau])
    HWH = (H.T * weights) @ H
    if n == 2:
        tau_span2 = max(float((tau[1] - tau[0]) ** 2), 1e-12)
        w_slope_eff = float((weights[0] * weights[1]) / max(weights[0] + weights[1], 1e-18)) * tau_span2
        slope_ridge = max(w_slope_eff, 1e-6)
    else:
        slope_ridge = 1e-9
    ridge = np.diag([1e-12, slope_ridge])
    cov_beta = np.linalg.pinv(HWH + ridge)
    beta = cov_beta @ ((H.T * weights) @ y_arr)

    h_star = np.array([1.0, tau_star], dtype=np.float64)
    trend_pred = float(h_star @ beta)
    residuals = y_arr - (H @ beta)

    dof = max(1, n - 2)
    chi2_red = float(np.sum(weights * residuals * residuals)) / dof if n >= 3 else 1.0
    param_var = float(h_star @ cov_beta @ h_star)
    res_scale = float(np.std(residuals))

    if n < 3 or chi2_red <= 1.25 or res_scale <= 1e-12:
        total_std = max(math.sqrt(max(param_var, 0.0)), float(min_std_floor))
        return trend_pred, total_std

    y_norm = residuals / res_scale
    alpha_norm = np.maximum((s_arr / res_scale) ** 2, 1e-6)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=2.0, length_scale_bounds=(0.5, 15.0), nu=1.5)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_norm,
        optimizer="fmin_l_bfgs_b" if n >= 4 else None,
        n_restarts_optimizer=0,
        normalize_y=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gpr.fit(tau.reshape(-1, 1), y_norm)
        r_mean_norm, r_std_norm = gpr.predict(np.array([[tau_star]], dtype=np.float64), return_std=True)

    res_pred = float(r_mean_norm[0]) * res_scale
    res_var = (float(r_std_norm[0]) * res_scale) ** 2
    total_std = max(math.sqrt(max(param_var + res_var, 0.0)), float(min_std_floor))
    return trend_pred + res_pred, total_std


def forecast_persistence(
    mu_vals: np.ndarray,
    chi_vals: np.ndarray,
    d_vals: np.ndarray,
    mu_errs: np.ndarray,
    chi_errs: np.ndarray,
    d_errs: np.ndarray,
    *,
    step_floors: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
) -> GPForecast:
    """Zero-velocity persistence baseline: predicts the latest observed centroid ``y_t``."""
    m = np.asarray(mu_vals, dtype=np.float64).ravel()
    c = np.asarray(chi_vals, dtype=np.float64).ravel()
    d = np.asarray(d_vals, dtype=np.float64).ravel()
    if len(m) == 0:
        return GPForecast(mean=(0.0, 0.0, 0.0), std=step_floors)
    em = max(float(np.asarray(mu_errs, dtype=np.float64).ravel()[-1]), 0.35 * max(step_floors[0], 1e-12))
    ec = max(float(np.asarray(chi_errs, dtype=np.float64).ravel()[-1]), 0.35 * max(step_floors[1], 1e-12))
    ed = max(float(np.asarray(d_errs, dtype=np.float64).ravel()[-1]), 0.35 * max(step_floors[2], 1e-12))
    return GPForecast(mean=(float(m[-1]), float(c[-1]), float(d[-1])), std=(em, ec, ed))


def forecast_constant_velocity_2pt(
    times: np.ndarray,
    mu_vals: np.ndarray,
    chi_vals: np.ndarray,
    d_vals: np.ndarray,
    mu_errs: np.ndarray,
    chi_errs: np.ndarray,
    d_errs: np.ndarray,
    target_time: float,
    *,
    step_floors: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
) -> GPForecast:
    """Unregularized 2-point finite-difference constant-velocity baseline from ``(t_{n-2}, t_{n-1})``."""
    t = np.asarray(times, dtype=np.float64).ravel()
    if len(t) < 2 or float(t[-1] - t[-2]) <= 1e-12:
        return forecast_persistence(mu_vals, chi_vals, d_vals, mu_errs, chi_errs, d_errs, step_floors=step_floors)
    dt_prev = float(t[-1] - t[-2])
    dt_fwd = float(target_time) - float(t[-1])
    ratio = dt_fwd / dt_prev

    def _extrap(vals: np.ndarray, errs: np.ndarray, flr: float) -> tuple[float, float]:
        v = np.asarray(vals, dtype=np.float64).ravel()
        e = np.maximum(np.asarray(errs, dtype=np.float64).ravel(), 0.35 * max(flr, 1e-12))
        pred = float(v[-1]) + ratio * float(v[-1] - v[-2])
        var = ((1.0 + ratio) ** 2) * float(e[-1] ** 2) + (ratio**2) * float(e[-2] ** 2)
        return pred, math.sqrt(max(var, (0.35 * max(flr, 1e-12)) ** 2))

    pm, sm = _extrap(mu_vals, mu_errs, step_floors[0])
    pc, sc = _extrap(chi_vals, chi_errs, step_floors[1])
    pd, sd = _extrap(d_vals, d_errs, step_floors[2])
    return GPForecast(mean=(pm, pc, pd), std=(sm, sc, sd))


def forecast_kalman_cv(
    times: np.ndarray,
    mu_vals: np.ndarray,
    chi_vals: np.ndarray,
    d_vals: np.ndarray,
    mu_errs: np.ndarray,
    chi_errs: np.ndarray,
    d_errs: np.ndarray,
    target_time: float,
    *,
    step_floors: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
    accel_sigma: float = 2.5,
) -> GPForecast:
    """Causal constant-velocity Kalman filter baseline (:class:`TrackKalmanState`)."""
    t = np.asarray(times, dtype=np.float64).ravel()
    m = np.asarray(mu_vals, dtype=np.float64).ravel()
    c = np.asarray(chi_vals, dtype=np.float64).ravel()
    d = np.asarray(d_vals, dtype=np.float64).ravel()
    if len(t) == 0:
        return GPForecast(mean=(0.0, 0.0, 0.0), std=step_floors)
    em = np.maximum(np.asarray(mu_errs, dtype=np.float64).ravel(), 0.35 * max(step_floors[0], 1e-12))
    ec = np.maximum(np.asarray(chi_errs, dtype=np.float64).ravel(), 0.35 * max(step_floors[1], 1e-12))
    ed = np.maximum(np.asarray(d_errs, dtype=np.float64).ravel(), 0.35 * max(step_floors[2], 1e-12))
    if len(t) == 1:
        return GPForecast(mean=(float(m[0]), float(c[0]), float(d[0])), std=(float(em[0]), float(ec[0]), float(ed[0])))

    diffs = np.diff(t)
    dt_scale = float(np.median(diffs[diffs > 1e-12])) if np.any(diffs > 1e-12) else 1.0
    tau = t / dt_scale
    tau_target = float(target_time) / dt_scale

    scales = (
        max(float(step_floors[0]), 1e-12),
        max(float(step_floors[1]), 1e-12),
        max(float(step_floors[2]), 1e-12),
    )
    accel_steps = (0.6, float(accel_sigma), float(accel_sigma))
    kf = TrackKalmanState.initialize(
        {
            "centroid_mu": float(m[0]),
            "centroid_chi": float(c[0]),
            "centroid_d": float(d[0]),
            "centroid_mu_err": float(em[0]),
            "centroid_chi_err": float(ec[0]),
            "centroid_d_err": float(ed[0]),
            "step_mu": scales[0],
            "step_chi": scales[1],
            "step_d": scales[2],
        },
        time_val=float(tau[0]),
        init_vel_std_steps=max(5.0, 2.0 * float(accel_sigma)),
    )
    for k in range(1, len(tau)):
        kf.update(
            {
                "centroid_mu": float(m[k]),
                "centroid_chi": float(c[k]),
                "centroid_d": float(d[k]),
                "centroid_mu_err": float(em[k]),
                "centroid_chi_err": float(ec[k]),
                "centroid_d_err": float(ed[k]),
                "step_mu": scales[0],
                "step_chi": scales[1],
                "step_d": scales[2],
            },
            time_val=float(tau[k]),
            accel_std_steps=accel_steps,
        )

    dt_fwd = max(0.0, tau_target - float(tau[-1]))
    if dt_fwd <= 1e-9:
        pred_mean = (float(kf.pos[0]), float(kf.pos[1]), float(kf.pos[2]))
        raw_std = (
            math.sqrt(max(float(kf.cov[0, 0, 0]), 1e-12)),
            math.sqrt(max(float(kf.cov[1, 0, 0]), 1e-12)),
            math.sqrt(max(float(kf.cov[2, 0, 0]), 1e-12)),
        )
    else:
        pred_mean, raw_std = kf.predict(dt=dt_fwd, accel_std_steps=accel_steps, scales=scales)
    sm = max(raw_std[0], 0.35 * scales[0])
    sc = max(raw_std[1], 0.35 * scales[1])
    sd = max(raw_std[2], 0.35 * scales[2])
    return GPForecast(mean=pred_mean, std=(sm, sc, sd))


def forecast_grain_trajectory_gp(
    times: np.ndarray,
    mu_vals: np.ndarray,
    chi_vals: np.ndarray,
    d_vals: np.ndarray,
    mu_errs: np.ndarray,
    chi_errs: np.ndarray,
    d_errs: np.ndarray,
    target_time: float,
    *,
    step_floors: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
    domain: ForecastDomain = "time",
    stimuli: Sequence[ScanStimulusPoint] | Mapping[int, ScanStimulusPoint] | None = None,
    scan_indices: Sequence[int] | None = None,
    lead_scans: float | None = None,
) -> GPForecast:
    """Forecast 3D ``(mu, chi, d)`` grain setpoint and 1-sigma uncertainty via heteroscedastic GP."""
    coord_arr = np.asarray(times, dtype=np.float64).ravel()
    coord_target = float(target_time)

    if domain == "electro_state" and stimuli is not None and len(coord_arr) >= 2:
        if scan_indices is not None:
            s_idx_seq: Sequence[int] = scan_indices
        else:
            s_idx_seq = [int(round(float(x))) for x in coord_arr]
        if lead_scans is not None:
            eff_lead = float(lead_scans)
        else:
            diffs = np.diff(coord_arr)
            dt_med = float(np.median(diffs[diffs > 1e-12])) if np.any(diffs > 1e-12) else 1.0
            eff_lead = max(0.0, (float(target_time) - float(coord_arr[-1])) / dt_med)
        state_res = compute_electrochemical_state_coord(stimuli, s_idx_seq, lead_scans=eff_lead)
        if state_res is not None:
            coord_arr, coord_target = state_res

    pm, sm = forecast_1d_gp(coord_arr, mu_vals, mu_errs, coord_target, min_std_floor=0.35 * max(step_floors[0], 1e-12))
    pc, sc = forecast_1d_gp(
        coord_arr, chi_vals, chi_errs, coord_target, min_std_floor=0.35 * max(step_floors[1], 1e-12)
    )
    pd, sd = forecast_1d_gp(coord_arr, d_vals, d_errs, coord_target, min_std_floor=0.35 * max(step_floors[2], 1e-12))
    return GPForecast(mean=(pm, pc, pd), std=(sm, sc, sd))


def evaluate_acquisition(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    kind: AcquisitionKind = "ucb",
    beta: float = 1.96,
    best_observed: float = 0.0,
    xi: float = 0.01,
) -> np.ndarray:
    """Evaluate standard Bayesian acquisition scores over posterior ``(mean, std)``."""
    mu = np.asarray(mean, dtype=np.float64)
    sig = np.maximum(np.asarray(std, dtype=np.float64), 0.0)

    if kind == "ucb":
        return mu + float(beta) * sig
    if kind == "max_variance":
        return sig
    if kind == "ei":
        imp = mu - float(best_observed) - float(xi)
        safe_sig = np.where(sig > 1e-12, sig, 1.0)
        z = np.where(sig > 1e-12, imp / safe_sig, 0.0)
        ei = imp * norm.cdf(z) + safe_sig * norm.pdf(z)
        return np.where(sig > 1e-12, np.maximum(ei, 0.0), np.maximum(imp, 0.0))
    raise ValueError(f"Unsupported acquisition kind: {kind!r}. Expected 'ucb', 'ei', or 'max_variance'.")


def _standardize_features(X: np.ndarray) -> np.ndarray:
    if len(X) <= 1:
        return np.zeros_like(X, dtype=np.float64)
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    scale = np.where(mad > 1e-9, mad, np.maximum(np.std(X, axis=0), 1.0))
    return (X - med) / scale


def score_candidates_gp(
    candidate_records: list[dict[str, Any]],
    *,
    acquisition: AcquisitionKind = "ucb",
    beta: float = 1.96,
    xi: float = 0.01,
    lead_scans: float = 1.0,
    zoom_outcomes: dict[int, tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute GP posterior ``(mean, std, acquisition_score)`` across candidate grains.

    Each entry in ``candidate_records`` provides:
    - ``score_times`` / ``score_history``: causal precursor score time series for that grain.
    - ``feature_vec``: ``[abs_lineage_strain, abs_strain_rate, log_anisotropy, subpeak_norm, prec_score]``.

    When ``zoom_outcomes`` (from ``tell_zoom_outcome``) has ``>= 1`` measured high-resolution
    DFXM observations, a ``Matérn(nu=2.5)`` GP in descriptor space blends real DFXM feedback
    into the posterior mean and epistemic uncertainty.
    """
    n_cand = len(candidate_records)
    if n_cand == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty

    temp_mu = np.zeros(n_cand, dtype=np.float64)
    temp_std = np.zeros(n_cand, dtype=np.float64)
    feats = np.vstack([np.asarray(rec["feature_vec"], dtype=np.float64) for rec in candidate_records])

    for i, rec in enumerate(candidate_records):
        s_times = np.asarray(rec["score_times"], dtype=np.float64)
        s_vals = np.asarray(rec["score_history"], dtype=np.float64)
        if len(s_vals) == 0:
            temp_mu[i] = float(rec.get("prec_score", 0.0))
            temp_std[i] = 0.15
        elif len(s_vals) == 1:
            temp_mu[i] = float(s_vals[0])
            temp_std[i] = 0.12
        else:
            diffs = np.diff(s_times)
            dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
            noise_est = np.full_like(s_vals, 0.05, dtype=np.float64)
            pm, ps = forecast_1d_gp(
                s_times,
                s_vals,
                noise_est,
                float(s_times[-1]) + float(lead_scans) * dt,
                min_std_floor=0.02,
            )
            temp_mu[i] = float(np.clip(pm, 0.0, 1.5))
            temp_std[i] = float(np.clip(ps, 0.02, 0.50))

    # Cross-grain epistemic uncertainty in standardized descriptor space
    if zoom_outcomes and len(zoom_outcomes) >= 1:
        obs_X = np.vstack([v[0] for v in zoom_outcomes.values()])
        obs_y = np.asarray([v[1] for v in zoom_outcomes.values()], dtype=np.float64)
        all_X = np.vstack([obs_X, feats])
        all_X_std = _standardize_features(all_X)
        obs_X_std = all_X_std[: len(obs_X)]
        cand_X_std = all_X_std[len(obs_X) :]

        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.5, length_scale_bounds=(0.2, 10.0), nu=2.5)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.02,
            optimizer="fmin_l_bfgs_b" if len(obs_X) >= 3 else None,
            normalize_y=len(obs_X) >= 2,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gpr.fit(obs_X_std, obs_y)
            ens_mu, ens_std = gpr.predict(cand_X_std, return_std=True)

        post_mu = 0.5 * temp_mu + 0.5 * np.asarray(ens_mu, dtype=np.float64)
        post_std = np.sqrt(temp_std**2 + np.maximum(ens_std, 0.02) ** 2)
        best_obs = float(np.max(obs_y))
    else:
        cand_X_std = _standardize_features(feats)
        if n_cand >= 2:
            # Matérn-like kernel similarity to ensemble peers: isolated outlier grains carry higher epistemic std
            sq_dists = np.sum((cand_X_std[:, None, :] - cand_X_std[None, :, :]) ** 2, axis=-1)
            dists = np.sqrt(np.maximum(sq_dists, 0.0))
            k_mat = (1.0 + math.sqrt(3.0) * dists / 2.0) * np.exp(-math.sqrt(3.0) * dists / 2.0)
            np.fill_diagonal(k_mat, 0.0)
            mean_peer_sim = np.sum(k_mat, axis=1) / max(1, n_cand - 1)
            epistemic_std = 0.12 * np.sqrt(np.clip(1.0 - mean_peer_sim, 0.0, 1.0))
        else:
            epistemic_std = np.full(n_cand, 0.05, dtype=np.float64)

        post_mu = temp_mu
        post_std = np.sqrt(temp_std**2 + epistemic_std**2)
        # Baseline reference for EI when no zoom outcomes exist yet: median precursor level of ensemble
        best_obs = float(np.median(post_mu)) if n_cand >= 2 else 0.25

    acq_scores = evaluate_acquisition(
        post_mu,
        post_std,
        kind=acquisition,
        beta=beta,
        best_observed=best_obs,
        xi=xi,
    )
    return post_mu, post_std, acq_scores
