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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

AcquisitionKind = Literal["ucb", "ei", "max_variance"]


@dataclass(frozen=True)
class GPForecast:
    """Posterior predictive mean and 1-sigma uncertainty for ``(mu, chi, d)`` at ``target_time``."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


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

    # Normalize time axis relative to latest observation using median step size
    diffs = np.diff(t_arr)
    pos_diffs = diffs[diffs > 1e-12]
    dt_scale = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0
    t_last = float(t_arr[-1])
    tau = (t_arr - t_last) / dt_scale
    tau_star = (float(target_time) - t_last) / dt_scale

    # Weighted linear basis H = [1, tau] with mild ridge on slope when n == 2
    weights = 1.0 / np.maximum(s_arr * s_arr, 1e-18)
    H = np.column_stack([np.ones_like(tau), tau])
    HWH = (H.T * weights) @ H
    ridge = np.diag([1e-12, 1e-9 if n >= 3 else 1e-6])
    cov_beta = np.linalg.pinv(HWH + ridge)
    beta = cov_beta @ ((H.T * weights) @ y_arr)

    h_star = np.array([1.0, tau_star], dtype=np.float64)
    trend_pred = float(h_star @ beta)
    residuals = y_arr - (H @ beta)

    # Parametric WLS trend variance; fit Matérn-3/2 residual GP only when residuals exceed noise
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
) -> GPForecast:
    """Forecast 3D ``(mu, chi, d)`` grain setpoint and 1-sigma uncertainty via heteroscedastic GP."""
    pm, sm = forecast_1d_gp(times, mu_vals, mu_errs, target_time, min_std_floor=0.35 * max(step_floors[0], 1e-12))
    pc, sc = forecast_1d_gp(times, chi_vals, chi_errs, target_time, min_std_floor=0.35 * max(step_floors[1], 1e-12))
    pd, sd = forecast_1d_gp(times, d_vals, d_errs, target_time, min_std_floor=0.35 * max(step_floors[2], 1e-12))
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
