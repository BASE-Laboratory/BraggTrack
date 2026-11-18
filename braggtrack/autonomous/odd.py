"""Operational Design Domain (ODD), Bulk Reference Priors, and Sim-to-Real Audit for FAL Level 3.

Implements the formal autonomy safety, prior-calibration, and audit constructs from
Le Houx, *"Facility Autonomy Levels: A Hierarchical Taxonomy for Self-Driving Beamlines"*:

- **FAL Level 3 Crossing (Inference Barrier)**:
  1. Acts on inferred physical sample state (strain, mosaicity covariance, rocking-curve kurtosis $g_2$).
  2. Evaluates measurements against an external/calibrated reference prior (:class:`BulkReferencePrior` + ``DINOv2``).
  3. Adapts acquisition policy to the declared experimental hypothesis (:data:`HypothesisMode`).
- **Latency Criterion (FAL Eq. 1)**: $\\tau_{\\text{loop}} \\le t_{\\text{decision}} / \\kappa$.
- **Sim-to-Real Gap (FAL Eq. 4)**: Closed-form differential-entropy-normalized Gaussian surprise
  $\\Delta_{\\text{gap}} = \\frac{1}{2}\\sum_k (s'_{\\text{obs},k} - \\hat{\\mu}_k)^2 / \\sigma_{\\text{pred},k}^2 - \\frac{3}{2}$,
  which is affine/unit-invariant with $\\mathbb{E}[\\Delta_{\\text{gap}}] = 0$ under nominal calibration.
- **Discovery Potential Weighting (FAL Eq. 6)**: $D(x) = H(x) \\cdot \\mathcal{W}(\\mathbf{s}_t)$, suppressing
  spurious triggers during upstream beam dumps or shutter closures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from braggtrack.io.operando_log import ScanStimulusPoint

HypothesisMode = Literal["mechanical_fracture", "phase_nucleation", "intragranular_disorder"]
ExecutionMode = Literal["replay", "shadow", "watch", "active"]
FALActiveLevel = Literal["L3", "L2_fallback", "L1_halt"]

# Exact chi-square critical values for df=3 (3D (mu, chi, d) predictive setpoint Gaussian)
CHI2_DF3_P95: float = 7.814727903251179
CHI2_DF3_P99: float = 11.344866730144373
DEFAULT_VALIDITY_EPSILON: float = 0.5 * (CHI2_DF3_P99 - 3.0)  # 4.172433365...


@dataclass(frozen=True)
class OperationalDesignDomain:
    """Explicit Operational Design Domain ($\\mathcal{O}_{\\text{ODD}}$) bounding Level 3 agency."""

    mu_bounds: tuple[float, float] | None = None
    chi_bounds: tuple[float, float] | None = None
    d_bounds: tuple[float, float] | None = None
    max_extrapolation_steps: float = 12.0
    voltage_bounds_v: tuple[float, float] | None = None
    temperature_bounds_c: tuple[float, float] | None = None
    max_zooms_per_lineage: int | None = None
    scan_cadence_sec: float | None = None
    latency_safety_factor_kappa: float = 2.0
    min_instrument_health: float = 0.25
    validity_epsilon: float = DEFAULT_VALIDITY_EPSILON

    def clamp_setpoint(
        self,
        pred: tuple[float, float, float],
        half_widths: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], bool]:
        """Clamp ``(mu, chi, d)`` setpoints and scan intervals to physical motor soft-limits."""
        pm, pc, pd = pred
        hm, hc, hd = half_widths
        clamped = False

        def _clamp_axis(val: float, hw: float, bounds: tuple[float, float] | None) -> tuple[float, float]:
            nonlocal clamped
            if bounds is None:
                return val, hw
            lo, hi = (min(bounds[0], bounds[1]), max(bounds[0], bounds[1]))
            new_val = min(max(val, lo), hi)
            if abs(new_val - val) > 1e-12:
                clamped = True
            # Ensure [new_val - new_hw, new_val + new_hw] stays inside [lo, hi] when span permits
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

        pm, hm = _clamp_axis(pm, hm, self.mu_bounds)
        pc, hc = _clamp_axis(pc, hc, self.chi_bounds)
        pd, hd = _clamp_axis(pd, hd, self.d_bounds)
        return (pm, pc, pd), (hm, hc, hd), clamped

    def check_sample_environment(self, stimulus: ScanStimulusPoint | None) -> tuple[bool, list[str]]:
        """Check whether electrochemical potential and sample temperature remain inside $\\mathcal{O}_{\\text{ODD}}$."""
        if stimulus is None:
            return True, []
        violations: list[str] = []
        if self.voltage_bounds_v is not None and stimulus.voltage_v is not None and math.isfinite(stimulus.voltage_v):
            v_lo, v_hi = min(self.voltage_bounds_v), max(self.voltage_bounds_v)
            if not (v_lo <= float(stimulus.voltage_v) <= v_hi):
                violations.append(f"voltage_out_of_odd={stimulus.voltage_v:.3f}V_not_in_[{v_lo:.3f},{v_hi:.3f}]")
        if (
            self.temperature_bounds_c is not None
            and stimulus.temperature_c is not None
            and math.isfinite(stimulus.temperature_c)
        ):
            t_lo, t_hi = min(self.temperature_bounds_c), max(self.temperature_bounds_c)
            if not (t_lo <= float(stimulus.temperature_c) <= t_hi):
                violations.append(
                    f"temperature_out_of_odd={stimulus.temperature_c:.2f}C_not_in_[{t_lo:.2f},{t_hi:.2f}]"
                )
        return len(violations) == 0, violations

    def evaluate_latency_budget(self, step_latency_sec: float) -> tuple[bool, float | None]:
        """Evaluate FAL Eq. (1): $\\tau_{\\text{loop}} \\le t_{\\text{decision}} / \\kappa$."""
        if self.scan_cadence_sec is None or self.scan_cadence_sec <= 0.0:
            return True, None
        kappa = max(1.0, float(self.latency_safety_factor_kappa))
        budget_sec = float(self.scan_cadence_sec) / kappa
        return float(step_latency_sec) <= budget_sec, budget_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu_bounds": list(self.mu_bounds) if self.mu_bounds is not None else None,
            "chi_bounds": list(self.chi_bounds) if self.chi_bounds is not None else None,
            "d_bounds": list(self.d_bounds) if self.d_bounds is not None else None,
            "max_extrapolation_steps": float(self.max_extrapolation_steps),
            "voltage_bounds_v": list(self.voltage_bounds_v) if self.voltage_bounds_v is not None else None,
            "temperature_bounds_c": list(self.temperature_bounds_c) if self.temperature_bounds_c is not None else None,
            "max_zooms_per_lineage": self.max_zooms_per_lineage,
            "scan_cadence_sec": self.scan_cadence_sec,
            "latency_safety_factor_kappa": float(self.latency_safety_factor_kappa),
            "min_instrument_health": float(self.min_instrument_health),
            "validity_epsilon": round(float(self.validity_epsilon), 6),
        }


@dataclass(frozen=True)
class BulkReferencePrior:
    """Reference prior $(\\mu_{\\text{bulk}}, \\sigma_{\\text{bulk}})$ for rocking-curve kurtosis $g_2$ (FAL Eq. 2)."""

    kurtosis_mean: float = 0.0
    kurtosis_std: float = 0.25
    sigma_multiplier: float = 3.0
    n_reference_spots: int = 0
    source: str = "analytic_gaussian_prior"

    @property
    def kurtosis_threshold(self) -> float:
        """Dynamic anomaly threshold $K_{\\text{thresh}} = \\mu_{\\text{bulk}} + k\\,\\sigma_{\\text{bulk}}$."""
        return float(self.kurtosis_mean + self.sigma_multiplier * self.kurtosis_std)

    @classmethod
    def from_spots(
        cls,
        spots: list[dict[str, Any]],
        *,
        sigma_multiplier: float = 3.0,
        min_std: float = 0.15,
    ) -> BulkReferencePrior:
        """Estimate bulk undeformed kurtosis statistics robustly via median and MAD on initial scan spots."""
        vals: list[float] = []
        for s in spots:
            if bool(s.get("touches_rocking_border", False)):
                continue
            g2_max = max(
                abs(float(s.get("kurtosis_mu", 0.0))),
                abs(float(s.get("kurtosis_chi", 0.0))),
                abs(float(s.get("kurtosis_d", 0.0))),
            )
            if math.isfinite(g2_max):
                vals.append(g2_max)
        if not vals:
            return cls(sigma_multiplier=sigma_multiplier)
        arr = np.asarray(vals, dtype=np.float64)
        med = float(np.median(arr))
        mad_std = float(1.4826 * np.median(np.abs(arr - med)))
        return cls(
            kurtosis_mean=med,
            kurtosis_std=max(float(min_std), mad_std),
            sigma_multiplier=float(sigma_multiplier),
            n_reference_spots=len(vals),
            source="initial_scan_bulk_ensemble",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kurtosis_mean": round(float(self.kurtosis_mean), 5),
            "kurtosis_std": round(float(self.kurtosis_std), 5),
            "sigma_multiplier": round(float(self.sigma_multiplier), 3),
            "kurtosis_threshold": round(self.kurtosis_threshold, 5),
            "n_reference_spots": int(self.n_reference_spots),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BulkReferencePrior:
        return cls(
            kurtosis_mean=float(data.get("kurtosis_mean", 0.0)),
            kurtosis_std=float(data.get("kurtosis_std", 0.25)),
            sigma_multiplier=float(data.get("sigma_multiplier", 3.0)),
            n_reference_spots=int(data.get("n_reference_spots", 0)),
            source=str(data.get("source", "external_json_prior")),
        )


def spot_kurtosis_non_gaussianity(obs: dict[str, Any]) -> float:
    """Return $\\max_{k \\in \\{\\mu, \\chi, d\\}} |g_{2,k}|$ quantifying departure from Gaussian rocking peak."""
    return max(
        abs(float(obs.get("kurtosis_mu", 0.0))),
        abs(float(obs.get("kurtosis_chi", 0.0))),
        abs(float(obs.get("kurtosis_d", 0.0))),
    )


def compute_sim_to_real_gap(
    predicted_mean: tuple[float, float, float],
    predicted_std: tuple[float, float, float],
    observed_centroid: tuple[float, float, float],
    *,
    validity_epsilon: float = DEFAULT_VALIDITY_EPSILON,
) -> tuple[float, float, bool, bool]:
    """Compute FAL Eq. (4) differential-entropy-normalized Sim-to-Real Gap $\\Delta_{\\text{gap}}$.

    For a 3D diagonal Gaussian forward model $P_{\\text{sim}}(\\mathbf{s}' \\mid \\mathbf{s}_t, \\mathbf{a}_t) =
    \\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\mathrm{diag}(\\sigma_\\mu^2, \\sigma_\\chi^2, \\sigma_d^2))$,
    $-\\ln P_{\\text{sim}}(\\mathbf{s}'_{\\text{obs}}) - H[P_{\\text{sim}}]$ cancels $\\frac{3}{2}\\ln(2\\pi)$ and
    $\\frac{1}{2}\\ln|\\boldsymbol{\\Sigma}|$ identically, yielding:

    $$\\Delta_{\\text{gap}} = \\frac{1}{2}\\sum_{k=1}^{3} \\frac{(s'_{\\text{obs},k} - \\hat{\\mu}_k)^2}{\\sigma_k^2} - \\frac{3}{2}$$

    Returns ``(delta_gap, mahalanobis_sq, within_95pct_ci, within_validity_domain)``.
    """
    m2 = 0.0
    for mu_k, sig_k, obs_k in zip(predicted_mean, predicted_std, observed_centroid, strict=True):
        denom = max(float(sig_k), 1e-12)
        z = (float(obs_k) - float(mu_k)) / denom
        m2 += z * z
    delta_gap = 0.5 * m2 - 1.5
    within_95 = m2 <= CHI2_DF3_P95
    within_valid = delta_gap <= float(validity_epsilon)
    return float(delta_gap), float(m2), bool(within_95), bool(within_valid)


@dataclass(frozen=True)
class SimToRealAuditRecord:
    """Immutable trajectory audit entry $(\\mathbf{s}_t, \\mathbf{a}_t, P_{\\text{sim}}, \\mathbf{s}'_{\\text{obs}}, \\Delta_{\\text{gap}})$."""

    source_scan_idx: int
    eval_scan_idx: int
    track_id: int
    root_track_id: int
    execution_mode: ExecutionMode
    actuated: bool
    hypothesis: HypothesisMode
    predicted_setpoint: tuple[float, float, float]
    predictive_std: tuple[float, float, float]
    observed_centroid: tuple[float, float, float]
    scan_half_widths: tuple[float, float, float]
    setpoint_hit: bool
    delta_gap: float
    mahalanobis_sq: float
    within_95pct_ci: bool
    within_validity_domain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "fal_audit.v1",
            "source_scan_idx": int(self.source_scan_idx),
            "eval_scan_idx": int(self.eval_scan_idx),
            "track_id": int(self.track_id),
            "root_track_id": int(self.root_track_id),
            "execution_mode": self.execution_mode,
            "actuated": bool(self.actuated),
            "hypothesis": self.hypothesis,
            "predicted_setpoint": [round(float(x), 6) for x in self.predicted_setpoint],
            "predictive_std": [round(float(x), 6) for x in self.predictive_std],
            "observed_centroid": [round(float(x), 6) for x in self.observed_centroid],
            "scan_half_widths": [round(float(x), 6) for x in self.scan_half_widths],
            "setpoint_hit": bool(self.setpoint_hit),
            "delta_gap": round(float(self.delta_gap), 5),
            "mahalanobis_sq": round(float(self.mahalanobis_sq), 5),
            "within_95pct_ci": bool(self.within_95pct_ci),
            "within_validity_domain": bool(self.within_validity_domain),
        }
