"""Pluggable cost functions for frame-to-frame spot association.

PositionShapeCost provides a physics-only baseline (position + shape).
GeometrySemanticCost adds a semantic term (cosine on fused embeddings).
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy.spatial.distance import cdist


class CostFunction(Protocol):
    """Interface that any association cost must satisfy."""

    def pairwise_cost_matrix(self, spots_t: list[dict], spots_t1: list[dict]) -> np.ndarray:
        """Dense ``(N, M)`` costs; ``inf`` for gated or inadmissible pairs."""

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        """Scalar cost for one pair (tests, debugging).

        Should return ``math.inf`` when the pair is inadmissible (gated out).
        """
        ...


def _spot_scales(spot: dict) -> tuple[float, float, float]:
    """Return positive per-axis voxel step scales ``(step_mu, step_chi, step_d)``."""
    sm = abs(float(spot.get("step_mu", 1.0)))
    sc = abs(float(spot.get("step_chi", 1.0)))
    sd = abs(float(spot.get("step_d", 1.0)))
    return (
        sm if sm > 1e-12 else 1.0,
        sc if sc > 1e-12 else 1.0,
        sd if sd > 1e-12 else 1.0,
    )


def _spot_eigs(spot: dict, *, normalize_by_step: bool) -> tuple[float, float, float]:
    """Return shape eigenvalues (dimensionless voxel-space when calibrated & normalized)."""
    if normalize_by_step and "vox_eig_1" in spot:
        return (
            float(spot.get("vox_eig_1", 0.0)),
            float(spot.get("vox_eig_2", 0.0)),
            float(spot.get("vox_eig_3", 0.0)),
        )
    e1 = float(spot.get("eig_1", 0.0))
    e2 = float(spot.get("eig_2", 0.0))
    e3 = float(spot.get("eig_3", 0.0))
    if normalize_by_step and bool(spot.get("calibrated", 0)):
        sm, sc, sd = _spot_scales(spot)
        s2 = max(sm, sc, sd) ** 2
        if s2 > 0.0:
            return (e1 / s2, e2 / s2, e3 / s2)
    return (e1, e2, e3)


def _spot_widths(spot: dict, *, normalize_by_step: bool) -> tuple[float, float, float]:
    """Return non-negative spot standard-deviation semi-axes ``sqrt(eig_k)`` (steps when normalized).

    Using standard-deviation widths rather than raw covariance eigenvalues ensures
    ``shape_d2`` has the same squared-length / squared-step dimensions as ``pos_d2``,
    preventing ``px^4`` variance differences on large spots from overwhelming ``px^2``
    centroid distances.
    """
    e1, e2, e3 = _spot_eigs(spot, normalize_by_step=normalize_by_step)
    return (
        math.sqrt(max(0.0, e1)),
        math.sqrt(max(0.0, e2)),
        math.sqrt(max(0.0, e3)),
    )


def _spot_intensity(spot: dict) -> float:
    """Return pedestal-subtracted ``net_intensity`` when positive, falling back to ``integrated_intensity``."""
    net = float(spot.get("net_intensity", 0.0))
    if net > 0.0:
        return max(net, 1.0)
    return max(float(spot.get("integrated_intensity", 1.0)), 1.0)


def _spot_step_cov(spot: dict, *, normalize_by_step: bool = True) -> np.ndarray:
    """Reconstruct symmetric positive-definite 3x3 covariance in ``(mu, chi, d)`` coordinates.

    Uses full second moments ``(cov_zz, cov_xx, cov_yy, cov_zx, cov_zy, cov_yx)`` when
    available (preserving 3D streak tilt/orientation), or falls back to diagonal
    eigenvalues ``diag(eig_1, eig_2, eig_3)`` for minimal synthetic dicts. Adds the
    uniform voxel quantization floor ``(1/12) I_3`` in step coordinates.
    """
    sm, sc, sd = _spot_scales(spot) if normalize_by_step else (1.0, 1.0, 1.0)
    if all(k in spot for k in ("cov_zz", "cov_xx", "cov_yy")):
        czz = float(spot.get("cov_zz", 0.0)) / (sm * sm)
        cxx = float(spot.get("cov_xx", 0.0)) / (sc * sc)
        cyy = float(spot.get("cov_yy", 0.0)) / (sd * sd)
        czx = float(spot.get("cov_zx", 0.0)) / (sm * sc)
        czy = float(spot.get("cov_zy", 0.0)) / (sm * sd)
        cyx = float(spot.get("cov_yx", 0.0)) / (sd * sc)
        cov = np.array(
            [
                [czz, czx, czy],
                [czx, cxx, cyx],
                [czy, cyx, cyy],
            ],
            dtype=np.float64,
        )
    else:
        e1, e2, e3 = _spot_eigs(spot, normalize_by_step=normalize_by_step)
        cov = np.diag([max(0.0, e1), max(0.0, e2), max(0.0, e3)]).astype(np.float64)

    q_floor = (
        np.diag([1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0])
        if normalize_by_step
        else np.diag([(sm * sm) / 12.0, (sc * sc) / 12.0, (sd * sd) / 12.0])
    )
    vals, vecs = np.linalg.eigh(cov + q_floor)
    vals_clamped = np.maximum(vals, 1e-6)
    return (vecs * vals_clamped) @ vecs.T


class PositionShapeCost:
    """Weighted position + shape cost with per-axis gating and step-scale invariance.

    Parameters
    ----------
    position_weight : float
        Multiplier for the position (centroid) distance term.
    shape_weight : float
        Multiplier for the shape (root-eigenvalue semi-axis width) distance term.
    gate_mu, gate_chi, gate_d : float
        Maximum allowed displacement along each reciprocal-space axis (evaluated
        in the spot's native coordinate units). Pairs exceeding *any* gate are
        assigned ``inf`` cost.
    max_step_mu, max_step_chi, max_step_d : float
        Optional maximum allowed displacement measured in dimensionless voxel/frame
        steps ``(|delta| / step)``. Works identically whether spots are calibrated
        or uncalibrated.
    intensity_weight : float
        Optional multiplier for scale-free log-intensity ratio squared
        ``(ln(max(I_i, 1)) - ln(max(I_j, 1)))**2`` enforcing approximate grain
        volume conservation across consecutive scans. Defaults to ``0.0``.
    normalize_by_step : bool
        When ``True`` (default), calibrated physical displacements ``(deg, deg, Å)``
        are divided by ``(step_mu, step_chi, step_d)`` and ``sqrt(vox_eig_*)`` are used
        so Euclidean association costs are dimensionless and invariant to choice
        of physical units. Uncalibrated spots (``step_* = 1``) are unchanged.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        shape_weight: float = 0.5,
        gate_mu: float = math.inf,
        gate_chi: float = math.inf,
        gate_d: float = math.inf,
        *,
        max_step_mu: float = math.inf,
        max_step_chi: float = math.inf,
        max_step_d: float = math.inf,
        intensity_weight: float = 0.0,
        normalize_by_step: bool = True,
    ) -> None:
        self.position_weight = position_weight
        self.shape_weight = shape_weight
        self.gate_mu = gate_mu
        self.gate_chi = gate_chi
        self.gate_d = gate_d
        self.max_step_mu = float(max_step_mu)
        self.max_step_chi = float(max_step_chi)
        self.max_step_d = float(max_step_d)
        self.intensity_weight = float(intensity_weight)
        self.normalize_by_step = bool(normalize_by_step)

    def pairwise_cost_matrix(self, spots_t: list[dict], spots_t1: list[dict]) -> np.ndarray:
        n, m = len(spots_t), len(spots_t1)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)

        x = np.array(
            [[s["centroid_mu"], s["centroid_chi"], s["centroid_d"]] for s in spots_t],
            dtype=np.float64,
        )
        y = np.array(
            [[s["centroid_mu"], s["centroid_chi"], s["centroid_d"]] for s in spots_t1],
            dtype=np.float64,
        )
        w_t = np.array(
            [_spot_widths(s, normalize_by_step=self.normalize_by_step) for s in spots_t],
            dtype=np.float64,
        )
        w_t1 = np.array(
            [_spot_widths(s, normalize_by_step=self.normalize_by_step) for s in spots_t1],
            dtype=np.float64,
        )

        dmu = np.abs(x[:, 0][:, np.newaxis] - y[:, 0][np.newaxis, :])
        dchi = np.abs(x[:, 1][:, np.newaxis] - y[:, 1][np.newaxis, :])
        dd = np.abs(x[:, 2][:, np.newaxis] - y[:, 2][np.newaxis, :])
        bad = (dmu > self.gate_mu) | (dchi > self.gate_chi) | (dd > self.gate_d)

        sx = np.array([_spot_scales(s) for s in spots_t], dtype=np.float64)
        sy = np.array([_spot_scales(s) for s in spots_t1], dtype=np.float64)
        smu = 0.5 * (sx[:, 0][:, np.newaxis] + sy[:, 0][np.newaxis, :])
        schi = 0.5 * (sx[:, 1][:, np.newaxis] + sy[:, 1][np.newaxis, :])
        sd = 0.5 * (sx[:, 2][:, np.newaxis] + sy[:, 2][np.newaxis, :])
        step_dmu = dmu / smu
        step_dchi = dchi / schi
        step_dd = dd / sd
        bad |= (step_dmu > self.max_step_mu) | (step_dchi > self.max_step_chi) | (step_dd > self.max_step_d)

        if self.normalize_by_step:
            pos_d2 = step_dmu**2 + step_dchi**2 + step_dd**2
        else:
            pos_d2 = cdist(x, y, metric="sqeuclidean")

        shape_d2 = cdist(w_t, w_t1, metric="sqeuclidean")

        cost = self.position_weight * pos_d2 + self.shape_weight * shape_d2
        if self.intensity_weight > 0.0:
            log_ia = np.log([_spot_intensity(s) for s in spots_t])[:, np.newaxis]
            log_ib = np.log([_spot_intensity(s) for s in spots_t1])[np.newaxis, :]
            cost = cost + self.intensity_weight * (log_ia - log_ib) ** 2
        cost[bad] = np.inf
        return cost

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        dmu = abs(float(spot_i["centroid_mu"]) - float(spot_j["centroid_mu"]))
        dchi = abs(float(spot_i["centroid_chi"]) - float(spot_j["centroid_chi"]))
        dd = abs(float(spot_i["centroid_d"]) - float(spot_j["centroid_d"]))

        if dmu > self.gate_mu or dchi > self.gate_chi or dd > self.gate_d:
            return math.inf

        smi, sci, sdi = _spot_scales(spot_i)
        smj, scj, sdj = _spot_scales(spot_j)
        step_dmu = dmu / (0.5 * (smi + smj))
        step_dchi = dchi / (0.5 * (sci + scj))
        step_dd = dd / (0.5 * (sdi + sdj))

        if step_dmu > self.max_step_mu or step_dchi > self.max_step_chi or step_dd > self.max_step_d:
            return math.inf

        if self.normalize_by_step:
            pos_dist2 = step_dmu**2 + step_dchi**2 + step_dd**2
        else:
            pos_dist2 = dmu**2 + dchi**2 + dd**2

        wi = _spot_widths(spot_i, normalize_by_step=self.normalize_by_step)
        wj = _spot_widths(spot_j, normalize_by_step=self.normalize_by_step)
        shape_dist2 = (wi[0] - wj[0]) ** 2 + (wi[1] - wj[1]) ** 2 + (wi[2] - wj[2]) ** 2

        total = self.position_weight * pos_dist2 + self.shape_weight * shape_dist2
        if self.intensity_weight > 0.0:
            ia = _spot_intensity(spot_i)
            ib = _spot_intensity(spot_j)
            total += self.intensity_weight * (math.log(ia) - math.log(ib)) ** 2
        return total


class WassersteinBuresCost:
    """Closed-form 3D Gaussian Optimal Transport (2-Wasserstein / Bures) association cost.

    Models each reflection as an oriented 3D Gaussian intensity distribution
    ``N(m_i, Sigma_i)`` in dimensionless step coordinates ``(mu, chi, d)`` and evaluates
    the exact squared 2-Wasserstein distance plus scale-free log-intensity conservation:

    .. math::

        W_2^2(i, j) = \\|m_i - m_j\\|_2^2
            + b_w \\, \\mathrm{Tr}\\!\\left(\\Sigma_i + \\Sigma_j
            - 2(\\Sigma_i^{1/2} \\Sigma_j \\Sigma_i^{1/2})^{1/2}\\right)
            + w_I \\left(\\ln \\frac{I_i^{\\mathrm{net}}}{I_j^{\\mathrm{net}}}\\right)^2

    Unlike eigenvalue-only :class:`PositionShapeCost` (which is blind to ``+45°`` vs ``-45°``
    streak tilt when semi-axis lengths match), the Bures covariance term preserves full 3D
    eigenvector orientation in native ``step^2`` units without neural network inference.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        bures_weight: float = 1.0,
        gate_mu: float = math.inf,
        gate_chi: float = math.inf,
        gate_d: float = math.inf,
        *,
        max_step_mu: float = math.inf,
        max_step_chi: float = math.inf,
        max_step_d: float = math.inf,
        intensity_weight: float = 15.0,
        normalize_by_step: bool = True,
    ) -> None:
        self.position_weight = float(position_weight)
        self.bures_weight = float(bures_weight)
        self.gate_mu = float(gate_mu)
        self.gate_chi = float(gate_chi)
        self.gate_d = float(gate_d)
        self.max_step_mu = float(max_step_mu)
        self.max_step_chi = float(max_step_chi)
        self.max_step_d = float(max_step_d)
        self.intensity_weight = float(intensity_weight)
        self.normalize_by_step = bool(normalize_by_step)

    @staticmethod
    def _matrix_sqrt_psd(covs: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(covs)
        sqrt_vals = np.sqrt(np.maximum(vals, 0.0))
        return np.einsum("...ik,...k,...jk->...ij", vecs, sqrt_vals, vecs)

    def pairwise_cost_matrix(self, spots_t: list[dict], spots_t1: list[dict]) -> np.ndarray:
        n, m = len(spots_t), len(spots_t1)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)

        x = np.array([[s["centroid_mu"], s["centroid_chi"], s["centroid_d"]] for s in spots_t], dtype=np.float64)
        y = np.array([[s["centroid_mu"], s["centroid_chi"], s["centroid_d"]] for s in spots_t1], dtype=np.float64)

        dmu = np.abs(x[:, 0][:, np.newaxis] - y[:, 0][np.newaxis, :])
        dchi = np.abs(x[:, 1][:, np.newaxis] - y[:, 1][np.newaxis, :])
        dd = np.abs(x[:, 2][:, np.newaxis] - y[:, 2][np.newaxis, :])
        bad = (dmu > self.gate_mu) | (dchi > self.gate_chi) | (dd > self.gate_d)

        sx = np.array([_spot_scales(s) for s in spots_t], dtype=np.float64)
        sy = np.array([_spot_scales(s) for s in spots_t1], dtype=np.float64)
        smu = 0.5 * (sx[:, 0][:, np.newaxis] + sy[:, 0][np.newaxis, :])
        schi = 0.5 * (sx[:, 1][:, np.newaxis] + sy[:, 1][np.newaxis, :])
        sd = 0.5 * (sx[:, 2][:, np.newaxis] + sy[:, 2][np.newaxis, :])
        step_dmu = dmu / smu
        step_dchi = dchi / schi
        step_dd = dd / sd
        bad |= (step_dmu > self.max_step_mu) | (step_dchi > self.max_step_chi) | (step_dd > self.max_step_d)

        pos_d2 = (
            step_dmu**2 + step_dchi**2 + step_dd**2 if self.normalize_by_step else cdist(x, y, metric="sqeuclidean")
        )

        if self.bures_weight > 0.0:
            cov_t = np.stack([_spot_step_cov(s, normalize_by_step=self.normalize_by_step) for s in spots_t], axis=0)
            cov_t1 = np.stack([_spot_step_cov(s, normalize_by_step=self.normalize_by_step) for s in spots_t1], axis=0)
            sqrt_t = self._matrix_sqrt_psd(cov_t)
            # Symmetric PSD sandwich S_i @ Sigma_j @ S_i of shape (N, M, 3, 3)
            prod = np.einsum("nik,mkl,nlj->nmij", sqrt_t, cov_t1, sqrt_t)
            prod = 0.5 * (prod + np.swapaxes(prod, -1, -2))
            eigs_prod = np.linalg.eigvalsh(prod)
            tr_cross = np.sum(np.sqrt(np.maximum(eigs_prod, 0.0)), axis=-1)
            tr_t = np.trace(cov_t, axis1=-2, axis2=-1)[:, np.newaxis]
            tr_t1 = np.trace(cov_t1, axis1=-2, axis2=-1)[np.newaxis, :]
            bures_d2 = np.maximum(0.0, tr_t + tr_t1 - 2.0 * tr_cross)
        else:
            bures_d2 = 0.0

        cost = self.position_weight * pos_d2 + self.bures_weight * bures_d2
        if self.intensity_weight > 0.0:
            log_ia = np.log([_spot_intensity(s) for s in spots_t])[:, np.newaxis]
            log_ib = np.log([_spot_intensity(s) for s in spots_t1])[np.newaxis, :]
            cost = cost + self.intensity_weight * (log_ia - log_ib) ** 2
        cost[bad] = np.inf
        return cost

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        mat = self.pairwise_cost_matrix([spot_i], [spot_j])
        return float(mat[0, 0])


class GeometrySemanticCost:
    """α × geometry + β × (1 − cos(f_i, f_j)) with pluggable geometric cost.

    Embeddings must be L2-normalised; ``sim`` is the dot product.  When
    ``cost_beta`` is zero, the semantic branch is skipped and missing
    embeddings are ignored.  For ``cost_beta > 0``, missing embeddings
    yield infinite cost.
    """

    def __init__(
        self,
        geometry: CostFunction,
        *,
        cost_alpha: float = 1.0,
        cost_beta: float = 0.0,
    ) -> None:
        self.geometry = geometry
        self.cost_alpha = float(cost_alpha)
        self.cost_beta = float(cost_beta)

    def pairwise_cost_matrix(self, spots_t: list[dict], spots_t1: list[dict]) -> np.ndarray:
        geo = self.geometry.pairwise_cost_matrix(spots_t, spots_t1)
        if self.cost_beta == 0.0:
            return self.cost_alpha * geo

        dim: int | None = None
        for s in (*spots_t, *spots_t1):
            e = s.get("embedding")
            if e is None:
                continue
            v = np.asarray(e, dtype=np.float64).ravel()
            if v.size > 0:
                dim = int(v.size)
                break

        n, m = len(spots_t), len(spots_t1)
        if dim is None:
            out = np.empty((n, m), dtype=np.float64)
            out[:] = np.inf
            return out

        a_mat = np.zeros((n, dim), dtype=np.float64)
        ok_a = np.zeros(n, dtype=bool)
        for i, s in enumerate(spots_t):
            e = s.get("embedding")
            if e is None:
                continue
            v = np.asarray(e, dtype=np.float64).ravel()
            if v.shape == (dim,):
                a_mat[i] = v
                ok_a[i] = True

        b_mat = np.zeros((m, dim), dtype=np.float64)
        ok_b = np.zeros(m, dtype=bool)
        for j, s in enumerate(spots_t1):
            e = s.get("embedding")
            if e is None:
                continue
            v = np.asarray(e, dtype=np.float64).ravel()
            if v.shape == (dim,):
                b_mat[j] = v
                ok_b[j] = True

        sim = a_mat @ b_mat.T
        np.clip(sim, -1.0, 1.0, out=sim)
        geo_term = np.zeros_like(geo) if self.cost_alpha == 0.0 else self.cost_alpha * geo
        out = geo_term + self.cost_beta * (1.0 - sim)
        out[~np.isfinite(geo)] = np.inf
        out[~ok_a, :] = np.inf
        out[:, ~ok_b] = np.inf
        return out

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        g = self.geometry(spot_i, spot_j)
        if not math.isfinite(g):
            return g
        if self.cost_beta == 0.0:
            return 0.0 if self.cost_alpha == 0.0 else self.cost_alpha * g
        fi = spot_i.get("embedding")
        fj = spot_j.get("embedding")
        if fi is None or fj is None:
            return math.inf
        a = np.asarray(fi, dtype=np.float64).ravel()
        b = np.asarray(fj, dtype=np.float64).ravel()
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return math.inf
        sim = float(np.dot(a, b))
        sim = max(-1.0, min(1.0, sim))
        geo_val = 0.0 if self.cost_alpha == 0.0 else self.cost_alpha * g
        return geo_val + self.cost_beta * (1.0 - sim)
