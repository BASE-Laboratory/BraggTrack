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
            log_ia = np.log(np.maximum([float(s.get("integrated_intensity", 1.0)) for s in spots_t], 1.0))[
                :, np.newaxis
            ]
            log_ib = np.log(np.maximum([float(s.get("integrated_intensity", 1.0)) for s in spots_t1], 1.0))[
                np.newaxis, :
            ]
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
            ia = max(float(spot_i.get("integrated_intensity", 1.0)), 1.0)
            ib = max(float(spot_j.get("integrated_intensity", 1.0)), 1.0)
            total += self.intensity_weight * (math.log(ia) - math.log(ib)) ** 2
        return total


class GeometrySemanticCost:
    """α × geometry + β × (1 − cos(f_i, f_j)) with geometry from :class:`PositionShapeCost`.

    Embeddings must be L2-normalised; ``sim`` is the dot product.  When
    ``cost_beta`` is zero, the semantic branch is skipped and missing
    embeddings are ignored.  For ``cost_beta > 0``, missing embeddings
    yield infinite cost.
    """

    def __init__(
        self,
        geometry: PositionShapeCost,
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
