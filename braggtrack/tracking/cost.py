"""Pluggable cost functions for frame-to-frame spot association.

Week 3 provides a physics-only baseline (position + shape).
Week 4 adds a semantic term (cosine on fused DINO-style embeddings).
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy.spatial.distance import cdist


class CostFunction(Protocol):
    """Interface that any association cost must satisfy."""

    def pairwise_cost_matrix(
        self, spots_t: list[dict], spots_t1: list[dict]
    ) -> np.ndarray:
        """Dense ``(N, M)`` costs; ``inf`` for gated or inadmissible pairs."""

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        """Scalar cost for one pair (tests, debugging).

        Should return ``math.inf`` when the pair is inadmissible (gated out).
        """
        ...


class PositionShapeCost:
    """Weighted position + shape cost with per-axis gating.

    Parameters
    ----------
    position_weight : float
        Multiplier for the position (centroid) distance term.
    shape_weight : float
        Multiplier for the shape (eigenvalue) distance term.
    gate_mu, gate_chi, gate_d : float
        Maximum allowed displacement along each reciprocal-space axis.
        Pairs exceeding *any* gate are assigned ``inf`` cost.  Defaults
        are deliberately loose; tighten ``gate_d`` less aggressively
        than mu/chi for operando data where d-spacing shifts are large.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        shape_weight: float = 0.5,
        gate_mu: float = math.inf,
        gate_chi: float = math.inf,
        gate_d: float = math.inf,
    ) -> None:
        self.position_weight = position_weight
        self.shape_weight = shape_weight
        self.gate_mu = gate_mu
        self.gate_chi = gate_chi
        self.gate_d = gate_d

    def pairwise_cost_matrix(
        self, spots_t: list[dict], spots_t1: list[dict]
    ) -> np.ndarray:
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
        e = np.array(
            [
                [s.get("eig_1", 0.0), s.get("eig_2", 0.0), s.get("eig_3", 0.0)]
                for s in spots_t
            ],
            dtype=np.float64,
        )
        f = np.array(
            [
                [s.get("eig_1", 0.0), s.get("eig_2", 0.0), s.get("eig_3", 0.0)]
                for s in spots_t1
            ],
            dtype=np.float64,
        )

        pos_d2 = cdist(x, y, metric="sqeuclidean")
        shape_d2 = cdist(e, f, metric="sqeuclidean")

        dmu = np.abs(x[:, 0][:, np.newaxis] - y[:, 0][np.newaxis, :])
        dchi = np.abs(x[:, 1][:, np.newaxis] - y[:, 1][np.newaxis, :])
        dd = np.abs(x[:, 2][:, np.newaxis] - y[:, 2][np.newaxis, :])
        bad = (dmu > self.gate_mu) | (dchi > self.gate_chi) | (dd > self.gate_d)

        cost = self.position_weight * pos_d2 + self.shape_weight * shape_d2
        cost[bad] = np.inf
        return cost

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        dmu = abs(spot_i["centroid_mu"] - spot_j["centroid_mu"])
        dchi = abs(spot_i["centroid_chi"] - spot_j["centroid_chi"])
        dd = abs(spot_i["centroid_d"] - spot_j["centroid_d"])

        if dmu > self.gate_mu or dchi > self.gate_chi or dd > self.gate_d:
            return math.inf

        pos_dist2 = dmu ** 2 + dchi ** 2 + dd ** 2

        deig1 = spot_i.get("eig_1", 0.0) - spot_j.get("eig_1", 0.0)
        deig2 = spot_i.get("eig_2", 0.0) - spot_j.get("eig_2", 0.0)
        deig3 = spot_i.get("eig_3", 0.0) - spot_j.get("eig_3", 0.0)
        shape_dist2 = deig1 ** 2 + deig2 ** 2 + deig3 ** 2

        return self.position_weight * pos_dist2 + self.shape_weight * shape_dist2


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

    def pairwise_cost_matrix(
        self, spots_t: list[dict], spots_t1: list[dict]
    ) -> np.ndarray:
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
        out = self.cost_alpha * geo + self.cost_beta * (1.0 - sim)
        out[~np.isfinite(geo)] = np.inf
        out[~ok_a, :] = np.inf
        out[:, ~ok_b] = np.inf
        return out

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        g = self.geometry(spot_i, spot_j)
        if not math.isfinite(g):
            return g
        if self.cost_beta == 0.0:
            return self.cost_alpha * g
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
        return self.cost_alpha * g + self.cost_beta * (1.0 - sim)
