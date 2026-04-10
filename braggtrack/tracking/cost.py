"""Pluggable cost functions for frame-to-frame spot association.

Week 3 provides a physics-only baseline (position + shape).
Week 4 will add a semantic term without touching assignment logic.
"""

from __future__ import annotations

import math
from typing import Protocol


class CostFunction(Protocol):
    """Interface that any association cost must satisfy."""

    def __call__(self, spot_i: dict, spot_j: dict) -> float:
        """Return the linking cost between two spot feature dicts.

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
