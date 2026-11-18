"""Hungarian (Jonker-Volgenant) global assignment with gating."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cost import CostFunction


def associate_frames(
    spots_t: list[dict],
    spots_t1: list[dict],
    cost_fn: CostFunction,
    max_cost: float = math.inf,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match spots between two consecutive frames.

    Parameters
    ----------
    spots_t, spots_t1 : list[dict]
        Feature-dict rows from :func:`extract_instance_table` for
        frame *t* and frame *t+1*.
    cost_fn : CostFunction
        Must implement ``pairwise_cost_matrix(spots_t, spots_t1)`` returning
        a dense ``(N, M)`` matrix (``inf`` for gated pairs).
    max_cost : float
        Hard upper bound — any assignment with cost > max_cost is
        discarded after the Hungarian solve.

    Returns
    -------
    matches : list[tuple[int, int]]
        ``(index_in_t, index_in_t1)`` for every accepted association.
    unmatched_t : list[int]
        Indices in *spots_t* with no match (terminations).
    unmatched_t1 : list[int]
        Indices in *spots_t1* with no match (births).
    """
    n_t = len(spots_t)
    n_t1 = len(spots_t1)

    if n_t == 0:
        return [], [], list(range(n_t1))
    if n_t1 == 0:
        return [], list(range(n_t)), []

    raw = np.asarray(cost_fn.pairwise_cost_matrix(spots_t, spots_t1), dtype=np.float64)
    admissible = np.isfinite(raw) & (raw <= max_cost)
    if not np.any(admissible):
        return [], list(range(n_t)), list(range(n_t1))

    # Jaqaman et al. (Nature Methods 2008) augmented (n_t + n_t1) x (n_t + n_t1) LAP:
    # Explicit death (top-right) and birth (bottom-left) diagonal blocks with zero
    # bottom-right transpose block ensure gated (inf) pairs never force cascades that
    # displace low-cost physical matches just to dodge a rectangular 1e18 sentinel.
    max_adm = float(np.max(raw[admissible]))
    if math.isfinite(max_cost):
        # Exact objective: minimize sum_{(i,j) in M} (C_ij - max_cost)
        # Prefer strict match when C_ij == max_cost via tiny positive tie-breaker.
        unmatched_cost = float(max_cost) + 1e-9 * max(1.0, abs(float(max_cost)))
    else:
        # Maximum-cardinality finite matching, breaking ties by minimum total cost.
        unmatched_cost = max(1.0, float(n_t + n_t1 + 1) * max(max_adm, 1.0) + 1.0)

    half_u = 0.5 * unmatched_cost
    big_m = max(1e12, float(n_t + n_t1 + 4) * unmatched_cost * 100.0)

    total_dim = n_t + n_t1
    aug = np.full((total_dim, total_dim), big_m, dtype=np.float64)
    aug[:n_t, :n_t1] = np.where(admissible, raw, big_m)
    # Top-right (n_t x n_t): spot termination slack on diagonal
    aug[np.arange(n_t), n_t1 + np.arange(n_t)] = half_u
    # Bottom-left (n_t1 x n_t1): spot birth slack on diagonal
    aug[n_t + np.arange(n_t1), np.arange(n_t1)] = half_u
    # Bottom-right (n_t1 x n_t): zero-cost dummy completion block
    aug[n_t:, n_t1:] = 0.0

    row_idx, col_idx = linear_sum_assignment(aug)

    matches: list[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_t1: set[int] = set()

    for r, c in zip(row_idx, col_idx):
        if r < n_t and c < n_t1 and admissible[r, c]:
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_t1.add(int(c))

    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_t1 = [j for j in range(n_t1) if j not in matched_t1]

    return matches, unmatched_t, unmatched_t1


def compute_assignment_margins(
    spots_t: list[dict],
    spots_t1: list[dict],
    matches: list[tuple[int, int]],
    cost_fn: CostFunction,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Compute ``(assigned_cost, margin)`` for each accepted Hungarian match.

    The margin is ``min(second_best_row - c_ij, second_best_col - c_ij)``,
    measuring how close a competing spot in frame *t* or *t+1* came to
    claiming the link (``math.inf`` when no finite competitor exists).
    """
    if not matches or not spots_t or not spots_t1:
        return {}

    raw = np.asarray(cost_fn.pairwise_cost_matrix(spots_t, spots_t1), dtype=np.float64)
    n_t, n_t1 = raw.shape
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for r, c in matches:
        c_ij = float(raw[r, c])
        if not math.isfinite(c_ij):
            out[(r, c)] = (math.inf, 0.0)
            continue
        row_others = [float(raw[r, k]) for k in range(n_t1) if k != c and math.isfinite(raw[r, k])]
        col_others = [float(raw[k, c]) for k in range(n_t) if k != r and math.isfinite(raw[k, c])]
        alt_costs = row_others + col_others
        margin = (min(alt_costs) - c_ij) if alt_costs else math.inf
        out[(r, c)] = (c_ij, float(margin))
    return out
