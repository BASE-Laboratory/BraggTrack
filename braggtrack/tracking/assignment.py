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
        Callable returning the linking cost (``inf`` for gated pairs).
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

    # Build cost matrix; use a large sentinel for gated-out pairs so
    # the solver can still operate on a dense matrix.
    SENTINEL = 1e18
    cost_matrix = np.full((n_t, n_t1), SENTINEL, dtype=np.float64)
    for i, si in enumerate(spots_t):
        for j, sj in enumerate(spots_t1):
            c = cost_fn(si, sj)
            if math.isfinite(c):
                cost_matrix[i, j] = c

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_t1: set[int] = set()

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < SENTINEL and cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_t1.add(int(c))

    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_t1 = [j for j in range(n_t1) if j not in matched_t1]

    return matches, unmatched_t, unmatched_t1
