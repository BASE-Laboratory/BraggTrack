"""Multi-scan gap closing for transient beam dropouts and rocking-clip recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .cost import CostFunction, GeometrySemanticCost, PositionShapeCost, _spot_eigs, _spot_scales
from .lifecycle import TrackEvent


@dataclass(frozen=True)
class GapClosingConfig:
    """Configuration for reconnecting tracks across missing intermediate scans.

    Parameters
    ----------
    max_gap : int
        Maximum number of missing intermediate scans (``1 <= gap_scans <= max_gap``).
        ``0`` disables gap closing.
    gap_penalty_per_scan : float
        Multiplicative penalty per missing scan: ``cost * (1 + gap_penalty_per_scan * gap_scans)``.
    max_cost : float
        Upper bound on the penalized association cost for accepting a gap bridge.
    min_cosine_similarity : float
        Minimum embedding cosine similarity when both endpoints carry ``embedding``.
    """

    max_gap: int = 2
    gap_penalty_per_scan: float = 0.15
    max_cost: float = 25.0
    min_cosine_similarity: float = 0.50


@dataclass(frozen=True)
class GapCloseRecord:
    """Metadata record for an accepted multi-scan gap closure."""

    track_id: int
    from_scan: int
    to_scan: int
    gap_scans: int
    from_node: str
    to_node: str
    cost: float


def _cosine_sim(a_raw: Any, b_raw: Any) -> float | None:
    if a_raw is None or b_raw is None:
        return None
    a = np.asarray(a_raw, dtype=np.float64).ravel()
    b = np.asarray(b_raw, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return None
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _eval_gap_cost(
    cost_fn: CostFunction,
    u_data: dict[str, Any],
    v_data: dict[str, Any],
    delta_scans: int,
    min_cos_sim: float,
) -> float:
    """Evaluate gap-normalized cost across ``delta_scans = to_scan - from_scan >= 2``."""
    sim = _cosine_sim(u_data.get("embedding"), v_data.get("embedding"))
    if sim is not None and sim < min_cos_sim:
        return math.inf

    geo: PositionShapeCost | None = None
    alpha = 1.0
    beta = 0.0
    if isinstance(cost_fn, PositionShapeCost):
        geo = cost_fn
    elif isinstance(cost_fn, GeometrySemanticCost):
        geo = cost_fn.geometry
        alpha = cost_fn.cost_alpha
        beta = cost_fn.cost_beta

    if geo is not None:
        dmu = abs(float(u_data["centroid_mu"]) - float(v_data["centroid_mu"]))
        dchi = abs(float(u_data["centroid_chi"]) - float(v_data["centroid_chi"]))
        dd = abs(float(u_data["centroid_d"]) - float(v_data["centroid_d"]))
        step_scale = float(max(1, delta_scans))
        if (dmu / step_scale) > geo.gate_mu or (dchi / step_scale) > geo.gate_chi or (dd / step_scale) > geo.gate_d:
            return math.inf

        if geo.normalize_by_step:
            smu, schi, sd = _spot_scales(u_data)
            smv, schiv, sdv = _spot_scales(v_data)
            dmu /= 0.5 * (smu + smv)
            dchi /= 0.5 * (schi + schiv)
            dd /= 0.5 * (sd + sdv)

        # Diffusive scaling (dx^2 / delta_t) keeps gap costs comparable to 1-step costs.
        pos_dist2 = (dmu**2 + dchi**2 + dd**2) / step_scale
        eu = _spot_eigs(u_data, normalize_by_step=geo.normalize_by_step)
        ev = _spot_eigs(v_data, normalize_by_step=geo.normalize_by_step)
        shape_dist2 = (eu[0] - ev[0]) ** 2 + (eu[1] - ev[1]) ** 2 + (eu[2] - ev[2]) ** 2
        g_cost = geo.position_weight * pos_dist2 + geo.shape_weight * shape_dist2
        if beta == 0.0:
            return alpha * g_cost
        if sim is None:
            return math.inf
        return alpha * g_cost + beta * (1.0 - sim)

    return float(cost_fn(u_data, v_data))


def _propagate_track_id_forward(G: nx.DiGraph, start_nid: str, old_tid: int, new_tid: int) -> None:
    queue = [start_nid]
    visited: set[str] = set()
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        if G.nodes[curr].get("track_id") == old_tid:
            G.nodes[curr]["track_id"] = new_tid
            for _, succ in G.out_edges(curr):
                queue.append(succ)


def resolve_track_gaps(
    G: nx.DiGraph,
    *,
    n_scans: int,
    cost_fn: CostFunction | None = None,
    config: GapClosingConfig | None = None,
) -> list[GapCloseRecord]:
    """Reconnect prematurely terminated tracks to matching births across ``1..max_gap`` missing scans."""
    cfg = config or GapClosingConfig()
    if cfg.max_gap <= 0 or G.number_of_nodes() == 0 or n_scans < 3:
        G.graph.setdefault("gap_closures", [])
        return []

    active_cost: CostFunction = cost_fn if cost_fn is not None else PositionShapeCost()
    next_tid = max((int(G.nodes[n]["track_id"]) for n in G.nodes), default=0) + 1

    # Pass 0: Sever forced cross-detector 1-to-1 jump edges created when max_cost=inf
    # paired a dropping-out track with a distant stray spot on an intermediate frame.
    nodes_by_scan: dict[int, list[str]] = {}
    for nid in G.nodes:
        nodes_by_scan.setdefault(int(G.nodes[nid]["scan_idx"]), []).append(nid)

    sever_threshold = max(cfg.max_cost * 4.0, 100.0)

    def _edge_is_jump(a: str, b: str) -> bool:
        c_ab = float(active_cost(G.nodes[a], G.nodes[b]))
        return not math.isfinite(c_ab) or c_ab > sever_threshold

    edges_to_sever: list[tuple[str, str]] = []
    for u, v, edata in list(G.edges(data=True)):
        if edata.get("edge_type", "continuation") != "continuation":
            continue
        if not _edge_is_jump(u, v):
            continue
        su = int(G.nodes[u]["scan_idx"])
        sv = int(G.nodes[v]["scan_idx"])
        has_gap_partner = False
        # Forward check: u can bridge over the jump to w at su + gap + 1
        for gap in range(1, cfg.max_gap + 1):
            for w in nodes_by_scan.get(su + gap + 1, []):
                if G.in_degree(w) == 0 or all(_edge_is_jump(p, w) for p, _ in G.in_edges(w)):
                    gc = _eval_gap_cost(active_cost, G.nodes[u], G.nodes[w], gap + 1, cfg.min_cosine_similarity)
                    if gc * (1.0 + cfg.gap_penalty_per_scan * gap) <= cfg.max_cost * 0.6:
                        has_gap_partner = True
                        break
            if has_gap_partner:
                break
        # Backward check: v can receive a bridge from p at sv - gap - 1
        if not has_gap_partner:
            for gap in range(1, cfg.max_gap + 1):
                for p in nodes_by_scan.get(sv - gap - 1, []):
                    if G.out_degree(p) == 0 or all(_edge_is_jump(p, q) for _, q in G.out_edges(p)):
                        gc = _eval_gap_cost(active_cost, G.nodes[p], G.nodes[v], gap + 1, cfg.min_cosine_similarity)
                        if gc * (1.0 + cfg.gap_penalty_per_scan * gap) <= cfg.max_cost * 0.6:
                            has_gap_partner = True
                            break
                if has_gap_partner:
                    break
        if has_gap_partner:
            edges_to_sever.append((u, v))

    for u, v in edges_to_sever:
        old_tid = int(G.nodes[u]["track_id"])
        G.remove_edge(u, v)
        G.nodes[u]["event"] = TrackEvent.TERMINATED
        G.nodes[v]["event"] = TrackEvent.BORN
        _propagate_track_id_forward(G, v, old_tid, next_tid)
        next_tid += 1

    term_nodes = sorted(
        [n for n in G.nodes if G.out_degree(n) == 0 and int(G.nodes[n]["scan_idx"]) < n_scans - 1],
        key=lambda n: (int(G.nodes[n]["scan_idx"]), int(G.nodes[n]["spot_idx"])),
    )
    born_nodes = sorted(
        [n for n in G.nodes if G.in_degree(n) == 0 and int(G.nodes[n]["scan_idx"]) > 0],
        key=lambda n: (int(G.nodes[n]["scan_idx"]), int(G.nodes[n]["spot_idx"])),
    )

    n_term = len(term_nodes)
    n_born = len(born_nodes)
    if n_term == 0 or n_born == 0:
        G.graph["gap_closures"] = []
        G.graph["gap_closed_count"] = 0
        return []

    inadmissible = 1e12
    unmatched_cost = cfg.max_cost + 1e-6
    aug_matrix = np.full((n_term, n_born + n_term), inadmissible, dtype=np.float64)
    for i in range(n_term):
        aug_matrix[i, n_born + i] = unmatched_cost

    for i, u in enumerate(term_nodes):
        u_data = G.nodes[u]
        su = int(u_data["scan_idx"])
        for j, v in enumerate(born_nodes):
            v_data = G.nodes[v]
            sv = int(v_data["scan_idx"])
            gap_scans = sv - su - 1
            if gap_scans < 1 or gap_scans > cfg.max_gap:
                continue
            base = _eval_gap_cost(active_cost, u_data, v_data, gap_scans + 1, cfg.min_cosine_similarity)
            if not math.isfinite(base):
                continue
            penalized = base * (1.0 + cfg.gap_penalty_per_scan * gap_scans)
            if penalized <= cfg.max_cost:
                aug_matrix[i, j] = penalized

    row_ind, col_ind = linear_sum_assignment(aug_matrix)

    accepted_pairs: list[tuple[int, int, str, str, float]] = []
    for r, c in zip(row_ind, col_ind):
        if c < n_born and aug_matrix[r, c] <= cfg.max_cost:
            u = term_nodes[int(r)]
            v = born_nodes[int(c)]
            su = int(G.nodes[u]["scan_idx"])
            sv = int(G.nodes[v]["scan_idx"])
            accepted_pairs.append((su, sv, u, v, float(aug_matrix[r, c])))

    # Process chronologically so multi-gap chains (e.g., 0 -> 2 -> 4) propagate a single track_id.
    accepted_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    records: list[GapCloseRecord] = []
    for su, sv, u, v, cost_val in accepted_pairs:
        tid = int(G.nodes[u]["track_id"])
        old_v_tid = int(G.nodes[v]["track_id"])
        gap_scans = sv - su - 1

        G.add_edge(u, v, edge_type="gap_close", gap_scans=gap_scans, cost=round(cost_val, 4))
        _propagate_track_id_forward(G, v, old_v_tid, tid)

        G.nodes[u]["event"] = TrackEvent.BORN if G.in_degree(u) == 0 else TrackEvent.CONTINUED
        if G.out_degree(v) > 0 or sv == n_scans - 1:
            G.nodes[v]["event"] = TrackEvent.CONTINUED
        else:
            G.nodes[v]["event"] = TrackEvent.TERMINATED

        records.append(
            GapCloseRecord(
                track_id=tid,
                from_scan=su,
                to_scan=sv,
                gap_scans=gap_scans,
                from_node=u,
                to_node=v,
                cost=round(cost_val, 4),
            )
        )

    G.graph["gap_closures"] = records
    G.graph["gap_closed_count"] = len(records)
    return records
