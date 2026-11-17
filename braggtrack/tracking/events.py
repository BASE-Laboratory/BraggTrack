"""Chemomechanical split (1 -> 2+) and merge (2+ -> 1) lineage resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SplitMergeConfig:
    """Configuration for post-assignment split and merge detection."""

    max_distance: float = 25.0
    gate_mu: float = math.inf
    gate_chi: float = math.inf
    gate_d: float = math.inf
    min_intensity_ratio: float = 0.4
    max_intensity_ratio: float = 1.6
    min_fragment_share: float = 0.1
    min_semantic_similarity: float | None = None


@dataclass(frozen=True)
class LineageEvent:
    """Explicit split or merge event in the trajectory DAG."""

    event_id: int
    event_type: Literal["split", "merge"]
    scan_idx_from: int
    scan_idx_to: int
    parent_track_ids: tuple[int, ...]
    child_track_ids: tuple[int, ...]
    parent_nodes: tuple[str, ...]
    child_nodes: tuple[str, ...]
    centroid_mu: float
    centroid_chi: float
    centroid_d: float
    intensity_ratio: float
    mean_distance: float
    semantic_similarity: float | None = None
    confidence: float = 1.0
    timestamp: str | None = None


def _spot_coords(node_data: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(node_data["centroid_mu"]),
        float(node_data["centroid_chi"]),
        float(node_data["centroid_d"]),
    )


def _pair_distance_and_sim(
    a: dict[str, Any],
    b: dict[str, Any],
    config: SplitMergeConfig,
) -> tuple[float, float | None] | None:
    mu_a, chi_a, d_a = _spot_coords(a)
    mu_b, chi_b, d_b = _spot_coords(b)
    dmu = abs(mu_a - mu_b)
    dchi = abs(chi_a - chi_b)
    dd = abs(d_a - d_b)
    if dmu > config.gate_mu or dchi > config.gate_chi or dd > config.gate_d:
        return None
    dist = math.sqrt(dmu * dmu + dchi * dchi + dd * dd)
    if dist > config.max_distance:
        return None

    sim: float | None = None
    emb_a = a.get("embedding")
    emb_b = b.get("embedding")
    if emb_a is not None and emb_b is not None:
        va = np.asarray(emb_a, dtype=np.float64).ravel()
        vb = np.asarray(emb_b, dtype=np.float64).ravel()
        if va.size > 0 and va.shape == vb.shape:
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
            if na > 0 and nb > 0:
                sim = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
                if config.min_semantic_similarity is not None and sim < config.min_semantic_similarity:
                    return None
    elif config.min_semantic_similarity is not None:
        return None

    return dist, sim


def _event_confidence(
    mean_dist: float,
    max_dist: float,
    intensity_ratio: float,
    sem_sim: float | None,
) -> float:
    dist_score = max(0.0, 1.0 - (mean_dist / max(max_dist, 1e-6)))
    int_score = max(0.0, 1.0 - abs(intensity_ratio - 1.0))
    if sem_sim is not None:
        sem_score = max(0.0, 0.5 * (sem_sim + 1.0))
        raw = 0.45 * dist_score + 0.35 * int_score + 0.20 * sem_score
    else:
        raw = 0.55 * dist_score + 0.45 * int_score
    return round(float(np.clip(raw, 0.05, 1.0)), 4)


def _propagate_track_id(G: nx.DiGraph, start_nid: str, old_tid: int, new_tid: int) -> None:
    """Update track_id along simple continuation chain starting at start_nid."""
    curr: str | None = start_nid
    while curr is not None and G.nodes[curr].get("track_id") == old_tid:
        G.nodes[curr]["track_id"] = new_tid
        succs = [
            v
            for v in G.successors(curr)
            if G.edges[curr, v].get("edge_type", "continuation") == "continuation"
            and G.nodes[v].get("track_id") == old_tid
        ]
        curr = succs[0] if len(succs) == 1 else None


def resolve_splits_and_merges(
    G: nx.DiGraph,
    n_scans: int,
    config: SplitMergeConfig | None = None,
    *,
    scan_timestamps: dict[int, str | None] | None = None,
) -> list[LineageEvent]:
    """Augment a 1-to-1 trajectory graph with split (1 -> 2+) and merge (2+ -> 1) edges.

    Modifies *G* in place (adding ``edge_type``, ``lineage_role``, and
    ``parent_track_ids`` attributes, and assigning clean post-split / post-merge
    track IDs) and returns the ordered list of :class:`LineageEvent` records.
    """
    cfg = config or SplitMergeConfig()
    if len(G.nodes) == 0 or n_scans < 2:
        G.graph["lineage_events"] = []
        return []

    for u, v in G.edges:
        G.edges[u, v].setdefault("edge_type", "continuation")

    nodes_by_scan: dict[int, list[str]] = {t: [] for t in range(n_scans)}
    max_tid = 0
    for nid in sorted(G.nodes):
        s_idx = int(G.nodes[nid]["scan_idx"])
        if s_idx in nodes_by_scan:
            nodes_by_scan[s_idx].append(nid)
        max_tid = max(max_tid, int(G.nodes[nid]["track_id"]))

    next_tid = max_tid + 1
    events: list[LineageEvent] = []
    next_event_id = 1

    for t in range(n_scans - 1):
        scan_t_nodes = nodes_by_scan[t]
        scan_t1_nodes = nodes_by_scan[t + 1]
        if not scan_t_nodes or not scan_t1_nodes:
            continue

        ts = scan_timestamps.get(t + 1) if scan_timestamps else None

        # --- Pass 0: Sever forced long-jump 1-to-1 Hungarian edges ---
        # Maximum-cardinality Hungarian assignment (when max_cost=inf) will pair an
        # unrelated leftover merge-source in scan t with an unrelated split-daughter
        # in scan t+1 across the detector. Sever any continuation edge (u, v) that
        # either violates SplitMergeConfig gates/max_distance OR where both u and v
        # have local neighbours at < 50% of dist(u, v).
        from .lifecycle import TrackEvent

        existing_edges = [
            (u, v)
            for u in scan_t_nodes
            for v in list(G.successors(u))
            if v in scan_t1_nodes and G.edges[u, v].get("edge_type", "continuation") == "continuation"
        ]
        for u, v in existing_edges:
            uv_res = _pair_distance_and_sim(G.nodes[u], G.nodes[v], cfg)
            sever = uv_res is None
            if not sever and uv_res is not None:
                d_uv = uv_res[0]
                min_u_other = min(
                    (
                        r[0]
                        for c in scan_t1_nodes
                        if c != v and (r := _pair_distance_and_sim(G.nodes[u], G.nodes[c], cfg)) is not None
                    ),
                    default=math.inf,
                )
                min_v_other = min(
                    (
                        r[0]
                        for p in scan_t_nodes
                        if p != u and (r := _pair_distance_and_sim(G.nodes[p], G.nodes[v], cfg)) is not None
                    ),
                    default=math.inf,
                )
                if min_u_other < 0.5 * d_uv and min_v_other < 0.5 * d_uv:
                    sever = True
            if sever:
                G.remove_edge(u, v)
                G.nodes[u]["event"] = TrackEvent.TERMINATED
                old_v_tid = int(G.nodes[v]["track_id"])
                fresh_tid = next_tid
                next_tid += 1
                _propagate_track_id(G, v, old_v_tid, fresh_tid)
                G.nodes[v]["event"] = TrackEvent.BORN

        # --- Pass 1: Splits (1 -> 2+) ---
        unmatched_children = [nid for nid in scan_t1_nodes if G.in_degree(nid) == 0]
        parent_to_candidates: dict[str, list[tuple[str, float, float | None]]] = {}
        for c_nid in unmatched_children:
            best_p: str | None = None
            best_dist = math.inf
            best_sim: float | None = None
            for p_nid in scan_t_nodes:
                res = _pair_distance_and_sim(G.nodes[p_nid], G.nodes[c_nid], cfg)
                if res is not None and res[0] < best_dist:
                    best_p = p_nid
                    best_dist, best_sim = res
            if best_p is not None:
                parent_to_candidates.setdefault(best_p, []).append((c_nid, best_dist, best_sim))

        split_parents_used: set[str] = set()
        for p_nid, extra_list in sorted(parent_to_candidates.items()):
            existing_succs = list(G.successors(p_nid))
            all_children: list[str] = list(existing_succs) + [item[0] for item in extra_list]
            if len(all_children) < 2:
                continue

            i_parent = float(G.nodes[p_nid].get("integrated_intensity", 0.0))
            child_intensities = [max(0.0, float(G.nodes[c].get("integrated_intensity", 0.0))) for c in all_children]
            i_sum = sum(child_intensities)
            if i_parent <= 0.0 or i_sum <= 0.0:
                continue

            ratio = i_sum / i_parent
            if not (cfg.min_intensity_ratio <= ratio <= cfg.max_intensity_ratio):
                continue
            if any((ic / i_sum) < cfg.min_fragment_share for ic in child_intensities):
                continue

            # Compute mean distance and mean semantic similarity across all children.
            dists: list[float] = []
            sims: list[float] = []
            valid_all = True
            for c_nid in all_children:
                res = _pair_distance_and_sim(G.nodes[p_nid], G.nodes[c_nid], cfg)
                if res is None:
                    valid_all = False
                    break
                dists.append(res[0])
                if res[1] is not None:
                    sims.append(res[1])
            if not valid_all:
                continue

            parent_tid = int(G.nodes[p_nid]["track_id"])
            G.nodes[p_nid]["lineage_role"] = "split_parent"
            child_tids: list[int] = []
            for c_nid in sorted(all_children):
                if not G.has_edge(p_nid, c_nid):
                    G.add_edge(p_nid, c_nid, edge_type="split")
                else:
                    G.edges[p_nid, c_nid]["edge_type"] = "split"

                old_c_tid = int(G.nodes[c_nid]["track_id"])
                if old_c_tid == parent_tid:
                    new_c_tid = next_tid
                    next_tid += 1
                    _propagate_track_id(G, c_nid, old_c_tid, new_c_tid)
                else:
                    new_c_tid = old_c_tid

                G.nodes[c_nid]["lineage_role"] = "split_child"
                G.nodes[c_nid]["parent_track_ids"] = [parent_tid]
                child_tids.append(new_c_tid)

            mean_dist = float(np.mean(dists))
            mean_sim = float(np.mean(sims)) if sims else None
            p_mu, p_chi, p_d = _spot_coords(G.nodes[p_nid])
            events.append(
                LineageEvent(
                    event_id=next_event_id,
                    event_type="split",
                    scan_idx_from=t,
                    scan_idx_to=t + 1,
                    parent_track_ids=(parent_tid,),
                    child_track_ids=tuple(child_tids),
                    parent_nodes=(p_nid,),
                    child_nodes=tuple(sorted(all_children)),
                    centroid_mu=p_mu,
                    centroid_chi=p_chi,
                    centroid_d=p_d,
                    intensity_ratio=round(ratio, 4),
                    mean_distance=round(mean_dist, 4),
                    semantic_similarity=round(mean_sim, 4) if mean_sim is not None else None,
                    confidence=_event_confidence(mean_dist, cfg.max_distance, ratio, mean_sim),
                    timestamp=ts,
                )
            )
            next_event_id += 1
            split_parents_used.add(p_nid)

        # --- Pass 2: Merges (2+ -> 1) ---
        unmatched_parents = [nid for nid in scan_t_nodes if G.out_degree(nid) == 0 and nid not in split_parents_used]
        target_to_candidates: dict[str, list[tuple[str, float, float | None]]] = {}
        for p_nid in unmatched_parents:
            best_c: str | None = None
            best_dist = math.inf
            best_sim: float | None = None
            for c_nid in scan_t1_nodes:
                if G.nodes[c_nid].get("lineage_role") == "split_child":
                    continue
                res = _pair_distance_and_sim(G.nodes[p_nid], G.nodes[c_nid], cfg)
                if res is not None and res[0] < best_dist:
                    best_c = c_nid
                    best_dist, best_sim = res
            if best_c is not None:
                target_to_candidates.setdefault(best_c, []).append((p_nid, best_dist, best_sim))

        for c_nid, extra_parents in sorted(target_to_candidates.items()):
            existing_preds = list(G.predecessors(c_nid))
            all_parents: list[str] = list(existing_preds) + [item[0] for item in extra_parents]
            if len(all_parents) < 2:
                continue

            i_target = float(G.nodes[c_nid].get("integrated_intensity", 0.0))
            parent_intensities = [max(0.0, float(G.nodes[p].get("integrated_intensity", 0.0))) for p in all_parents]
            i_parents_sum = sum(parent_intensities)
            if i_target <= 0.0 or i_parents_sum <= 0.0:
                continue

            ratio = i_target / i_parents_sum
            if not (cfg.min_intensity_ratio <= ratio <= cfg.max_intensity_ratio):
                continue
            if any((ip / i_parents_sum) < cfg.min_fragment_share for ip in parent_intensities):
                continue

            dists = []
            sims = []
            valid_all = True
            for p_nid in all_parents:
                res = _pair_distance_and_sim(G.nodes[p_nid], G.nodes[c_nid], cfg)
                if res is None:
                    valid_all = False
                    break
                dists.append(res[0])
                if res[1] is not None:
                    sims.append(res[1])
            if not valid_all:
                continue

            parent_tids: list[int] = []
            for p_nid in sorted(all_parents):
                if not G.has_edge(p_nid, c_nid):
                    G.add_edge(p_nid, c_nid, edge_type="merge")
                else:
                    G.edges[p_nid, c_nid]["edge_type"] = "merge"
                G.nodes[p_nid]["lineage_role"] = "merge_source"
                parent_tids.append(int(G.nodes[p_nid]["track_id"]))

            old_c_tid = int(G.nodes[c_nid]["track_id"])
            merged_tid = next_tid
            next_tid += 1
            _propagate_track_id(G, c_nid, old_c_tid, merged_tid)
            G.nodes[c_nid]["lineage_role"] = "merge_target"
            G.nodes[c_nid]["parent_track_ids"] = sorted(parent_tids)

            mean_dist = float(np.mean(dists))
            mean_sim = float(np.mean(sims)) if sims else None
            c_mu, c_chi, c_d = _spot_coords(G.nodes[c_nid])
            events.append(
                LineageEvent(
                    event_id=next_event_id,
                    event_type="merge",
                    scan_idx_from=t,
                    scan_idx_to=t + 1,
                    parent_track_ids=tuple(sorted(parent_tids)),
                    child_track_ids=(merged_tid,),
                    parent_nodes=tuple(sorted(all_parents)),
                    child_nodes=(c_nid,),
                    centroid_mu=c_mu,
                    centroid_chi=c_chi,
                    centroid_d=c_d,
                    intensity_ratio=round(ratio, 4),
                    mean_distance=round(mean_dist, 4),
                    semantic_similarity=round(mean_sim, 4) if mean_sim is not None else None,
                    confidence=_event_confidence(mean_dist, cfg.max_distance, ratio, mean_sim),
                    timestamp=ts,
                )
            )
            next_event_id += 1

    G.graph["lineage_events"] = events
    return events


def events_to_table(events: list[LineageEvent]) -> list[dict[str, Any]]:
    """Serialize :class:`LineageEvent` records into CSV/JSON-friendly row dicts."""
    rows: list[dict[str, Any]] = []
    for ev in events:
        rows.append(
            {
                "event_id": ev.event_id,
                "event_type": ev.event_type,
                "scan_idx_from": ev.scan_idx_from,
                "scan_idx_to": ev.scan_idx_to,
                "timestamp": ev.timestamp or "",
                "parent_track_ids": ";".join(str(x) for x in ev.parent_track_ids),
                "child_track_ids": ";".join(str(x) for x in ev.child_track_ids),
                "parent_nodes": ";".join(ev.parent_nodes),
                "child_nodes": ";".join(ev.child_nodes),
                "centroid_mu": ev.centroid_mu,
                "centroid_chi": ev.centroid_chi,
                "centroid_d": ev.centroid_d,
                "intensity_ratio": ev.intensity_ratio,
                "mean_distance": ev.mean_distance,
                "semantic_similarity": ev.semantic_similarity if ev.semantic_similarity is not None else "",
                "confidence": ev.confidence,
            }
        )
    return rows
