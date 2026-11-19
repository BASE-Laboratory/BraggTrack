"""Scoring utilities for evaluating BraggTrack trajectories against blind human ground-truth links."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx


def _classify_spot_crowding(scans_meta: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """Return ``(scan_idx, spot_idx)`` pairs belonging to crowded rocking layers (>=3 spots within 55 steps)."""
    crowded: set[tuple[int, int]] = set()
    for s_entry in scans_meta:
        s_idx = int(s_entry["scan_idx"])
        spots = s_entry.get("spots", [])
        for a in spots:
            sp_a = int(a["spot_idx"])
            if a.get("layer_density") == "crowded":
                crowded.add((s_idx, sp_a))
                continue
            mu_a = float(a["centroid_mu"])
            chi_a = float(a["centroid_chi"])
            d_a = float(a["centroid_d"])
            step_mu = max(abs(float(a.get("step_mu", 1.0))), 1e-12)
            step_chi = max(abs(float(a.get("step_chi", 1.0))), 1e-12)
            step_d = max(abs(float(a.get("step_d", 1.0))), 1e-12)
            n_neighbors = 0
            for b in spots:
                if int(b["spot_idx"]) == sp_a:
                    continue
                if abs(float(b["centroid_mu"]) - mu_a) / step_mu > 4.5:
                    continue
                dist_det = math.hypot(
                    (float(b["centroid_chi"]) - chi_a) / step_chi,
                    (float(b["centroid_d"]) - d_a) / step_d,
                )
                if dist_det <= 55.0:
                    n_neighbors += 1
            if n_neighbors >= 2:
                crowded.add((s_idx, sp_a))
    return crowded


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }


def extract_predicted_edges_from_graph(
    G: nx.DiGraph,
) -> tuple[set[tuple[int, int, int, int]], list[dict[str, Any]]]:
    """Extract consecutive ``(s_from, sp_from, s_to, sp_to)`` 1-to-1 edges and split/merge events from DAG ``G``."""
    pred_1to1: set[tuple[int, int, int, int]] = set()
    for u, v, edata in G.edges(data=True):
        if edata.get("edge_type", "continuation") in ("split", "merge"):
            continue
        du, dv = G.nodes[u], G.nodes[v]
        s_u, sp_u = int(du["scan_idx"]), int(du["spot_idx"])
        s_v, sp_v = int(dv["scan_idx"]), int(dv["spot_idx"])
        if s_v == s_u + 1:
            pred_1to1.add((s_u, sp_u, s_v, sp_v))

    pred_events: list[dict[str, Any]] = []
    for ev in G.graph.get("lineage_events", []):
        p_spots = [int(G.nodes[n]["spot_idx"]) for n in ev.parent_nodes if n in G.nodes]
        c_spots = [int(G.nodes[n]["spot_idx"]) for n in ev.child_nodes if n in G.nodes]
        pred_events.append(
            {
                "event_type": ev.event_type,
                "scan_from": int(ev.scan_idx_from),
                "scan_to": int(ev.scan_idx_to),
                "source_spots": sorted(p_spots),
                "target_spots": sorted(c_spots),
            }
        )
    return pred_1to1, pred_events


def extract_predicted_edges_from_tables(
    tracks_rows: list[dict[str, Any]],
    events_rows: list[dict[str, Any]] | None = None,
) -> tuple[set[tuple[int, int, int, int]], list[dict[str, Any]]]:
    """Extract consecutive 1-to-1 edges and split/merge events from serialized CSV rows."""
    by_tid: dict[int, list[tuple[int, int]]] = {}
    tid_spot_at_scan: dict[tuple[int, int], int] = {}
    for r in tracks_rows:
        tid = int(r["track_id"])
        s_idx = int(r["scan_idx"])
        sp_idx = int(r["spot_idx"])
        by_tid.setdefault(tid, []).append((s_idx, sp_idx))
        tid_spot_at_scan[(tid, s_idx)] = sp_idx

    pred_1to1: set[tuple[int, int, int, int]] = set()
    for obs_list in by_tid.values():
        obs_sorted = sorted(obs_list)
        for (s0, sp0), (s1, sp1) in zip(obs_sorted[:-1], obs_sorted[1:], strict=False):
            if s1 == s0 + 1:
                pred_1to1.add((s0, sp0, s1, sp1))

    pred_events: list[dict[str, Any]] = []
    for ev in events_rows or []:
        s_from = int(ev["scan_idx_from"])
        s_to = int(ev["scan_idx_to"])
        ptids = [int(x) for x in str(ev.get("parent_track_ids", "")).split(";") if x.strip()]
        ctids = [int(x) for x in str(ev.get("child_track_ids", "")).split(";") if x.strip()]
        p_spots = sorted(tid_spot_at_scan[(t, s_from)] for t in ptids if (t, s_from) in tid_spot_at_scan)
        c_spots = sorted(tid_spot_at_scan[(t, s_to)] for t in ctids if (t, s_to) in tid_spot_at_scan)
        pred_events.append(
            {
                "event_type": str(ev["event_type"]),
                "scan_from": s_from,
                "scan_to": s_to,
                "source_spots": p_spots,
                "target_spots": c_spots,
            }
        )
    return pred_1to1, pred_events


def score_predictions_against_gt(
    pred_1to1: set[tuple[int, int, int, int]],
    pred_events: list[dict[str, Any]],
    gt_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compute overall, isolated, and crowded 1-to-1 F1 plus split/merge recall against ``gt_payload``."""
    crowded_sources = _classify_spot_crowding(gt_payload.get("scans", []))
    gt_1to1: set[tuple[int, int, int, int]] = set()
    annotated_pairs: set[tuple[int, int]] = set()
    uncertain_sources: set[tuple[int, int]] = set()
    uncertain_targets: set[tuple[int, int]] = set()
    gt_splits: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    gt_merges: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []

    for link in gt_payload.get("links", []):
        s_from, s_to = int(link["scan_from"]), int(link["scan_to"])
        annotated_pairs.add((s_from, s_to))
        ltype = str(link["link_type"]).lower()
        srcs = [int(x) for x in link.get("source_spots", [])]
        tgts = [int(x) for x in link.get("target_spots", [])]
        if link.get("layer_density") == "crowded":
            for sp in srcs:
                crowded_sources.add((s_from, sp))

        if ltype == "uncertain":
            for sp in srcs:
                uncertain_sources.add((s_from, sp))
            for sp in tgts:
                uncertain_targets.add((s_to, sp))
        elif ltype == "1-to-1" and len(srcs) == 1 and len(tgts) == 1:
            gt_1to1.add((s_from, srcs[0], s_to, tgts[0]))
        elif ltype == "split" and len(srcs) == 1 and len(tgts) >= 2:
            gt_splits.append((s_from, s_to, tuple(sorted(srcs)), tuple(sorted(tgts))))
            for sp in srcs:
                uncertain_sources.add((s_from, sp))
            for sp in tgts:
                uncertain_targets.add((s_to, sp))
        elif ltype == "merge" and len(srcs) >= 2 and len(tgts) == 1:
            gt_merges.append((s_from, s_to, tuple(sorted(srcs)), tuple(sorted(tgts))))
            for sp in srcs:
                uncertain_sources.add((s_from, sp))
            for sp in tgts:
                uncertain_targets.add((s_to, sp))

    eval_pred_1to1 = {
        e
        for e in pred_1to1
        if (e[0], e[2]) in annotated_pairs
        and (e[0], e[1]) not in uncertain_sources
        and (e[2], e[3]) not in uncertain_targets
    }
    tp_edges = eval_pred_1to1 & gt_1to1
    fp_edges = eval_pred_1to1 - gt_1to1
    fn_edges = gt_1to1 - eval_pred_1to1

    tp_crowd = sum(1 for e in tp_edges if (e[0], e[1]) in crowded_sources)
    fp_crowd = sum(1 for e in fp_edges if (e[0], e[1]) in crowded_sources)
    fn_crowd = sum(1 for e in fn_edges if (e[0], e[1]) in crowded_sources)

    pred_split_set = {
        (int(ev["scan_from"]), int(ev["scan_to"]), tuple(ev["source_spots"]), tuple(ev["target_spots"]))
        for ev in pred_events
        if ev["event_type"] == "split" and (int(ev["scan_from"]), int(ev["scan_to"])) in annotated_pairs
    }
    pred_merge_set = {
        (int(ev["scan_from"]), int(ev["scan_to"]), tuple(ev["source_spots"]), tuple(ev["target_spots"]))
        for ev in pred_events
        if ev["event_type"] == "merge" and (int(ev["scan_from"]), int(ev["scan_to"])) in annotated_pairs
    }
    gt_split_set, gt_merge_set = set(gt_splits), set(gt_merges)

    return {
        "annotator_id": gt_payload.get("annotator_id", "unknown"),
        "n_annotated_scan_pairs": len(annotated_pairs),
        "n_gt_1to1": len(gt_1to1),
        "n_uncertain_excluded": len(uncertain_sources),
        "overall_1to1": _prf(len(tp_edges), len(fp_edges), len(fn_edges)),
        "isolated_1to1": _prf(len(tp_edges) - tp_crowd, len(fp_edges) - fp_crowd, len(fn_edges) - fn_crowd),
        "crowded_1to1": _prf(tp_crowd, fp_crowd, fn_crowd),
        "splits": _prf(
            len(pred_split_set & gt_split_set),
            len(pred_split_set - gt_split_set),
            len(gt_split_set - pred_split_set),
        ),
        "merges": _prf(
            len(pred_merge_set & gt_merge_set),
            len(pred_merge_set - gt_merge_set),
            len(gt_merge_set - pred_merge_set),
        ),
    }


def score_graph_against_gt(G: nx.DiGraph, gt_payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper scoring a :func:`build_tracks` graph directly against ``gt_payload``."""
    pred_1to1, pred_events = extract_predicted_edges_from_graph(G)
    return score_predictions_against_gt(pred_1to1, pred_events, gt_payload)
