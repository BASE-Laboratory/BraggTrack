"""Tracking quality metrics: ID-switch rate and track fragmentation."""

from __future__ import annotations

from typing import Any

import networkx as nx


def compute_tracking_metrics(
    G: nx.DiGraph,
    n_scans: int,
    ground_truth: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Compute tracking quality metrics from the trajectory DAG.

    Parameters
    ----------
    G : nx.DiGraph
        Trajectory graph produced by :func:`build_tracks`.
    n_scans : int
        Total number of scans in the sequence.
    ground_truth : list[dict[str, int]] | None
        Optional list of ``{"scan_idx": ..., "spot_idx": ..., "true_id": ...}``
        rows.  When provided, ID-switch rate is computed by comparing
        predicted ``track_id`` against ``true_id`` for each continued
        observation.

    Returns
    -------
    dict
        ``total_tracks``, ``full_length_tracks``, ``fragmentation_ratio``,
        ``born_count``, ``terminated_count``, ``continued_count``,
        ``id_switch_count``, ``id_switch_rate``.
    """
    # Collect per-track info.
    tracks: dict[int, list[dict]] = {}
    born_count = 0
    continued_count = 0
    terminated_count = 0

    for nid in G.nodes:
        data = G.nodes[nid]
        tid = data["track_id"]
        tracks.setdefault(tid, []).append(data)
        event = data["event"]
        event_val = event.value if hasattr(event, "value") else event
        if event_val == "born":
            born_count += 1
        elif event_val == "continued":
            continued_count += 1
        elif event_val == "terminated":
            terminated_count += 1

    total_tracks = len(tracks)
    full_length_tracks = sum(
        1 for obs in tracks.values() if len(obs) >= n_scans
    )
    fragmentation_ratio = (
        1.0 - full_length_tracks / total_tracks if total_tracks > 0 else 0.0
    )

    # ID-switch rate (requires ground truth).
    id_switch_count = 0
    id_switch_rate = 0.0
    if ground_truth is not None:
        gt_map: dict[tuple[int, int], int] = {}
        for row in ground_truth:
            gt_map[(row["scan_idx"], row["spot_idx"])] = row["true_id"]

        # For each predicted continuation edge, check if the true_id
        # changed while the track_id stayed the same, or if the
        # track_id stayed but the true_id flipped.
        pred_map: dict[tuple[int, int], int] = {}
        for nid in G.nodes:
            data = G.nodes[nid]
            pred_map[(data["scan_idx"], data["spot_idx"])] = data["track_id"]

        # Walk edges and check for switches.
        for u, v in G.edges:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            u_key = (u_data["scan_idx"], u_data["spot_idx"])
            v_key = (v_data["scan_idx"], v_data["spot_idx"])
            u_true = gt_map.get(u_key)
            v_true = gt_map.get(v_key)
            if u_true is not None and v_true is not None and u_true != v_true:
                id_switch_count += 1

        n_edges = G.number_of_edges()
        id_switch_rate = id_switch_count / n_edges if n_edges > 0 else 0.0

    return {
        "total_tracks": total_tracks,
        "full_length_tracks": full_length_tracks,
        "fragmentation_ratio": round(fragmentation_ratio, 4),
        "born_count": born_count,
        "continued_count": continued_count,
        "terminated_count": terminated_count,
        "id_switch_count": id_switch_count,
        "id_switch_rate": round(id_switch_rate, 4),
    }
