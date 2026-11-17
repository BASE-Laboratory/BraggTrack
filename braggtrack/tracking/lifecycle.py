"""Track lifecycle management and trajectory DAG construction."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import networkx as nx

from .assignment import associate_frames
from .cost import CostFunction, PositionShapeCost

if TYPE_CHECKING:
    from .events import SplitMergeConfig
    from .gap_closing import GapClosingConfig


class TrackEvent(str, enum.Enum):
    BORN = "born"
    CONTINUED = "continued"
    TERMINATED = "terminated"


def _node_id(scan_idx: int, spot_idx: int) -> str:
    """Canonical node key: ``'s{scan_idx}_i{spot_idx}'``."""
    return f"s{scan_idx}_i{spot_idx}"


def build_tracks(
    scan_tables: list[list[dict]],
    cost_fn: CostFunction | None = None,
    max_cost: float = float("inf"),
    *,
    gap_closing_config: GapClosingConfig | None = None,
    split_merge_config: SplitMergeConfig | None = None,
    scan_timestamps: dict[int, str | None] | None = None,
) -> nx.DiGraph:
    """Build a trajectory DAG across an ordered sequence of scan tables.

    Each node represents a spot observation ``(scan_index, spot_index)``
    and carries the full feature dict plus a ``track_id`` and ``event``
    attribute.  Edges connect matched spots across consecutive frames,
    optional *gap_closing_config* reconnects missing intermediate scans, and
    optional *split_merge_config* resolves ``1 -> 2+`` splits and ``2+ -> 1``
    merges.
    """
    active_cost: CostFunction = cost_fn if cost_fn is not None else PositionShapeCost()
    G = nx.DiGraph()
    n_scans = len(scan_tables)
    if n_scans == 0:
        return G

    next_track_id = 1

    # Initialise first frame — every spot is BORN.
    track_map: dict[int, int] = {}  # spot_idx -> track_id for current frame
    for idx, spot in enumerate(scan_tables[0]):
        nid = _node_id(0, idx)
        G.add_node(nid, scan_idx=0, spot_idx=idx, track_id=next_track_id, event=TrackEvent.BORN, **spot)
        track_map[idx] = next_track_id
        next_track_id += 1

    # Process each transition.
    for t in range(n_scans - 1):
        spots_t = scan_tables[t]
        spots_t1 = scan_tables[t + 1]

        matches, unmatched_t, unmatched_t1 = associate_frames(
            spots_t,
            spots_t1,
            cost_fn=active_cost,
            max_cost=max_cost,
        )

        new_track_map: dict[int, int] = {}

        # Matched pairs → CONTINUED
        for i_t, i_t1 in matches:
            tid = track_map[i_t]
            nid_prev = _node_id(t, i_t)
            nid_next = _node_id(t + 1, i_t1)
            G.add_node(
                nid_next, scan_idx=t + 1, spot_idx=i_t1, track_id=tid, event=TrackEvent.CONTINUED, **spots_t1[i_t1]
            )
            G.add_edge(nid_prev, nid_next, edge_type="continuation")
            new_track_map[i_t1] = tid

        # Unmatched in t → TERMINATED (mark on existing node)
        for i_t in unmatched_t:
            nid = _node_id(t, i_t)
            G.nodes[nid]["event"] = TrackEvent.TERMINATED

        # Unmatched in t+1 → BORN (new track)
        for i_t1 in unmatched_t1:
            nid = _node_id(t + 1, i_t1)
            G.add_node(
                nid, scan_idx=t + 1, spot_idx=i_t1, track_id=next_track_id, event=TrackEvent.BORN, **spots_t1[i_t1]
            )
            new_track_map[i_t1] = next_track_id
            next_track_id += 1

        track_map = new_track_map

    # Mark survivors in last frame that haven't been explicitly terminated.
    last_scan = n_scans - 1
    for idx in range(len(scan_tables[last_scan])):
        nid = _node_id(last_scan, idx)
        if nid in G.nodes and G.nodes[nid]["event"] == TrackEvent.CONTINUED:
            # Still alive at end — leave as CONTINUED (not terminated)
            pass

    if gap_closing_config is not None and gap_closing_config.max_gap > 0:
        from .gap_closing import resolve_track_gaps

        resolve_track_gaps(
            G,
            n_scans=n_scans,
            cost_fn=active_cost,
            config=gap_closing_config,
        )

    if split_merge_config is not None:
        from .events import resolve_splits_and_merges

        resolve_splits_and_merges(
            G,
            n_scans=n_scans,
            config=split_merge_config,
            scan_timestamps=scan_timestamps,
        )

    return G


def tracks_to_table(G: nx.DiGraph) -> list[dict[str, Any]]:
    """Flatten the trajectory DAG into a row-per-observation table.

    Each row contains ``track_id``, ``scan_idx``, ``spot_idx``,
    ``event``, and all feature columns.
    """
    has_lineage = bool(G.graph.get("lineage_events"))
    rows: list[dict[str, Any]] = []
    for nid in sorted(G.nodes):
        data = dict(G.nodes[nid])
        data["event"] = data["event"].value if isinstance(data["event"], TrackEvent) else data["event"]
        if has_lineage or "parent_track_ids" in data or "lineage_role" in data:
            ptids = data.get("parent_track_ids", [])
            if isinstance(ptids, (list, tuple)):
                data["parent_track_ids"] = ";".join(str(x) for x in ptids)
            data.setdefault("lineage_role", "")
        rows.append(data)
    return rows
