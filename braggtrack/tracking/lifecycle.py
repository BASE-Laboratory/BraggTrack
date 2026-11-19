"""Track lifecycle management and trajectory DAG construction."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from .assignment import associate_frames, compute_assignment_margins
from .cost import CostFunction, PositionShapeCost
from .drift import (
    DriftMode,
    TrackKalmanState,
    apply_predicted_shifts,
    estimate_band_shifts_consensus,
    estimate_slab_shifts_fft,
)

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
    drift_mode: DriftMode = "none",
    volumes: Sequence[np.ndarray] | dict[int, np.ndarray] | None = None,
    precomputed_pair_shifts: dict[int, list[tuple[float, float, float]]] | None = None,
    scan_elapsed_s: dict[int, float] | None = None,
) -> nx.DiGraph:
    """Build a trajectory DAG across an ordered sequence of scan tables.

    Each node represents a spot observation ``(scan_index, spot_index)``
    and carries the full feature dict plus a ``track_id`` and ``event``
    attribute.  Edges connect matched spots across consecutive frames,
    optional *drift_mode* compensates collective rocking-band translations
    and per-track constant-velocity Kalman motion, optional *gap_closing_config*
    reconnects missing intermediate scans, and optional *split_merge_config*
    resolves ``1 -> 2+`` splits and ``2+ -> 1`` merges.
    """
    active_cost: CostFunction = cost_fn if cost_fn is not None else PositionShapeCost()
    G = nx.DiGraph()
    n_scans = len(scan_tables)
    if n_scans == 0:
        return G

    vol_map: dict[int, np.ndarray] = {}
    if isinstance(volumes, dict):
        vol_map = volumes
    elif volumes is not None:
        vol_map = dict(enumerate(volumes))

    next_track_id = 1
    kalman_by_tid: dict[int, TrackKalmanState] = {}
    node_shifts: dict[str, tuple[float, float, float]] = {}

    def _scan_time(s_idx: int) -> float:
        if scan_elapsed_s and s_idx in scan_elapsed_s:
            return float(scan_elapsed_s[s_idx])
        return float(s_idx)

    # Initialise first frame — every spot is BORN.
    track_map: dict[int, int] = {}  # spot_idx -> track_id for current frame
    t0_val = _scan_time(0)
    for idx, spot in enumerate(scan_tables[0]):
        nid = _node_id(0, idx)
        G.add_node(
            nid,
            scan_idx=0,
            spot_idx=idx,
            track_id=next_track_id,
            event=TrackEvent.BORN,
            assignment_margin=float("inf"),
            **spot,
        )
        track_map[idx] = next_track_id
        if drift_mode == "fft_kalman":
            kalman_by_tid[next_track_id] = TrackKalmanState.initialize(spot, time_val=t0_val)
        next_track_id += 1

    # Process each transition.
    for t in range(n_scans - 1):
        spots_t = scan_tables[t]
        spots_t1 = scan_tables[t + 1]
        t_cur = _scan_time(t)
        t_next = _scan_time(t + 1)
        dt_step = max(t_next - t_cur, 1e-6)

        shifts_t: list[tuple[float, float, float]] | None = None
        if drift_mode != "none" and spots_t:
            if precomputed_pair_shifts is not None and t in precomputed_pair_shifts:
                raw_s = precomputed_pair_shifts[t]
                if len(raw_s) == len(spots_t):
                    shifts_t = [(float(a), float(b), float(c)) for a, b, c in raw_s]
            if shifts_t is None:
                if drift_mode in ("fft", "fft_kalman") and t in vol_map and (t + 1) in vol_map:
                    shifts_t, _ = estimate_slab_shifts_fft(vol_map[t], vol_map[t + 1], spots_t)
                else:
                    shifts_t = estimate_band_shifts_consensus(spots_t, spots_t1)

            if drift_mode == "fft_kalman":
                blended: list[tuple[float, float, float]] = []
                for idx_s, sp in enumerate(spots_t):
                    tid = track_map[idx_s]
                    kf = kalman_by_tid.get(tid)
                    slab_s = shifts_t[idx_s]
                    if kf is not None and kf.age >= 2:
                        pred_pos, _ = kf.predict(dt_step)
                        kf_s = (
                            pred_pos[0] - float(sp["centroid_mu"]),
                            pred_pos[1] - float(sp["centroid_chi"]),
                            pred_pos[2] - float(sp["centroid_d"]),
                        )
                        has_slab = abs(slab_s[0]) + abs(slab_s[1]) + abs(slab_s[2]) > 1e-6
                        if has_slab:
                            blended.append(
                                (
                                    0.5 * slab_s[0] + 0.5 * kf_s[0],
                                    0.5 * slab_s[1] + 0.5 * kf_s[1],
                                    0.5 * slab_s[2] + 0.5 * kf_s[2],
                                )
                            )
                        else:
                            blended.append(kf_s)
                    else:
                        blended.append(slab_s)
                shifts_t = blended

            for idx_s, sh in enumerate(shifts_t):
                node_shifts[_node_id(t, idx_s)] = sh

        assoc_spots_t = apply_predicted_shifts(spots_t, shifts_t) if shifts_t is not None else spots_t

        matches, unmatched_t, unmatched_t1 = associate_frames(
            assoc_spots_t,
            spots_t1,
            cost_fn=active_cost,
            max_cost=max_cost,
        )
        margins = compute_assignment_margins(assoc_spots_t, spots_t1, matches, active_cost)

        new_track_map: dict[int, int] = {}

        # Matched pairs → CONTINUED
        for i_t, i_t1 in matches:
            tid = track_map[i_t]
            nid_prev = _node_id(t, i_t)
            nid_next = _node_id(t + 1, i_t1)
            c_ij, margin = margins.get((i_t, i_t1), (0.0, float("inf")))
            G.add_node(
                nid_next,
                scan_idx=t + 1,
                spot_idx=i_t1,
                track_id=tid,
                event=TrackEvent.CONTINUED,
                assignment_margin=margin,
                **spots_t1[i_t1],
            )
            edge_attrs: dict[str, Any] = {
                "edge_type": "continuation",
                "cost": c_ij,
                "assignment_margin": margin,
            }
            if shifts_t is not None:
                edge_attrs["drift_shift"] = shifts_t[i_t]
            G.add_edge(nid_prev, nid_next, **edge_attrs)
            new_track_map[i_t1] = tid
            if drift_mode == "fft_kalman":
                if tid in kalman_by_tid:
                    kalman_by_tid[tid].update(spots_t1[i_t1], time_val=t_next)
                else:
                    kalman_by_tid[tid] = TrackKalmanState.initialize(spots_t1[i_t1], time_val=t_next)

        # Unmatched in t → TERMINATED (mark on existing node)
        for i_t in unmatched_t:
            nid = _node_id(t, i_t)
            G.nodes[nid]["event"] = TrackEvent.TERMINATED

        # Unmatched in t+1 → BORN (new track)
        for i_t1 in unmatched_t1:
            nid = _node_id(t + 1, i_t1)
            G.add_node(
                nid,
                scan_idx=t + 1,
                spot_idx=i_t1,
                track_id=next_track_id,
                event=TrackEvent.BORN,
                assignment_margin=float("inf"),
                **spots_t1[i_t1],
            )
            new_track_map[i_t1] = next_track_id
            if drift_mode == "fft_kalman":
                kalman_by_tid[next_track_id] = TrackKalmanState.initialize(spots_t1[i_t1], time_val=t_next)
            next_track_id += 1

        track_map = new_track_map

    # Mark survivors in last frame that haven't been explicitly terminated.
    last_scan = n_scans - 1
    for idx in range(len(scan_tables[last_scan])):
        nid = _node_id(last_scan, idx)
        if nid in G.nodes and G.nodes[nid]["event"] == TrackEvent.CONTINUED:
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
            predicted_shifts=node_shifts if node_shifts else None,
        )

    G.graph["drift_mode"] = drift_mode
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
