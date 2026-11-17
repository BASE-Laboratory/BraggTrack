"""Publication-grade 4D trajectory and split/merge lineage DAG visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from braggtrack.tracking.lifecycle import tracks_to_table

from ._style import OKABE_ITO, _require_matplotlib, apply_axis_style, maybe_save


def _parse_int_list(val: Any) -> list[int]:
    if val is None or val == "":
        return []
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    return [int(tok) for tok in str(val).split(";") if tok.strip()]


def plot_lineage_dag(
    tracks_or_graph: nx.DiGraph | list[dict[str, Any]],
    *,
    events: list[Any] | None = None,
    max_tracks: int = 40,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot a scan-layered 4D trajectory & lineage DAG highlighting splits, merges, and gap closures."""
    _, plt = _require_matplotlib()

    if isinstance(tracks_or_graph, nx.DiGraph):
        rows = tracks_to_table(tracks_or_graph)
        if events is None:
            events = list(tracks_or_graph.graph.get("lineage_events", []))
    else:
        rows = [dict(r) for r in tracks_or_graph]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if not rows:
        ax.set_title(title or "4D Trajectory Lineage DAG (empty)")
        apply_axis_style(ax)
        maybe_save(fig, save_path)
        return fig, ax

    by_track: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_track.setdefault(int(float(r["track_id"])), []).append(r)
    for tid in by_track:
        by_track[tid].sort(key=lambda r: int(float(r["scan_idx"])))

    # Prioritize tracks involved in split/merge events, then longest & brightest tracks.
    event_records: list[dict[str, Any]] = []
    involved_tids: set[int] = set()
    for ev in events or []:
        if isinstance(ev, dict):
            etype = str(ev.get("event_type", ""))
            pscan = int(float(ev.get("scan_idx_from", ev.get("parent_scan", 0))))
            cscan = int(float(ev.get("scan_idx_to", ev.get("child_scan", pscan + 1))))
            ptids = _parse_int_list(ev.get("parent_track_ids"))
            ctids = _parse_int_list(ev.get("child_track_ids"))
        else:
            etype = str(getattr(ev, "event_type", ""))
            pscan = int(getattr(ev, "scan_idx_from", getattr(ev, "parent_scan", 0)))
            cscan = int(getattr(ev, "scan_idx_to", getattr(ev, "child_scan", pscan + 1)))
            ptids = [int(x) for x in getattr(ev, "parent_track_ids", ())]
            ctids = [int(x) for x in getattr(ev, "child_track_ids", ())]
        event_records.append(
            {
                "event_type": etype,
                "parent_scan": pscan,
                "child_scan": cscan,
                "parent_track_ids": ptids,
                "child_track_ids": ctids,
            }
        )
        involved_tids.update(ptids)
        involved_tids.update(ctids)

    ranked_tids = sorted(
        by_track.keys(),
        key=lambda tid: (
            0 if tid in involved_tids else 1,
            -len(by_track[tid]),
            -max(float(r.get("integrated_intensity", 0.0)) for r in by_track[tid]),
            tid,
        ),
    )[:max_tracks]
    selected_set = set(ranked_tids)

    # Order vertical lanes by initial centroid_d so spatially nearby fragments sit adjacent.
    ordered_tids = sorted(ranked_tids, key=lambda tid: (float(by_track[tid][0]["centroid_d"]), tid))
    lane_y = {tid: float(idx) for idx, tid in enumerate(ordered_tids)}

    node_xy: dict[tuple[int, int], tuple[float, float]] = {}
    all_x: list[float] = []
    all_y: list[float] = []
    all_strain: list[float] = []

    gap_labeled = False
    cont_labeled = False
    for tid in ordered_tids:
        obs = by_track[tid]
        y_val = lane_y[tid]
        d0 = float(obs[0]["centroid_d"])
        scans = [int(float(r["scan_idx"])) for r in obs]
        for r in obs:
            s_idx = int(float(r["scan_idx"]))
            d_cur = float(r["centroid_d"])
            strain_pct = 100.0 * (d_cur - d0) / d0 if abs(d0) > 1e-9 else 0.0
            node_xy[(tid, s_idx)] = (float(s_idx), y_val)
            all_x.append(float(s_idx))
            all_y.append(y_val)
            all_strain.append(strain_pct)

        for i in range(len(scans) - 1):
            s0, s1 = scans[i], scans[i + 1]
            if s1 - s0 > 1:
                ax.plot(
                    [s0, s1],
                    [y_val, y_val],
                    linestyle="--",
                    linewidth=1.8,
                    color=OKABE_ITO[4],
                    label="Gap-closed bridge" if not gap_labeled else None,
                    zorder=2,
                )
                gap_labeled = True
            else:
                ax.plot(
                    [s0, s1],
                    [y_val, y_val],
                    linestyle="-",
                    linewidth=1.8,
                    color="#64748B",
                    alpha=0.75,
                    label="Continuation" if not cont_labeled else None,
                    zorder=2,
                )
                cont_labeled = True

    split_labeled = False
    merge_labeled = False
    for ev in event_records:
        etype = ev["event_type"]
        pscan = ev["parent_scan"]
        cscan = ev["child_scan"]
        for ptid in ev["parent_track_ids"]:
            for ctid in ev["child_track_ids"]:
                if ptid not in selected_set or ctid not in selected_set:
                    continue
                p_xy = node_xy.get((ptid, pscan), (float(pscan), lane_y[ptid]))
                c_xy = node_xy.get((ctid, cscan), (float(cscan), lane_y[ctid]))
                color = OKABE_ITO[1] if etype == "split" else OKABE_ITO[2]
                lbl = None
                if etype == "split" and not split_labeled:
                    lbl = "1 -> 2+ Split"
                    split_labeled = True
                elif etype == "merge" and not merge_labeled:
                    lbl = "2+ -> 1 Merge"
                    merge_labeled = True
                ax.plot(
                    [p_xy[0], c_xy[0]],
                    [p_xy[1], c_xy[1]],
                    color=color,
                    linewidth=2.4,
                    alpha=0.92,
                    label=lbl,
                    zorder=3,
                )
                ax.scatter(
                    [c_xy[0] if etype == "split" else p_xy[0]],
                    [c_xy[1] if etype == "split" else p_xy[1]],
                    marker="D" if etype == "split" else "s",
                    s=58,
                    color=color,
                    edgecolor="#FFFFFF",
                    linewidth=0.8,
                    zorder=5,
                )

    vmax = max(0.05, float(np.max(np.abs(all_strain)))) if all_strain else 0.1
    sc = ax.scatter(
        all_x,
        all_y,
        c=all_strain,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=42,
        edgecolors="#1E293B",
        linewidths=0.6,
        zorder=4,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"Relative $d$-shift $\Delta d / d_0$ (%)", fontsize=9)

    ax.set_yticks([lane_y[tid] for tid in ordered_tids])
    ax.set_yticklabels([f"T{tid}" for tid in ordered_tids], fontsize=8)
    max_scan = int(max(all_x)) if all_x else 0
    ax.set_xticks(list(range(max_scan + 1)))
    ax.set_xlabel("Scan index", fontsize=9.5)
    ax.set_ylabel("Grain trajectory lane", fontsize=9.5)
    ax.set_title(
        title or f"4D Bragg Reflection Lineage DAG ({len(ordered_tids)} tracks)", fontsize=11, fontweight="bold"
    )
    apply_axis_style(ax)
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    maybe_save(fig, save_path)
    return fig, ax
