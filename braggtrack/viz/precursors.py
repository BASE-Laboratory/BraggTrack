"""Persistent 4D fracture precursor heatmap and component decomposition plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.tracking.precursors import ObservationPrecursorScore, PrecursorCandidate

from ._style import OKABE_ITO, _require_matplotlib, apply_axis_style, maybe_save


def _obs_to_dict(obs: ObservationPrecursorScore | dict[str, Any]) -> dict[str, Any]:
    if isinstance(obs, dict):
        return {
            "track_id": int(float(obs["track_id"])),
            "scan_idx": int(float(obs["scan_idx"])),
            "anisotropy_term": float(obs.get("anisotropy_term", 0.0)),
            "spread_term": float(obs.get("spread_term", 0.0)),
            "roi_dispersion_term": float(obs.get("roi_dispersion_term", 0.0)),
            "roi_subpeak_count": int(float(obs.get("roi_subpeak_count", 1))),
            "precursor_score": float(obs.get("precursor_score", 0.0)),
        }
    return {
        "track_id": obs.track_id,
        "scan_idx": obs.scan_idx,
        "anisotropy_term": obs.anisotropy_term,
        "spread_term": obs.spread_term,
        "roi_dispersion_term": obs.roi_dispersion_term,
        "roi_subpeak_count": obs.roi_subpeak_count,
        "precursor_score": obs.precursor_score,
    }


def plot_precursor_timeline(
    obs_scores: list[ObservationPrecursorScore] | list[dict[str, Any]],
    *,
    candidates: list[PrecursorCandidate] | list[dict[str, Any]] | None = None,
    max_tracks: int = 15,
    score_threshold: float = 0.40,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot a track-by-scan fracture precursor heatmap and top-candidate component decomposition."""
    _, plt = _require_matplotlib()
    rows = [_obs_to_dict(x) for x in obs_scores]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    if not rows:
        for ax in axes:
            apply_axis_style(ax)
        fig.suptitle(title or "Fracture Precursor Timeline (no observations)")
        maybe_save(fig, save_path)
        return fig, axes

    by_track: dict[int, list[dict[str, Any]]] = {}
    max_scan = 0
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r)
        if r["scan_idx"] > max_scan:
            max_scan = r["scan_idx"]

    split_tids: set[int] = set()
    top_tid: int | None = None
    for idx_c, c in enumerate(candidates or []):
        tid_c = int(float(c["track_id"])) if isinstance(c, dict) else int(c.track_id)
        sub_split = (
            str(c.get("subsequent_split", "")).lower() == "true" if isinstance(c, dict) else bool(c.subsequent_split)
        )
        if idx_c == 0:
            top_tid = tid_c
        if sub_split:
            split_tids.add(tid_c)

    ranked_tids = sorted(
        by_track.keys(),
        key=lambda tid: (-max(x["precursor_score"] for x in by_track[tid]), tid),
    )[:max_tracks]
    if top_tid is None and ranked_tids:
        top_tid = ranked_tids[0]

    mat = np.full((len(ranked_tids), max_scan + 1), np.nan, dtype=np.float64)
    for r_i, tid in enumerate(ranked_tids):
        for r in by_track[tid]:
            mat[r_i, r["scan_idx"]] = r["precursor_score"]

    im = axes[0].imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[0].set_yticks(list(range(len(ranked_tids))))
    axes[0].set_yticklabels(
        [f"T{tid} *" if tid in split_tids else f"T{tid}" for tid in ranked_tids],
        fontsize=8.5,
    )
    axes[0].set_xticks(list(range(max_scan + 1)))
    axes[0].set_xlabel("Scan index", fontsize=9)
    axes[0].set_ylabel("Track ID (* = subsequent split)", fontsize=9)
    axes[0].set_title("Precursor Score Heatmap", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Precursor score")

    # Right panel: component breakdown for top candidate
    if top_tid is not None and top_tid in by_track:
        top_obs = sorted(by_track[top_tid], key=lambda x: x["scan_idx"])
        scans = [x["scan_idx"] for x in top_obs]
        axes[1].plot(
            scans,
            [x["anisotropy_term"] for x in top_obs],
            "o--",
            color=OKABE_ITO[0],
            linewidth=1.4,
            label="Shape anisotropy",
        )
        axes[1].plot(
            scans,
            [x["spread_term"] for x in top_obs],
            "s--",
            color=OKABE_ITO[2],
            linewidth=1.4,
            label="Covariance broadening",
        )
        axes[1].plot(
            scans,
            [x["roi_dispersion_term"] for x in top_obs],
            "^--",
            color=OKABE_ITO[3],
            linewidth=1.4,
            label="3D sub-peaks + DINO dispersion",
        )
        axes[1].plot(
            scans,
            [x["precursor_score"] for x in top_obs],
            "o-",
            color=OKABE_ITO[1],
            linewidth=2.2,
            label=f"Composite score (T{top_tid})",
        )
        axes[1].axhline(score_threshold, linestyle=":", color="#475569", linewidth=1.2, label="Threshold")
        axes[1].set_ylim(-0.04, 1.06)
        axes[1].set_xticks(list(range(max_scan + 1)))
        axes[1].set_xlabel("Scan index", fontsize=9)
        axes[1].set_ylabel("Normalized term [0, 1]", fontsize=9)
        axes[1].set_title(f"Top Precursor Decomposition — Track T{top_tid}", fontsize=10, fontweight="bold")
        axes[1].legend(fontsize=7.8, loc="best", frameon=True)
    apply_axis_style(axes[1])

    fig.suptitle(title or "Persistent 4D Fracture Precursor Analysis", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    maybe_save(fig, save_path)
    return fig, axes
