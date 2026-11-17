"""Orthogonal 3-view MIP spot cards and full-scan projection overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips

from ._style import _require_matplotlib, apply_axis_style, maybe_save


def plot_spot_mip_card(
    volume: np.ndarray | Any,
    labels: np.ndarray,
    label_id: int,
    bbox: dict[str, Any],
    *,
    margin: int = 3,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot a 3-panel orthogonal MIP card ``(d, chi)``, ``(mu, d)``, ``(mu, chi)`` for one 3D reflection."""
    _, plt = _require_matplotlib()
    masked_cube, mask = crop_spot_cube(volume, labels, int(label_id), bbox, margin=margin)
    mip_mu, mip_chi, mip_d = orthogonal_mips(masked_cube)
    mask_mu, mask_chi, mask_d = orthogonal_mips(mask)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    panels = [
        (mip_mu, mask_mu, r"$\mu$-collapsed $(d, \chi)$", r"$\chi$ (px)", r"$d$ (px)"),
        (mip_chi, mask_chi, r"$\chi$-collapsed $(\mu, d)$", r"$d$ (px)", r"$\mu$ (frame)"),
        (mip_d, mask_d, r"$d$-collapsed $(\mu, \chi)$", r"$\chi$ (px)", r"$\mu$ (frame)"),
    ]

    for ax, (mip, msk, subtitle, xlabel, ylabel) in zip(axes, panels):
        im = ax.imshow(mip, cmap="magma", origin="lower", aspect="auto")
        if msk.shape[0] >= 2 and msk.shape[1] >= 2 and np.any(msk > 0) and np.any(msk == 0):
            ax.contour(msk, levels=[0.5], colors=["#56B4E9"], linewidths=1.1)
        ax.set_title(subtitle, fontsize=9.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    vox = int(np.sum(mask > 0))
    tot_int = float(np.sum(masked_cube))
    header = title or f"Spot #{label_id}  |  {vox} voxels  |  I = {tot_int:.1f}"
    fig.suptitle(header, fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    maybe_save(fig, save_path)
    return fig, axes


def plot_scan_projection_overlay(
    volume: np.ndarray | Any,
    labels: np.ndarray,
    feature_rows: list[dict[str, Any]],
    *,
    max_labels: int = 25,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot a full-scan ``(d, chi)`` MIP with instance contours and ranked spot IDs."""
    _, plt = _require_matplotlib()
    vol_arr = volume[()] if not isinstance(volume, np.ndarray) else volume
    mip_dc = np.max(np.asarray(vol_arr, dtype=np.float64), axis=0)
    disp = np.log1p(np.maximum(0.0, mip_dc))
    fg_proj = np.max(labels > 0, axis=0).astype(np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(disp, cmap="inferno", origin="lower", aspect="auto")
    if fg_proj.shape[0] >= 2 and fg_proj.shape[1] >= 2 and np.any(fg_proj > 0) and np.any(fg_proj == 0):
        ax.contour(fg_proj, levels=[0.5], colors=["#56B4E9"], linewidths=0.8, alpha=0.8)

    ranked = sorted(feature_rows, key=lambda r: float(r.get("integrated_intensity", 0.0)), reverse=True)[:max_labels]
    for row in ranked:
        if "bbox_min_x" in row and "bbox_max_x" in row:
            cx = 0.5 * (float(row["bbox_min_x"]) + float(row["bbox_max_x"]))
            cy = 0.5 * (float(row["bbox_min_y"]) + float(row["bbox_max_y"]))
        else:
            cx = float(row["centroid_chi"])
            cy = float(row["centroid_d"])
        lid = int(float(row["label"]))
        ax.plot(cx, cy, marker="+", color="#00E5FF", markersize=6, markeredgewidth=1.2)
        ax.annotate(
            f"#{lid}",
            (cx, cy),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#FFFFFF",
            bbox={"boxstyle": "round,pad=0.15", "fc": "#111827", "ec": "none", "alpha": 0.65},
        )

    ax.set_title(title or f"Detector Projection (log1p MIP) — {len(feature_rows)} segmented reflections", fontsize=10.5)
    ax.set_xlabel(r"Detector $\chi$ / fast pixel", fontsize=9)
    ax.set_ylabel(r"Detector $d$ / slow pixel", fontsize=9)
    apply_axis_style(ax)
    fig.colorbar(im, ax=ax, label="log(1 + max intensity)")
    fig.tight_layout()
    maybe_save(fig, save_path)
    return fig, ax
