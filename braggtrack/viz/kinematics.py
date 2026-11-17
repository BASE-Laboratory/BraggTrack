"""Multi-panel per-grain kinematics dashboard (strain, misorientation, and intensity/anisotropy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from braggtrack.tracking.kinematics import GrainKinematics, compute_grain_kinematics

from ._style import OKABE_ITO, _require_matplotlib, apply_axis_style, maybe_save


def plot_kinematics_dashboard(
    kinematics_or_tracks: list[GrainKinematics] | list[dict[str, Any]],
    *,
    events: list[Any] | None = None,
    max_tracks: int = 12,
    calibrated: bool = False,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot 3-panel kinematic evolution: lattice strain, angular misorientation, and intensity/anisotropy."""
    _, plt = _require_matplotlib()

    if kinematics_or_tracks and isinstance(kinematics_or_tracks[0], GrainKinematics):
        gks: list[GrainKinematics] = list(kinematics_or_tracks)  # type: ignore[arg-type]
    else:
        rows = [dict(r) for r in kinematics_or_tracks]  # type: ignore[arg-type]
        if rows and "strain" in rows[0] and "misorientation_total" in rows[0] and "relative_intensity" in rows[0]:
            # Reconstruct GrainKinematics from kinematics.csv rows or compute from tracks.csv rows
            gks = compute_grain_kinematics(rows)
        else:
            gks = compute_grain_kinematics(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    if not gks:
        for ax in axes:
            apply_axis_style(ax)
        fig.suptitle(title or "Per-Grain Kinematics Dashboard (empty)")
        maybe_save(fig, save_path)
        return fig, axes

    ranked = sorted(gks, key=lambda g: (-len(g.scan_indices), -float(g.integrated_intensity[0]), g.track_id))[
        :max_tracks
    ]

    for idx, gk in enumerate(ranked):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        lbl = f"T{gk.track_id}"
        scans = gk.scan_indices
        axes[0].plot(scans, 100.0 * gk.strain, "o-", color=color, linewidth=1.7, markersize=4.5, label=lbl)
        axes[1].plot(scans, gk.misorientation_total, "o-", color=color, linewidth=1.7, markersize=4.5, label=lbl)
        axes[2].plot(scans, gk.anisotropy, "s-", color=color, linewidth=1.6, markersize=4.0, label=lbl)

    for ev in events or []:
        etype = ev.get("event_type") if isinstance(ev, dict) else getattr(ev, "event_type", "")
        raw_scan = (
            ev.get("scan_idx_to", ev.get("child_scan", 0))
            if isinstance(ev, dict)
            else getattr(ev, "scan_idx_to", getattr(ev, "child_scan", 0))
        )
        cscan = int(float(raw_scan))
        ev_col = OKABE_ITO[1] if etype == "split" else OKABE_ITO[2]
        for ax in axes:
            ax.axvline(cscan - 0.5, linestyle=":", linewidth=1.2, color=ev_col, alpha=0.8)

    strain_label = (
        r"Lattice strain $\varepsilon = \Delta d / d_0$ (%)"
        if calibrated
        else r"Relative $d$-shift $\Delta d / d_0$ (%)"
    )
    miso_label = (
        r"Misorientation $\sqrt{\Delta\mu^2 + \Delta\chi^2}$ (deg)"
        if calibrated
        else r"Angular shift $\sqrt{\Delta\mu^2 + \Delta\chi^2}$ (px)"
    )

    axes[0].axhline(0.0, color="#94A3B8", linewidth=0.8)
    axes[0].set_title("Strain / d-Spacing Evolution", fontsize=10, fontweight="bold")
    axes[0].set_ylabel(strain_label, fontsize=8.8)

    axes[1].set_title("Lattice Rotation / Misorientation", fontsize=10, fontweight="bold")
    axes[1].set_ylabel(miso_label, fontsize=8.8)

    axes[2].set_title(r"3D Shape Anisotropy ($\lambda_1 / \lambda_3$)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel(r"Eigenvalue ratio $\lambda_1 / \lambda_3$", fontsize=8.8)

    for ax in axes:
        ax.set_xlabel("Scan index", fontsize=9)
        apply_axis_style(ax)

    if len(ranked) <= 10:
        axes[0].legend(fontsize=7.5, ncol=2, frameon=True)

    fig.suptitle(
        title or f"Operando Grain Kinematics ({'Calibrated Physical Units' if calibrated else 'Detector Units'})",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    maybe_save(fig, save_path)
    return fig, axes
