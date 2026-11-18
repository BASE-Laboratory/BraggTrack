"""Synchronized operando electrochemistry / stimulus + lineage strain + fracture precursor figure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from braggtrack.tracking.kinematics import GrainKinematics, compute_grain_kinematics

from ._style import OKABE_ITO, _require_matplotlib, apply_axis_style, maybe_save

if TYPE_CHECKING:
    from braggtrack.io.operando_log import ScanStimulusPoint


def plot_operando_strain_and_events(
    kinematics_or_tracks: list[GrainKinematics] | list[dict[str, Any]],
    scan_stimuli: list[ScanStimulusPoint] | list[dict[str, Any]],
    *,
    events: list[Any] | None = None,
    precursor_observations: list[Any] | None = None,
    zoom_triggers: list[Any] | None = None,
    max_tracks: int = 10,
    calibrated: bool = False,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot 3-panel synchronized operando stimulus, ancestor-referenced strain ± error, and fracture precursors.

    Panels (shared scan-index horizontal axis):
    1. External stimulus (Cell Voltage ``V`` + optional twin-axis ``|dQ/dV|``, or Stress ``MPa`` / Temp ``°C``).
    2. Root-ancestor-referenced per-grain lattice strain ``lineage_strain ± lineage_strain_err`` (%) with
       dashed parent -> daughter cleavage connectors across ``1 -> 2+`` splits.
    3. Per-grain fracture precursor score ``[0, 1]`` with split lines and optional autonomous zoom trigger stars.
    """
    _, plt = _require_matplotlib()

    if kinematics_or_tracks and isinstance(kinematics_or_tracks[0], GrainKinematics):
        gks: list[GrainKinematics] = list(kinematics_or_tracks)  # type: ignore[arg-type]
    else:
        gks = compute_grain_kinematics([dict(r) for r in kinematics_or_tracks])  # type: ignore[arg-type]

    stim_dicts: list[dict[str, Any]] = [
        s.to_dict() if hasattr(s, "to_dict") else dict(s)  # type: ignore[union-attr, arg-type]
        for s in scan_stimuli
    ]
    stim_dicts.sort(key=lambda d: int(d.get("scan_idx", 0)))

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.45, 1.05]})

    # --- Panel 0: Operando Stimulus (Voltage + dQ/dV, or Stress / Temperature) ---
    ax_stim = axes[0]
    if stim_dicts:
        s_x = np.array([int(d["scan_idx"]) for d in stim_dicts], dtype=np.float64)
        volt = [d.get("voltage_v") for d in stim_dicts]
        dqdv = [d.get("dq_dv") for d in stim_dicts]
        stress = [d.get("stress_mpa") for d in stim_dicts]
        temp = [d.get("temperature_c") for d in stim_dicts]

        if any(v is not None and v != "" for v in volt):
            v_arr = np.array([float(v) if (v is not None and v != "") else np.nan for v in volt], dtype=np.float64)
            ax_stim.plot(s_x, v_arr, "o-", color="#0F172A", linewidth=2.0, markersize=4.5, label="Cell Voltage (V)")
            ax_stim.set_ylabel("Potential (V vs. Li/Li$^+$)", fontsize=9, fontweight="bold")
            if any(q is not None and q != "" for q in dqdv):
                q_arr = np.array(
                    [abs(float(q)) if (q is not None and q != "") else np.nan for q in dqdv],
                    dtype=np.float64,
                )
                if np.any(np.isfinite(q_arr)) and float(np.nanmax(q_arr)) > 0.0:
                    ax_twin = ax_stim.twinx()
                    ax_twin.fill_between(s_x, 0.0, q_arr, color=OKABE_ITO[5], alpha=0.22, label="|dQ/dV|")
                    ax_twin.plot(s_x, q_arr, "--", color=OKABE_ITO[0], linewidth=1.2, alpha=0.85)
                    ax_twin.set_ylabel("|dQ/dV|", fontsize=8.5, color=OKABE_ITO[0])
                    ax_twin.spines["top"].set_visible(False)
        elif any(s is not None and s != "" for s in stress):
            st_arr = np.array([float(s) if (s is not None and s != "") else np.nan for s in stress], dtype=np.float64)
            ax_stim.plot(s_x, st_arr, "s-", color=OKABE_ITO[1], linewidth=2.0, markersize=4.5, label="Stress (MPa)")
            ax_stim.set_ylabel("Applied Stress (MPa)", fontsize=9, fontweight="bold")
        elif any(t is not None and t != "" for t in temp):
            t_arr = np.array([float(t) if (t is not None and t != "") else np.nan for t in temp], dtype=np.float64)
            ax_stim.plot(s_x, t_arr, "^-", color=OKABE_ITO[3], linewidth=2.0, markersize=4.5, label="Temperature (°C)")
            ax_stim.set_ylabel("Temperature (°C)", fontsize=9, fontweight="bold")

    ax_stim.set_title("Synchronized Operando Stimulus", fontsize=10, fontweight="bold")
    apply_axis_style(ax_stim)

    # --- Panel 1: Ancestor-Referenced Per-Grain Strain ± 1-sigma Uncertainty ---
    ax_eps = axes[1]
    gk_by_tid = {g.track_id: g for g in gks}

    split_involved: set[int] = set()
    parsed_events: list[tuple[str, int, int, list[int], list[int]]] = []
    for ev in events or []:
        if isinstance(ev, dict):
            etype = str(ev.get("event_type", ""))
            p_scan = int(float(ev.get("scan_idx_from", ev.get("parent_scan", 0))))
            c_scan = int(float(ev.get("scan_idx_to", ev.get("child_scan", p_scan + 1))))
            raw_p = ev.get("parent_track_ids", [])
            raw_c = ev.get("child_track_ids", [])
            ptids = (
                [int(x) for x in str(raw_p).split(";") if x.strip()]
                if isinstance(raw_p, str)
                else [int(x) for x in raw_p]
            )
            ctids = (
                [int(x) for x in str(raw_c).split(";") if x.strip()]
                if isinstance(raw_c, str)
                else [int(x) for x in raw_c]
            )
        else:
            etype = str(getattr(ev, "event_type", ""))
            p_scan = int(getattr(ev, "scan_idx_from", 0))
            c_scan = int(getattr(ev, "scan_idx_to", p_scan + 1))
            ptids = [int(x) for x in getattr(ev, "parent_track_ids", ())]
            ctids = [int(x) for x in getattr(ev, "child_track_ids", ())]
        parsed_events.append((etype, p_scan, c_scan, ptids, ctids))
        split_involved.update(ptids)
        split_involved.update(ctids)

    ranked = sorted(
        gks,
        key=lambda g: (
            0 if g.track_id in split_involved else 1,
            -len(g.scan_indices),
            -float(g.integrated_intensity[0]) if len(g.integrated_intensity) else 0.0,
            g.track_id,
        ),
    )[:max_tracks]

    color_by_tid: dict[int, str] = {}
    for idx, gk in enumerate(ranked):
        col = OKABE_ITO[idx % len(OKABE_ITO)]
        color_by_tid[gk.track_id] = col
        xs = np.asarray(gk.scan_indices, dtype=np.float64)
        ys = 100.0 * (gk.lineage_strain if len(gk.lineage_strain) else gk.strain)
        yerr = 100.0 * (gk.lineage_strain_err if len(gk.lineage_strain_err) else gk.strain_err)
        lbl = f"T{gk.track_id}" + (f" (root T{gk.root_track_id})" if gk.root_track_id not in (0, gk.track_id) else "")
        ax_eps.plot(xs, ys, "o-", color=col, linewidth=1.8, markersize=4.5, label=lbl)
        if len(yerr) == len(ys) and np.any(yerr > 0.0):
            ax_eps.fill_between(xs, ys - yerr, ys + yerr, color=col, alpha=0.16)

    # Draw parent -> child split connectors in lineage strain space
    for etype, _p_scan, c_scan, ptids, ctids in parsed_events:
        ev_col = OKABE_ITO[1] if etype == "split" else OKABE_ITO[2]
        for ax in axes:
            ax.axvline(c_scan - 0.5, linestyle=":", linewidth=1.2, color=ev_col, alpha=0.75)
        if etype == "split":
            for ptid in ptids:
                pgk = gk_by_tid.get(ptid)
                if pgk is None or not pgk.scan_indices:
                    continue
                py = 100.0 * float((pgk.lineage_strain if len(pgk.lineage_strain) else pgk.strain)[-1])
                px = float(pgk.scan_indices[-1])
                for ctid in ctids:
                    cgk = gk_by_tid.get(ctid)
                    if cgk is None or not cgk.scan_indices:
                        continue
                    cy = 100.0 * float((cgk.lineage_strain if len(cgk.lineage_strain) else cgk.strain)[0])
                    cx = float(cgk.scan_indices[0])
                    ax_eps.plot([px, cx], [py, cy], "--", color=OKABE_ITO[1], linewidth=1.3, alpha=0.85)

    ax_eps.axhline(0.0, color="#94A3B8", linewidth=0.8)
    ax_eps.set_title("Ancestor-Referenced Per-Grain Strain (±1σ Uncertainty)", fontsize=10, fontweight="bold")
    ax_eps.set_ylabel(
        r"Lineage strain $\varepsilon_{\mathrm{root}} = (d - d_0^{\mathrm{root}}) / d_0^{\mathrm{root}}$ (%)"
        if calibrated
        else r"Lineage relative $d$-shift (%)",
        fontsize=8.8,
    )
    if len(ranked) <= 10 and ranked:
        ax_eps.legend(fontsize=7.5, ncol=2, frameon=True, loc="best")
    apply_axis_style(ax_eps)

    # --- Panel 2: Fracture Precursor Scores & Autonomous DFXM Zoom Triggers ---
    ax_prec = axes[2]
    prec_by_tid: dict[int, list[tuple[int, float]]] = {}
    for po in precursor_observations or []:
        if isinstance(po, dict):
            tid = int(float(po["track_id"]))
            s_i = int(float(po["scan_idx"]))
            sc = float(po.get("precursor_score", 0.0))
        else:
            tid = int(getattr(po, "track_id", 0))
            s_i = int(getattr(po, "scan_idx", 0))
            sc = float(getattr(po, "precursor_score", 0.0))
        prec_by_tid.setdefault(tid, []).append((s_i, sc))

    shown_tids = {gk.track_id for gk in ranked}
    for tid, pts in sorted(prec_by_tid.items()):
        if shown_tids and tid not in shown_tids:
            continue
        pts_sorted = sorted(pts)
        col = color_by_tid.get(tid, OKABE_ITO[tid % len(OKABE_ITO)])
        ax_prec.plot(
            [p[0] for p in pts_sorted],
            [p[1] for p in pts_sorted],
            "o-",
            color=col,
            linewidth=1.5,
            markersize=3.8,
            alpha=0.9,
        )

    for zt in zoom_triggers or []:
        z_scan = int(zt["scan_idx"] if isinstance(zt, dict) else getattr(zt, "scan_idx", 0))
        z_tid = int(zt["track_id"] if isinstance(zt, dict) else getattr(zt, "track_id", 0))
        z_score = float(
            zt.get("precursor_score", zt.get("priority_score", 0.75))
            if isinstance(zt, dict)
            else getattr(zt, "precursor_score", getattr(zt, "priority_score", 0.75))
        )
        ax_prec.scatter(
            [z_scan],
            [z_score],
            marker="*",
            s=155,
            color="#F59E0B",
            edgecolors="#0F172A",
            linewidths=0.9,
            zorder=6,
            label=f"Zoom Trigger T{z_tid}@S{z_scan}",
        )

    ax_prec.axhline(0.40, linestyle="--", linewidth=0.9, color="#EC4899", alpha=0.7)
    ax_prec.set_ylim(-0.03, 1.05)
    ax_prec.set_title("Persistent Fracture Precursor Score & Autonomous Zoom Triggers", fontsize=10, fontweight="bold")
    ax_prec.set_ylabel("Precursor Score [0, 1]", fontsize=8.8)
    ax_prec.set_xlabel("Scan index", fontsize=9.2, fontweight="bold")
    apply_axis_style(ax_prec)

    fig.suptitle(
        title or "Synchronized Operando Stimulus, Lineage Strain & Chemomechanical Precursors",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    maybe_save(fig, save_path)
    return fig, axes
