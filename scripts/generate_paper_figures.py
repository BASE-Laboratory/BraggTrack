#!/usr/bin/env python3
"""Generate publication-grade composite figures for paper.md and README.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from braggtrack.io import AxisCalibration, discover_operando_scans, load_primary_volume, sample_operando_root  # noqa: E402
from braggtrack.segmentation import (  # noqa: E402
    extract_instance_table,
    fill_holes_binary,
    merge_nearby_labels,
    otsu_threshold,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
    smooth_thresholds,
)
from braggtrack.semantic import make_patch_encoder  # noqa: E402
from braggtrack.tracking import (  # noqa: E402
    PositionShapeCost,
    PrecursorConfig,
    SplitMergeConfig,
    build_tracks,
    compute_fracture_precursors,
    compute_grain_kinematics,
    events_to_table,
    generate_fracture_scenario,
    tracks_to_table,
)
from braggtrack.viz._style import OKABE_ITO, apply_axis_style  # noqa: E402


def build_fracture_and_operando_figure(out_path: Path, *, backend: str = "auto") -> None:
    # 1. Synthetic 4D chemomechanical fracture + merge + persistent precursor decomposition
    vols, labs, tables, _ = generate_fracture_scenario(n_scans=5, seed=42)
    G_frac = build_tracks(
        tables,
        cost_fn=PositionShapeCost(),
        split_merge_config=SplitMergeConfig(max_distance=20.0),
    )
    events = G_frac.graph["lineage_events"]
    ev_rows = events_to_table(events)
    frac_rows = tracks_to_table(G_frac)
    cands, obs_scores = compute_fracture_precursors(
        frac_rows,
        PrecursorConfig(score_threshold=0.35, min_persistence=2),
        volumes=vols,
        label_volumes=labs,
        patch_encoder=make_patch_encoder(backend),
        lineage_events=events,
    )

    # 2. Real ESRF-ID03 operando cathode sequence with physical calibration
    cal = AxisCalibration(
        mu_origin=12.0,
        mu_step=0.02,
        chi_origin=-2.5,
        chi_step=0.015,
        d_origin=2.065,
        d_step=-0.00015,
        calibrated=True,
    )
    scans = discover_operando_scans(sample_operando_root())
    op_vols = [load_primary_volume(s.path) for s in scans]
    sm_thr = smooth_thresholds([otsu_threshold(v.ravel()) for v in op_vols], window=5)
    op_tables = []
    for v, thr in zip(op_vols, sm_thr):
        res = segment_classical(
            v,
            threshold=float(thr),
            blur_passes=1,
            h_value=0.1,
            min_seed_separation=2,
            seed_peak_fraction=0.2,
            seed_response_percentile=99.95,
        )
        lb = remove_small_objects(res.labeled_volume, min_size=8)
        lb = np.where(fill_holes_binary(lb > 0), lb, 0)
        lb = relabel_sequential(merge_nearby_labels(lb, v, max_centroid_distance=15))
        op_tables.append(extract_instance_table(lb, v, calibration=cal))

    G_op = build_tracks(op_tables, cost_fn=PositionShapeCost(gate_mu=2.0, gate_chi=2.0, gate_d=0.05))
    gks = compute_grain_kinematics(tracks_to_table(G_op))

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    ax_dag, ax_mip = axes[0]
    ax_prec, ax_op = axes[1]

    # Panel A: 4D Lineage DAG (Split 1->2 & Merge 2->1)
    by_track: dict[int, list[dict]] = {}
    node_pos: dict[tuple[int, int], tuple[float, float]] = {}
    for r in frac_rows:
        tid = int(r["track_id"])
        sidx = int(r["scan_idx"])
        by_track.setdefault(tid, []).append(r)
        node_pos[(tid, sidx)] = (float(sidx), float(r["centroid_d"]))

    for idx_t, (tid, obs) in enumerate(sorted(by_track.items())):
        obs.sort(key=lambda x: int(x["scan_idx"]))
        col = OKABE_ITO[idx_t % len(OKABE_ITO)]
        xs = [int(x["scan_idx"]) for x in obs]
        ys = [float(x["centroid_d"]) for x in obs]
        ax_dag.plot(xs, ys, "o-", color=col, linewidth=2.0, markersize=6.5, label=f"T{tid}")

    for ev in ev_rows:
        etype = ev["event_type"]
        s_from, s_to = int(ev["scan_idx_from"]), int(ev["scan_idx_to"])
        pids = [int(p) for p in str(ev["parent_track_ids"]).split(";") if p]
        cids = [int(c) for c in str(ev["child_track_ids"]).split(";") if c]
        ecol = "#DC2626" if etype == "split" else "#059669"
        ls = "--" if etype == "split" else "-."
        for pid in pids:
            for cid in cids:
                if (pid, s_from) in node_pos and (cid, s_to) in node_pos:
                    x0, y0 = node_pos[(pid, s_from)]
                    x1, y1 = node_pos[(cid, s_to)]
                    ax_dag.plot([x0, x1], [y0, y1], linestyle=ls, color=ecol, linewidth=2.1)
                    ax_dag.annotate(
                        etype.upper(),
                        xy=((x0 + x1) / 2, (y0 + y1) / 2),
                        fontsize=7.5,
                        fontweight="bold",
                        color=ecol,
                        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": ecol, "alpha": 0.9},
                    )
    ax_dag.set_title("A. 4D Trajectory Lineage DAG (1→2 Split & 2→1 Merge)", fontsize=10, fontweight="bold")
    ax_dag.set_xlabel("Scan index", fontsize=9)
    ax_dag.set_ylabel("Reciprocal-space position d (voxel)", fontsize=9)
    ax_dag.legend(fontsize=7.5, ncol=3, loc="upper left", frameon=True)
    apply_axis_style(ax_dag)

    # Panel B: Pre-split bimodal broadening vs post-split cleavage MIP strip
    # Show scan 0 (incipient), scan 1 (bimodal precursor), scan 2 (cleaved daughters)
    strip_imgs = []
    for s_i in (0, 1, 2):
        # Crop region around the fracturing grain (d ~ 10..26, chi ~ 12..28)
        mip_patch = vols[s_i].max(axis=0)[6:28, 10:30]
        mip_norm = mip_patch / max(float(mip_patch.max()), 1e-6)
        strip_imgs.append(mip_norm)
    sep = np.ones((strip_imgs[0].shape[0], 2), dtype=np.float64) * 0.95
    montage = np.concatenate([strip_imgs[0], sep, strip_imgs[1], sep, strip_imgs[2]], axis=1)
    im = ax_mip.imshow(montage, cmap="magma", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    w_p = strip_imgs[0].shape[1]
    ax_mip.set_xticks([w_p * 0.5, w_p * 1.5 + 2, w_p * 2.5 + 4])
    ax_mip.set_xticklabels(
        ["Scan 0\n(Incipient)", "Scan 1\n(Bimodal Precursor)", "Scan 2\n(1→2 Cleavage)"], fontsize=8.5
    )
    ax_mip.set_yticks([])
    ax_mip.set_title("B. Reciprocal-Space Reflection Broadening to Cleavage (μ-MIP)", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax_mip, fraction=0.042, pad=0.03, label="Normalized intensity")

    # Panel C: Persistent Precursor Decomposition (Fracturing Grain vs Control)
    top_tid = cands[0].track_id if cands else 0
    top_obs = sorted([o for o in obs_scores if o.track_id == top_tid], key=lambda o: o.scan_idx)
    ctrl_tids = [
        tid
        for tid in sorted({o.track_id for o in obs_scores})
        if tid != top_tid and len([o for o in obs_scores if o.track_id == tid]) == 5
    ]
    ctrl_tid = ctrl_tids[0] if ctrl_tids else top_tid
    ctrl_obs = sorted([o for o in obs_scores if o.track_id == ctrl_tid], key=lambda o: o.scan_idx)

    ax_prec.plot(
        [o.scan_idx for o in top_obs],
        [o.precursor_score for o in top_obs],
        "o-",
        color="#DC2626",
        linewidth=2.3,
        markersize=6.5,
        label=f"Fracturing grain (T{top_tid} composite)",
    )
    ax_prec.plot(
        [o.scan_idx for o in top_obs],
        [o.roi_dispersion_term for o in top_obs],
        "^--",
        color=OKABE_ITO[0],
        linewidth=1.6,
        label=f"T{top_tid} 3D sub-peak + DINO patch dispersion",
    )
    ax_prec.plot(
        [o.scan_idx for o in ctrl_obs],
        [o.precursor_score for o in ctrl_obs],
        "s-",
        color="#64748B",
        linewidth=1.8,
        markersize=5.5,
        label=f"Stable control grain (T{ctrl_tid})",
    )
    ax_prec.axhline(0.35, linestyle=":", color="#475569", linewidth=1.2, label="Persistence threshold")
    ax_prec.axvline(1.5, linestyle="--", color="#DC2626", alpha=0.65, linewidth=1.2, label="Cleavage onset")
    ax_prec.set_ylim(-0.03, 1.05)
    ax_prec.set_xticks(range(5))
    ax_prec.set_xlabel("Scan index", fontsize=9)
    ax_prec.set_ylabel("Precursor score [0, 1]", fontsize=9)
    ax_prec.set_title("C. Persistent Precursor Ranking vs. Control Grain", fontsize=10, fontweight="bold")
    ax_prec.legend(fontsize=7.8, loc="upper right", frameon=True)
    apply_axis_style(ax_prec)

    # Panel D: Bundled ESRF-ID03 Operando Multi-Scan Grain Kinematics
    multi_gks = [g for g in gks if len(g.scan_indices) >= 2]
    for idx_g, gk in enumerate(multi_gks[:6]):
        col = OKABE_ITO[idx_g % len(OKABE_ITO)]
        ax_op.plot(
            gk.scan_indices,
            100.0 * gk.strain,
            "o-",
            color=col,
            linewidth=1.9,
            markersize=5.5,
            label=f"ESRF T{gk.track_id} (I0={gk.integrated_intensity[0]:.0f})",
        )
    ax_op.axhline(0.0, color="#94A3B8", linewidth=0.9)
    ax_op.set_xticks([0, 1, 2])
    ax_op.set_xticklabels(["scan0001", "scan0002", "scan0003"], fontsize=8.5)
    ax_op.set_xlabel("ESRF-ID03 Operando Scan", fontsize=9)
    ax_op.set_ylabel(r"Calibrated lattice strain $\varepsilon = \Delta d / d_0$ (%)", fontsize=9)
    ax_op.set_title("D. ESRF-ID03 Ni-Rich Cathode Per-Grain Strain Trajectories", fontsize=10, fontweight="bold")
    ax_op.legend(fontsize=7.5, ncol=2, loc="best", frameon=True)
    apply_axis_style(ax_op)

    fig.suptitle(
        "BraggTrack: 4D Chemomechanical Fracture Lineage, Precursor Scoring & Operando Kinematics",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="docs/figures/fracture_lineage_and_precursors.png",
        help="Output PNG/PDF path",
    )
    parser.add_argument("--backend", default="auto", choices=("auto", "mock", "torch"))
    args = parser.parse_args(argv)
    build_fracture_and_operando_figure(Path(args.out), backend=args.backend)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
