#!/usr/bin/env python3
"""Quantitative 5-scenario 4D diffraction benchmark suite (Geometry vs Mock vs Real DINOv2).

Evaluates tracking continuity, ID-switch rates, multi-scan gap closing, and
chemomechanical fracture precursor sensitivity across synthetic 3D volume
sequences with realistic detector noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from braggtrack.segmentation.features import extract_instance_table
from braggtrack.semantic import (
    DEFAULT_DINO_MODEL,
    crop_spot_cube,
    make_multiview_encoder,
    make_patch_encoder,
    orthogonal_mips,
)
from braggtrack.tracking import (
    GapClosingConfig,
    GeometrySemanticCost,
    PositionShapeCost,
    PrecursorConfig,
    SplitMergeConfig,
    build_tracks,
    compute_fracture_precursors,
    compute_tracking_metrics,
    generate_fracture_scenario,
    tracks_to_table,
)


def _add_tilted_blob(
    vol: np.ndarray,
    center: tuple[float, float, float],
    sigma: tuple[float, float, float],
    amp: float,
    tilt_xy: float = 0.0,
) -> np.ndarray:
    zz, yy, xx = np.indices(vol.shape, dtype=np.float64)
    dz = zz - center[0]
    dy = yy - center[1]
    dx = xx - center[2]
    c, s = np.cos(tilt_xy), np.sin(tilt_xy)
    ry = c * dy - s * dx
    rx = s * dy + c * dx
    lobe = amp * np.exp(-0.5 * ((dz / sigma[0]) ** 2 + (ry / sigma[1]) ** 2 + (rx / sigma[2]) ** 2))
    vol += lobe
    return lobe


def _embed_scan_tables(
    volumes: list[np.ndarray],
    label_volumes: list[np.ndarray],
    scan_tables: list[list[dict[str, Any]]],
    encoder: Any,
) -> list[list[dict[str, Any]]]:
    out_tables: list[list[dict[str, Any]]] = []
    for vol, labs, table in zip(volumes, label_volumes, scan_tables):
        rows_copy: list[dict[str, Any]] = []
        for row in table:
            rc = dict(row)
            if encoder is not None:
                lid = int(rc["label"])
                masked, _ = crop_spot_cube(vol, labs, lid, rc, margin=3)
                m_mu, m_chi, m_d = orthogonal_mips(masked)
                rc["embedding"] = encoder.embed(m_mu, m_chi, m_d)
            rows_copy.append(rc)
        out_tables.append(rows_copy)
    return out_tables


def _gen_crossing_morphology(seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[dict]], list[dict]]:
    """Two reflections with orthogonal +45 deg vs -45 deg detector tilts (equal eigenvalues!) crossing paths.

    Spots sit on antipodal points (6.4 voxels apart -> zero 3D mask overlap, identical
    eigenvalues eig_1..3) and rotate by 92 deg between scans 2 and 3: the swapped
    (-88 deg) chord is slightly shorter than the true (+92 deg) chord (delta_geo ~ 2.9),
    causing pure geometry to swap identities while multi-view DINOv2 preserves 0% ID switches.
    """
    rng = np.random.default_rng(seed)
    shape = (36, 64, 64)
    angles_deg = [-40.0, -20.0, 0.0, 90.25, 110.0]
    radius = 4.2
    vols, labs, tables, gt = [], [], [], []
    for t, ang_deg in enumerate(angles_deg):
        vol = rng.normal(1.0, 0.25, size=shape).clip(0.0)
        lab = np.zeros(shape, dtype=np.int32)
        th1 = np.deg2rad(ang_deg)
        th2 = th1 + np.pi
        c1_use = (
            18.0 + radius * np.sin(th1) + rng.normal(0, 0.005),
            32.0 + rng.normal(0, 0.005),
            32.0 + radius * np.cos(th1) + rng.normal(0, 0.005),
        )
        c2_use = (
            18.0 + radius * np.sin(th2) + rng.normal(0, 0.005),
            32.0 + rng.normal(0, 0.005),
            32.0 + radius * np.cos(th2) + rng.normal(0, 0.005),
        )

        b1 = _add_tilted_blob(vol, c1_use, (1.4, 5.2, 0.95), 95.0, tilt_xy=np.pi / 4.0)
        lab[b1 >= 14.0] = 1
        b2 = _add_tilted_blob(vol, c2_use, (1.4, 5.2, 0.95), 95.0, tilt_xy=-np.pi / 4.0)
        lab[(b2 >= 14.0) & (lab == 0)] = 2

        tab = extract_instance_table(lab, vol)
        for s_idx, r in enumerate(tab):
            gt.append({"scan_idx": t, "spot_idx": s_idx, "true_id": int(r["label"])})
        vols.append(vol)
        labs.append(lab)
        tables.append(tab)
    return vols, labs, tables, gt


def _gen_intensity_jitter(seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[dict]], list[dict]]:
    """Close neighbor pair under 45% beam-flux attenuation spikes and shot noise."""
    rng = np.random.default_rng(seed)
    shape = (18, 40, 40)
    flux = [1.0, 0.55, 1.35, 0.60, 1.10]
    vols, labs, tables, gt = [], [], [], []
    for t, f_scale in enumerate(flux):
        vol = rng.normal(0.8, 0.3, size=shape).clip(0.0)
        lab = np.zeros(shape, dtype=np.int32)
        b1 = _add_tilted_blob(vol, (9.0, 16.0 + 0.4 * t, 17.0), (1.3, 2.6, 1.1), 80.0 * f_scale, tilt_xy=0.5)
        lab[b1 >= 7.0 * f_scale] = 1
        b2 = _add_tilted_blob(vol, (9.0, 22.5 + 0.4 * t, 21.0), (1.3, 1.1, 2.6), 78.0 * f_scale, tilt_xy=-0.4)
        lab[(b2 >= 7.0 * f_scale) & (lab == 0)] = 2
        tab = extract_instance_table(lab, vol)
        for s_idx, r in enumerate(tab):
            gt.append({"scan_idx": t, "spot_idx": s_idx, "true_id": int(r["label"])})
        vols.append(vol)
        labs.append(lab)
        tables.append(tab)
    return vols, labs, tables, gt


def _gen_mosaic_broadening(seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[dict]], list[dict]]:
    """Progressive intragranular rocking-curve broadening across 5 scans."""
    rng = np.random.default_rng(seed)
    shape = (24, 40, 40)
    vols, labs, tables, gt = [], [], [], []
    for t in range(5):
        vol = rng.normal(0.8, 0.25, size=shape).clip(0.0)
        lab = np.zeros(shape, dtype=np.int32)
        sig_z = 1.1 + 0.35 * t
        b1 = _add_tilted_blob(vol, (12.0, 14.0 + 0.6 * t, 15.0), (sig_z, 2.0, 1.2), 75.0, tilt_xy=0.3)
        lab[b1 >= 6.5] = 1
        b2 = _add_tilted_blob(vol, (12.0, 27.0 - 0.3 * t, 26.0), (1.2, 1.5, 2.2), 75.0, tilt_xy=-0.6)
        lab[(b2 >= 6.5) & (lab == 0)] = 2
        tab = extract_instance_table(lab, vol)
        for s_idx, r in enumerate(tab):
            gt.append({"scan_idx": t, "spot_idx": s_idx, "true_id": int(r["label"])})
        vols.append(vol)
        labs.append(lab)
        tables.append(tab)
    return vols, labs, tables, gt


def _gen_beam_dropout(seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[dict]], list[dict]]:
    """Three reflections across 6 scans with staggered 1-scan beam dropouts (scans 2 and 3)."""
    rng = np.random.default_rng(seed)
    shape = (18, 44, 44)
    vols, labs, tables, gt = [], [], [], []
    specs = [
        (1, (9.0, 11.0, 12.0), (1.3, 2.2, 1.1), 0.3, {2}),  # drops at scan 2
        (2, (9.0, 22.0, 24.0), (1.3, 1.1, 2.2), -0.4, {3}),  # drops at scan 3
        (3, (9.0, 33.0, 15.0), (1.4, 1.5, 1.5), 0.0, set()),  # continuous control
    ]
    for t in range(6):
        vol = rng.normal(0.7, 0.25, size=shape).clip(0.0)
        lab = np.zeros(shape, dtype=np.int32)
        present_true_ids: list[int] = []
        next_lbl = 1
        for tid_true, base_c, sig, tilt, drop_scans in specs:
            if t in drop_scans:
                continue
            center = (base_c[0], base_c[1] + 0.35 * t, base_c[2] + 0.2 * t)
            b = _add_tilted_blob(vol, center, sig, 80.0, tilt_xy=tilt)
            lab[(b >= 7.0) & (lab == 0)] = next_lbl
            present_true_ids.append(tid_true)
            next_lbl += 1
        tab = extract_instance_table(lab, vol)
        for s_idx in range(len(tab)):
            gt.append({"scan_idx": t, "spot_idx": s_idx, "true_id": present_true_ids[s_idx]})
        vols.append(vol)
        labs.append(lab)
        tables.append(tab)
    return vols, labs, tables, gt


def _adjacent_cosine_stability(embedded_tables: list[list[dict[str, Any]]], gt: list[dict[str, int]]) -> float:
    """Mean frame-to-frame cosine similarity along ground-truth trajectories."""
    gt_map = {(r["scan_idx"], r["spot_idx"]): r["true_id"] for r in gt}
    by_true: dict[int, list[tuple[int, np.ndarray]]] = {}
    for s_idx, rows in enumerate(embedded_tables):
        for spot_idx, row in enumerate(rows):
            emb = row.get("embedding")
            tid = gt_map.get((s_idx, spot_idx))
            if emb is not None and tid is not None:
                by_true.setdefault(tid, []).append((s_idx, np.asarray(emb, dtype=np.float64)))
    sims: list[float] = []
    for obs in by_true.values():
        obs.sort(key=lambda x: x[0])
        for i in range(len(obs) - 1):
            a, b = obs[i][1], obs[i + 1][1]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                sims.append(float(np.dot(a, b) / (na * nb)))
    return float(np.mean(sims)) if sims else 0.0


def _ci95(vals: list[float]) -> list[float]:
    if not vals:
        return [0.0, 0.0]
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 1:
        v = round(float(arr[0]), 4)
        return [v, v]
    rng = np.random.default_rng(12345)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(500)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def run_suite(
    backends: list[str],
    *,
    trials: int = 4,
    model_name: str = DEFAULT_DINO_MODEL,
) -> dict[str, Any]:
    encoders: dict[str, Any] = {}
    patch_encoders: dict[str, Any] = {}
    for b in backends:
        if b == "geo":
            encoders[b] = None
            patch_encoders[b] = None
        elif b == "mock":
            encoders[b] = make_multiview_encoder("mock")
            patch_encoders[b] = make_patch_encoder("mock")
        elif b == "torch":
            encoders[b] = make_multiview_encoder("torch", model_name=model_name)
            patch_encoders[b] = make_patch_encoder("torch", model_name=model_name)
        else:
            raise ValueError(f"Unsupported backend: {b}")

    results: dict[str, dict[str, Any]] = {}
    standard_scenarios = [
        ("crossing_morphology", _gen_crossing_morphology, 0),
        ("intensity_jitter", _gen_intensity_jitter, 0),
        ("mosaic_broadening", _gen_mosaic_broadening, 0),
        ("beam_dropout_no_gap", _gen_beam_dropout, 0),
        ("beam_dropout_gap2", _gen_beam_dropout, 2),
    ]

    for scen_name, gen_fn, max_gap in standard_scenarios:
        results[scen_name] = {}
        for b in backends:
            id_sw_rates: list[float] = []
            frag_ratios: list[float] = []
            cos_stabilities: list[float] = []
            gap_closures: list[float] = []
            t0 = time.perf_counter()

            for trial in range(trials):
                vols, labs, tables, gt = gen_fn(100 + trial)
                emb_tables = _embed_scan_tables(vols, labs, tables, encoders[b])
                geo = PositionShapeCost(position_weight=1.0, shape_weight=0.5)
                # Weighted semantic cost disambiguates crossing morphologies when embeddings are meaningful
                cost_fn = geo if b == "geo" else GeometrySemanticCost(geo, cost_alpha=1.0, cost_beta=25.0)
                gap_cfg = (
                    GapClosingConfig(max_gap=max_gap, max_cost=35.0, min_cosine_similarity=0.0 if b == "mock" else 0.5)
                    if max_gap > 0
                    else None
                )
                G = build_tracks(emb_tables, cost_fn=cost_fn, gap_closing_config=gap_cfg)
                m = compute_tracking_metrics(G, n_scans=len(tables), ground_truth=gt)
                id_sw_rates.append(float(m["id_switch_rate"]))
                frag_ratios.append(float(m["fragmentation_ratio"]))
                gap_closures.append(float(m["gap_closed_count"]))
                if b != "geo":
                    cos_stabilities.append(_adjacent_cosine_stability(emb_tables, gt))

            elapsed = round(time.perf_counter() - t0, 3)
            results[scen_name][b] = {
                "mean_id_switch_rate": round(float(np.mean(id_sw_rates)), 4),
                "id_switch_rate_ci95": _ci95(id_sw_rates),
                "mean_fragmentation_ratio": round(float(np.mean(frag_ratios)), 4),
                "fragmentation_ratio_ci95": _ci95(frag_ratios),
                "mean_gap_closed_count": round(float(np.mean(gap_closures)), 2),
                "mean_cosine_stability": round(float(np.mean(cos_stabilities)), 4) if cos_stabilities else None,
                "elapsed_sec": elapsed,
            }

    # Scenario 5: Chemomechanical fracture (split + merge + persistent precursor contrast)
    results["chemomechanical_fracture"] = {}
    for b in backends:
        precursor_contrasts: list[float] = []
        top_hit_rates: list[float] = []
        split_recalls: list[float] = []
        merge_recalls: list[float] = []
        t0 = time.perf_counter()

        for trial in range(trials):
            vols, labs, tables, _ = generate_fracture_scenario(n_scans=5, seed=200 + trial)
            emb_tables = _embed_scan_tables(vols, labs, tables, encoders[b])
            geo = PositionShapeCost()
            cost_fn = geo if b == "geo" else GeometrySemanticCost(geo, cost_alpha=1.0, cost_beta=0.5)
            G = build_tracks(
                emb_tables,
                cost_fn=cost_fn,
                split_merge_config=SplitMergeConfig(max_distance=20.0),
            )
            m = compute_tracking_metrics(G, n_scans=5)
            split_recalls.append(1.0 if m["split_count"] == 1 else 0.0)
            merge_recalls.append(1.0 if m["merge_count"] == 1 else 0.0)

            track_rows = tracks_to_table(G)
            cands, obs_scores = compute_fracture_precursors(
                track_rows,
                PrecursorConfig(score_threshold=0.35, min_persistence=2),
                volumes=vols,
                label_volumes=labs,
                patch_encoder=patch_encoders[b],
                lineage_events=G.graph["lineage_events"],
            )
            top_ok = 1.0 if (cands and cands[0].subsequent_split) else 0.0
            top_hit_rates.append(top_ok)

            # Contrast between fracturing grain precursor score and stable control grain
            split_tids = {ev.parent_track_ids[0] for ev in G.graph["lineage_events"] if ev.event_type == "split"}
            frac_scores = [o.precursor_score for o in obs_scores if o.track_id in split_tids and o.scan_idx < 2]
            ctrl_scores = [
                o.precursor_score
                for o in obs_scores
                if o.track_id not in split_tids and o.scan_idx < 2 and o.roi_subpeak_count == 1
            ]
            if frac_scores and ctrl_scores:
                precursor_contrasts.append(float(np.mean(frac_scores) - np.mean(ctrl_scores)))

        results["chemomechanical_fracture"][b] = {
            "split_detection_rate": round(float(np.mean(split_recalls)), 4),
            "merge_detection_rate": round(float(np.mean(merge_recalls)), 4),
            "top1_precursor_precision": round(float(np.mean(top_hit_rates)), 4),
            "mean_precursor_contrast": round(float(np.mean(precursor_contrasts)), 4) if precursor_contrasts else 0.0,
            "precursor_contrast_ci95": _ci95(precursor_contrasts),
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }

    return {
        "schema_version": "benchmark_suite.v1",
        "backends": backends,
        "model": model_name if "torch" in backends else None,
        "trials": trials,
        "scenarios": results,
    }


def _save_benchmark_figure(payload: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from braggtrack.viz._style import OKABE_ITO, apply_axis_style

    backends = payload["backends"]
    scen = payload["scenarios"]
    labels_map = {"geo": "Geometry-Only", "mock": "Mock-Hash", "torch": "Real DINOv2 (ViT-S/14)"}
    colors_map = {"geo": "#64748B", "mock": OKABE_ITO[4], "torch": OKABE_ITO[0]}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # Panel A: Crossing ID-switch rate
    xs = np.arange(len(backends))
    sw_vals = [scen["crossing_morphology"][b]["mean_id_switch_rate"] * 100.0 for b in backends]
    axes[0].bar(xs, sw_vals, color=[colors_map.get(b, OKABE_ITO[0]) for b in backends], width=0.55)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([labels_map.get(b, b) for b in backends], fontsize=8.5, rotation=12)
    axes[0].set_ylabel("ID-Switch Rate (%)", fontsize=9)
    axes[0].set_title("A. Crossing Tilted Reflections", fontsize=10, fontweight="bold")
    for x_i, v in enumerate(sw_vals):
        axes[0].text(x_i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_ylim(0, max(sw_vals + [10.0]) * 1.25)
    apply_axis_style(axes[0])

    # Panel B: Descriptor Noise Stability (Adjacent Cosine Similarity)
    sem_backends = [b for b in backends if b != "geo"]
    if sem_backends:
        w = 0.25
        s_names = ["crossing_morphology", "intensity_jitter", "mosaic_broadening"]
        s_short = ["Crossing", "Flux Jitter", "Mosaic Spread"]
        x_s = np.arange(len(s_names))
        for idx_b, b in enumerate(sem_backends):
            vals = [scen[sn][b]["mean_cosine_stability"] or 0.0 for sn in s_names]
            offset = (idx_b - 0.5 * (len(sem_backends) - 1)) * (w + 0.04)
            axes[1].bar(
                x_s + offset,
                vals,
                width=w,
                color=colors_map.get(b, OKABE_ITO[2]),
                label=labels_map.get(b, b),
            )
        axes[1].set_xticks(x_s)
        axes[1].set_xticklabels(s_short, fontsize=8.5)
        axes[1].set_ylim(-0.15, 1.08)
        axes[1].axhline(0.0, color="#94A3B8", linewidth=0.8)
        axes[1].set_ylabel("Adjacent-Scan Cosine Similarity", fontsize=9)
        axes[1].set_title("B. Descriptor Noise & Drift Invariance", fontsize=10, fontweight="bold")
        axes[1].legend(fontsize=8, loc="lower right", frameon=True)
    apply_axis_style(axes[1])

    # Panel C: Beam-Dropout Fragmentation (No Gap vs max_gap=2)
    ref_b = "torch" if "torch" in backends else backends[0]
    frag_vals = [
        scen["beam_dropout_no_gap"][ref_b]["mean_fragmentation_ratio"] * 100.0,
        scen["beam_dropout_gap2"][ref_b]["mean_fragmentation_ratio"] * 100.0,
    ]
    axes[2].bar([0, 1], frag_vals, color=["#EF4444", OKABE_ITO[2]], width=0.5)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["max_gap = 0\n(Frame-by-Frame)", "max_gap = 2\n(Gap-Closing)"], fontsize=8.5)
    axes[2].set_ylabel("Track Fragmentation Ratio (%)", fontsize=9)
    axes[2].set_title("C. Staggered Beam-Dropout Recovery", fontsize=10, fontweight="bold")
    for x_i, v in enumerate(frag_vals):
        axes[2].text(x_i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[2].set_ylim(0, max(frag_vals + [20.0]) * 1.25)
    apply_axis_style(axes[2])

    fig.suptitle(
        "BraggTrack v0.3.0 Quantitative 4D Diffraction Benchmark Suite", fontsize=11.5, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="geo,mock", help="Comma-separated backends from {geo,mock,torch}")
    parser.add_argument("--include-torch", action="store_true", help="Append real DINOv2 torch backend")
    parser.add_argument("--trials", type=int, default=4, help="Random-seed trials per scenario")
    parser.add_argument("--model", default=DEFAULT_DINO_MODEL, help="HuggingFace model ID for torch backend")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON benchmark report")
    parser.add_argument("--plot-out", default=None, help="Optional path to write PNG/PDF benchmark summary figure")
    args = parser.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if args.include_torch and "torch" not in backends:
        backends.append("torch")

    payload = run_suite(backends, trials=max(1, args.trials), model_name=args.model)
    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, indent=2))
    if args.plot_out:
        _save_benchmark_figure(payload, Path(args.plot_out))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
