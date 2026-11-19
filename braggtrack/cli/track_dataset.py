"""Run tracking across segmented scan feature tables."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.cli._utils import load_feature_csv, synth_volume_from_file, write_csv, write_qc_notebook
from braggtrack.io import (
    AxisCalibration,
    BeamlineAdapter,
    DetectorGeometry,
    LazyVolume,
    MissingH5DependencyError,
    annotate_crystallographic_columns,
    discover_operando_scans,
    export_imaged11_flt,
    export_nexus_process,
    open_lazy_volume,
    resolve_dataset_root,
)
from braggtrack.tracking import (
    CostFunction,
    GapClosingConfig,
    GeometrySemanticCost,
    PositionShapeCost,
    PrecursorConfig,
    SplitMergeConfig,
    WassersteinBuresCost,
    build_tracks,
    compute_fracture_precursors,
    compute_grain_kinematics,
    compute_tracking_metrics,
    events_to_table,
    kinematics_to_table,
    precursors_to_table,
    summarize_kinematics,
    summary_to_dict,
    tracks_to_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", default="artifacts/segmentation", help="Directory with per-scan feature CSVs"
    )
    parser.add_argument("--outdir", default="artifacts/tracking", help="Output artifact directory")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional raw HDF5 scan directory for lazy rocking-slab FFT drift registration (--drift-mode fft/fft_kalman)",
    )
    parser.add_argument(
        "--cost-model",
        choices=("position_shape", "wasserstein_bures"),
        default="position_shape",
        help="Geometric association cost model ('position_shape' or 3D Gaussian 'wasserstein_bures')",
    )
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=0.5)
    parser.add_argument(
        "--bures-weight",
        type=float,
        default=1.0,
        help="Bures 3D covariance transport weight when --cost-model=wasserstein_bures",
    )
    parser.add_argument(
        "--drift-mode",
        choices=("none", "consensus", "fft", "fft_kalman"),
        default="none",
        help="Collective rocking-band drift compensation ('none', 'consensus', 'fft', or 'fft_kalman')",
    )
    parser.add_argument("--gate-mu", type=float, default=float("inf"))
    parser.add_argument("--gate-chi", type=float, default=float("inf"))
    parser.add_argument("--gate-d", type=float, default=float("inf"))
    parser.add_argument("--max-cost", type=float, default=float("inf"))
    parser.add_argument(
        "--max-gap",
        type=int,
        default=0,
        help="Maximum missing intermediate scans to bridge via gap closing (0 disables)",
    )
    parser.add_argument(
        "--gap-max-cost",
        type=float,
        default=25.0,
        help="Maximum penalized cost allowed when bridging missing scans",
    )
    parser.add_argument(
        "--embedding-dir",
        default=None,
        help="Embedding root with scanXXXX/embeddings.npz (from embed_dataset)",
    )
    parser.add_argument(
        "--cost-alpha",
        type=float,
        default=1.0,
        help="Multiplier on geometry term when using semantic cost",
    )
    parser.add_argument(
        "--cost-beta",
        type=float,
        default=0.0,
        help="Multiplier on (1 - cos(embedding)); 0 disables semantic term",
    )
    parser.add_argument(
        "--detect-events",
        action="store_true",
        help="Resolve 1->2+ splits, 2+->1 merges, and persistent fracture precursors",
    )
    parser.add_argument(
        "--split-max-distance",
        type=float,
        default=25.0,
        help="Maximum centroid distance for split/merge association when --detect-events is enabled",
    )
    parser.add_argument(
        "--precursor-threshold",
        type=float,
        default=0.40,
        help="Minimum per-observation precursor score for persistent fracture candidate ranking",
    )
    parser.add_argument(
        "--precursor-persistence",
        type=int,
        default=2,
        help="Minimum contiguous scans above threshold to rank as a fracture precursor",
    )
    parser.add_argument(
        "--precursor-baseline-reference",
        action="store_true",
        help="Score precursors by intra-grain morphology deltas relative to initial baseline scans",
    )
    parser.add_argument(
        "--precursor-baseline-scans",
        type=int,
        default=2,
        help="Number of initial scans per lineage used as baseline when --precursor-baseline-reference is set",
    )
    parser.add_argument(
        "--detector-json",
        default=None,
        help="Optional DetectorGeometry JSON path to annotate (omega, 2theta, eta, d, |Q|) columns",
    )
    parser.add_argument(
        "--export-imaged11",
        default=None,
        help="Optional path to write an ImageD11 .flt columnfile of tracked reflections",
    )
    parser.add_argument(
        "--export-nexus",
        default=None,
        help="Optional path to write a NeXus NXprocess HDF5 summary file",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow deterministic synthetic volume fallback when opening --dataset-root HDF5 stubs",
    )
    return parser


def _load_embeddings_npz(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as z:
        labels = z["labels"]
        vectors = z["vectors"]
    out: dict[int, np.ndarray] = {}
    for i in range(int(labels.shape[0])):
        out[int(labels[i])] = np.asarray(vectors[i], dtype=np.float64)
    return out


def _merge_embeddings(rows: list[dict[str, Any]], emb: dict[int, np.ndarray]) -> None:
    for row in rows:
        lid = int(row["label"])
        if lid in emb:
            row["embedding"] = emb[lid]


def _detect_calibration(scan_dirs: list[Path]) -> AxisCalibration | None:
    for sd in scan_dirs:
        summary_path = sd / "summary.json"
        if not summary_path.exists():
            continue
        try:
            meta = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cal = meta.get("calibration")
        if isinstance(cal, dict) and bool(cal.get("calibrated", False)):
            return AxisCalibration.from_dict(cal)
    return None


_TRACKING_QC_CODE = [
    "import csv, json\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "root = Path('artifacts/tracking')\n",
    "tracks = list(csv.DictReader((root / 'tracks.csv').open()))\n",
    "\n",
    "# Group by track_id\n",
    "by_track = {}\n",
    "for r in tracks:\n",
    "    tid = int(r['track_id'])\n",
    "    by_track.setdefault(tid, []).append(r)\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "for tid, obs in sorted(by_track.items()):\n",
    "    scans = [int(r['scan_idx']) for r in obs]\n",
    "    mus = [float(r['centroid_mu']) for r in obs]\n",
    "    chis = [float(r['centroid_chi']) for r in obs]\n",
    "    ds = [float(r['centroid_d']) for r in obs]\n",
    "    axes[0].plot(scans, mus, 'o-', label=f'T{tid}')\n",
    "    axes[1].plot(scans, chis, 'o-', label=f'T{tid}')\n",
    "    axes[2].plot(scans, ds, 'o-', label=f'T{tid}')\n",
    "\n",
    "for ax, lbl in zip(axes, ['centroid_mu', 'centroid_chi', 'centroid_d']):\n",
    "    ax.set_xlabel('Scan index')\n",
    "    ax.set_ylabel(lbl)\n",
    "    ax.legend(fontsize=7)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "metrics = json.loads((root / 'tracking_metrics.json').read_text())\n",
    "print('Tracking metrics:')\n",
    "for k, v in metrics.items():\n",
    "    print(f'  {k}: {v}')\n",
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    indir = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover per-scan feature tables.
    scan_dirs = sorted(d for d in indir.iterdir() if d.is_dir() and d.name.startswith("scan"))
    scan_tables: list[list[dict]] = []
    scan_names: list[str] = []
    used_scan_dirs: list[Path] = []
    for sd in scan_dirs:
        feat_path = sd / "features.csv"
        if feat_path.exists():
            scan_tables.append(load_feature_csv(feat_path))
            scan_names.append(sd.name)
            used_scan_dirs.append(sd)

    if not scan_tables:
        print(json.dumps({"error": "No feature tables found", "indir": str(indir)}))
        return 1

    if args.cost_beta != 0.0 and not args.embedding_dir:
        print(json.dumps({"error": "--cost-beta > 0 requires --embedding-dir"}))
        return 1

    emb_root = Path(args.embedding_dir) if args.embedding_dir else None
    if emb_root is not None and args.cost_beta != 0.0:
        for rows, sname in zip(scan_tables, scan_names):
            npz = emb_root / sname / "embeddings.npz"
            if not npz.exists():
                print(json.dumps({"error": "Missing embeddings.npz", "path": str(npz)}))
                return 1
            _merge_embeddings(rows, _load_embeddings_npz(npz))

    geo: CostFunction
    if args.cost_model == "wasserstein_bures":
        geo = WassersteinBuresCost(
            position_weight=args.position_weight,
            bures_weight=args.bures_weight,
            gate_mu=args.gate_mu,
            gate_chi=args.gate_chi,
            gate_d=args.gate_d,
        )
    else:
        geo = PositionShapeCost(
            position_weight=args.position_weight,
            shape_weight=args.shape_weight,
            gate_mu=args.gate_mu,
            gate_chi=args.gate_chi,
            gate_d=args.gate_d,
        )
    if args.cost_beta != 0.0:
        cost_fn: CostFunction = GeometrySemanticCost(
            geo,
            cost_alpha=args.cost_alpha,
            cost_beta=args.cost_beta,
        )
    else:
        cost_fn = geo

    gap_cfg = GapClosingConfig(max_gap=args.max_gap, max_cost=args.gap_max_cost) if args.max_gap > 0 else None
    sm_cfg = (
        SplitMergeConfig(
            max_distance=args.split_max_distance,
            gate_mu=args.gate_mu,
            gate_chi=args.gate_chi,
            gate_d=args.gate_d,
        )
        if args.detect_events
        else None
    )

    with ExitStack() as stack:
        volumes: dict[int, LazyVolume] = {}
        scan_timestamps: dict[int, str | None] = {}
        if args.dataset_root and args.drift_mode in ("fft", "fft_kalman"):
            try:
                ds_root = resolve_dataset_root(args.dataset_root)
                scans_fs = {s.scan_name: s for s in discover_operando_scans(ds_root)}
                try:
                    seq = BeamlineAdapter(ds_root).build_sequence()
                    meta_by_name = {m.scan_name: m for m in seq.scans}
                    for idx, sname in enumerate(scan_names):
                        m = meta_by_name.get(sname)
                        if m and m.start_time:
                            scan_timestamps[idx] = m.start_time.isoformat()
                except Exception:
                    pass
                for idx, sname in enumerate(scan_names):
                    sf = scans_fs.get(sname)
                    if sf is None:
                        continue
                    try:
                        volumes[idx] = stack.enter_context(open_lazy_volume(sf.path))
                    except MissingH5DependencyError:
                        volumes[idx] = LazyVolume.from_array(synth_volume_from_file(sf.path))
                    except (KeyError, ValueError):
                        if args.allow_synthetic_fallback:
                            volumes[idx] = LazyVolume.from_array(synth_volume_from_file(sf.path))
            except Exception:
                volumes = {}

        G = build_tracks(
            scan_tables,
            cost_fn=cost_fn,
            max_cost=args.max_cost,
            gap_closing_config=gap_cfg,
            split_merge_config=sm_cfg,
            scan_timestamps=scan_timestamps or None,
            drift_mode=args.drift_mode,
            volumes=volumes or None,
        )

    metrics = compute_tracking_metrics(G, n_scans=len(scan_tables))

    active_cal = _detect_calibration(used_scan_dirs)
    calibrated_axes = active_cal is not None
    detector_geom = DetectorGeometry.from_json(args.detector_json) if args.detector_json else None

    track_rows = tracks_to_table(G)
    for tr in track_rows:
        tr.pop("embedding", None)
    if detector_geom is not None:
        track_rows = annotate_crystallographic_columns(track_rows, detector_geom, calibration=active_cal)
    write_csv(outdir / "tracks.csv", track_rows)
    (outdir / "tracking_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Per-grain kinematics export.
    gk = compute_grain_kinematics(track_rows)
    kin_summary = summarize_kinematics(gk, n_scans=len(scan_tables), calibrated_axes=calibrated_axes)
    write_csv(outdir / "kinematics.csv", kinematics_to_table(gk, include_delta_d=True))
    (outdir / "kinematics_summary.json").write_text(
        json.dumps({"schema_version": "kinematics.v1", **summary_to_dict(kin_summary)}, indent=2)
    )

    if args.detect_events:
        lineage_events = list(G.graph.get("lineage_events", []))
        event_rows = events_to_table(lineage_events)
        write_csv(outdir / "events.csv", event_rows)
        (outdir / "events.json").write_text(json.dumps(event_rows, indent=2))
        prec_candidates, _ = compute_fracture_precursors(
            track_rows,
            PrecursorConfig(
                score_threshold=args.precursor_threshold,
                min_persistence=max(1, args.precursor_persistence),
                baseline_reference=bool(args.precursor_baseline_reference),
                baseline_scans=max(1, int(args.precursor_baseline_scans)),
            ),
            lineage_events=lineage_events,
        )
        write_csv(outdir / "precursors.csv", precursors_to_table(prec_candidates))

    schema_version = "tracking_semantic.v1" if args.cost_beta != 0.0 else "tracking.v1"
    summary: dict[str, Any] = {
        "scan_names": scan_names,
        "n_scans": len(scan_tables),
        "spots_per_scan": [len(t) for t in scan_tables],
        **metrics,
        "calibrated_axes": calibrated_axes,
        "schema_version": schema_version,
        "cost_model": args.cost_model,
        "drift_mode": args.drift_mode,
        "cost_alpha": args.cost_alpha,
        "cost_beta": args.cost_beta,
        "max_gap": args.max_gap,
        "embedding_dir": str(emb_root) if emb_root else None,
        "detect_events": bool(args.detect_events),
        "detector_geometry": detector_geom.to_dict() if detector_geom is not None else None,
    }
    (outdir / "tracking_summary.json").write_text(json.dumps(summary, indent=2))
    write_qc_notebook(outdir / "qc" / "tracking_qc.ipynb", title="Tracking QC", code_source=_TRACKING_QC_CODE)

    if args.export_imaged11:
        export_imaged11_flt(track_rows, args.export_imaged11, geometry=detector_geom, calibration=active_cal)
    if args.export_nexus:
        export_nexus_process(outdir, args.export_nexus, geometry=detector_geom, calibration=active_cal)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
