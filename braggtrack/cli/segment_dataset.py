"""Run segmentation over discovered scan files and write artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from braggtrack.cli._utils import synth_volume_from_file, write_csv, write_qc_notebook
from braggtrack.io import (
    AxisCalibration,
    MissingH5DependencyError,
    discover_operando_scans,
    discover_primary_volume_path,
    extract_axis_calibration,
    load_primary_volume,
    resolve_dataset_root,
)
from braggtrack.segmentation import (
    extract_instance_table,
    fill_holes_binary,
    flag_outlier_frames,
    merge_nearby_labels,
    otsu_threshold,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
    segment_dino,
    smooth_thresholds,
)
from braggtrack.semantic import DEFAULT_DINO_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root with scan folders (default: data/sample_operando if present, else .)",
    )
    parser.add_argument("--outdir", default="artifacts/segmentation", help="Output artifact directory")
    parser.add_argument(
        "--method",
        choices=["classical", "dino"],
        default="classical",
        help="Segmentation method: classical (LoG + watershed) or dino (experimental DINO features + HDBSCAN)",
    )
    parser.add_argument("--blur-passes", type=int, default=1)
    parser.add_argument("--seed-separation", type=int, default=2)
    parser.add_argument("--h-value", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument(
        "--seed-peak-fraction",
        type=float,
        default=0.2,
        help="Seed must exceed this fraction of the robust peak (p99.99) of LoG response inside the foreground",
    )
    parser.add_argument(
        "--seed-response-percentile",
        type=float,
        default=99.95,
        help="Seed must also exceed this percentile of the LoG response inside the foreground",
    )
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=1.0,
        help="Multiply Otsu threshold by this fraction to capture diffuse spots (1.0 = original Otsu)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling-median window across scans for classical Otsu threshold smoothing (1 disables)",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=15.0,
        help="Merge adjacent labels whose centroids are within this many voxels (0 disables)",
    )
    # Physical axis calibration
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="Path to JSON file defining AxisCalibration (mu/chi/d origin and step)",
    )
    parser.add_argument("--mu-origin", type=float, default=None, help="Physical mu origin (degrees)")
    parser.add_argument("--mu-step", type=float, default=None, help="Physical mu step per z-voxel (degrees)")
    parser.add_argument("--chi-origin", type=float, default=None, help="Physical chi origin (degrees)")
    parser.add_argument("--chi-step", type=float, default=None, help="Physical chi step per x-voxel (degrees)")
    parser.add_argument("--d-origin", type=float, default=None, help="Physical d-spacing origin (angstrom)")
    parser.add_argument("--d-step", type=float, default=None, help="Physical d-spacing step per y-voxel (angstrom)")
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow deterministic synthetic volume fallback if an HDF5 file lacks a valid 3D dataset",
    )
    # DINO-specific arguments
    parser.add_argument(
        "--dino-backend",
        choices=["auto", "mock", "torch"],
        default=None,
        help="DINO backend (default: auto-detect torch, fall back to mock)",
    )
    parser.add_argument(
        "--dino-model",
        default=DEFAULT_DINO_MODEL,
        help="HuggingFace model ID for the DINO torch backend",
    )
    parser.add_argument("--dino-pca-components", type=int, default=16, help="PCA components for DINO feature reduction")
    parser.add_argument("--dino-min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size")
    parser.add_argument("--dino-min-samples", type=int, default=2, help="HDBSCAN min_samples")
    parser.add_argument(
        "--dino-min-overlap",
        type=float,
        default=0.2,
        help="Min overlap fraction for 3D slice stitching",
    )
    return parser


def _resolve_cli_calibration(args: argparse.Namespace) -> AxisCalibration | None:
    raw: dict[str, object] = {}
    if args.calibration_json:
        raw.update(json.loads(Path(args.calibration_json).read_text()))
    cli_overrides = {
        "mu_origin": args.mu_origin,
        "mu_step": args.mu_step,
        "chi_origin": args.chi_origin,
        "chi_step": args.chi_step,
        "d_origin": args.d_origin,
        "d_step": args.d_step,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            raw[key] = val
    if not raw:
        return None
    return AxisCalibration.from_dict(raw)


_SEGMENTATION_QC_CODE = [
    "import csv, json\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "root = Path('artifacts/segmentation')\n",
    "for scan_dir in sorted(root.glob('scan*')):\n",
    "    table = scan_dir / 'features.csv'\n",
    "    if not table.exists():\n",
    "        continue\n",
    "    rows = list(csv.DictReader(table.open()))\n",
    "    rows = sorted(rows, key=lambda r: float(r['integrated_intensity']), reverse=True)[:20]\n",
    "    print(scan_dir.name, 'objects:', len(rows))\n",
    "    fig, ax = plt.subplots(figsize=(8, 4))\n",
    "    ax.set_title(f'{scan_dir.name} top-20 object intensities')\n",
    "    ax.bar(range(len(rows)), [float(r['integrated_intensity']) for r in rows])\n",
    "    ax.set_xlabel('Object rank')\n",
    "    ax.set_ylabel('Integrated intensity')\n",
    "    plt.show()\n",
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scans = discover_operando_scans(resolve_dataset_root(args.root))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    explicit_cal = _resolve_cli_calibration(args)

    def _load_one(scan_obj: object) -> tuple[np.ndarray, str, str | None, AxisCalibration]:
        scan_path = scan_obj.path  # type: ignore[attr-defined]
        src = "nexus"
        dpath: str | None = None
        scan_cal = explicit_cal or AxisCalibration()
        try:
            dpath = discover_primary_volume_path(scan_path)
            vol = load_primary_volume(scan_path)
            if explicit_cal is None:
                scan_cal = extract_axis_calibration(scan_path)
        except MissingH5DependencyError:
            vol = synth_volume_from_file(scan_path)
            src = "synthetic_fallback"
        except (KeyError, ValueError):
            if not args.allow_synthetic_fallback:
                raise
            vol = synth_volume_from_file(scan_path)
            src = "synthetic_fallback"
        return vol, src, dpath, scan_cal

    # First pass: compute scalar per-scan Otsu thresholds in O(1) volume memory.
    raw_thresholds: list[float] = []
    for scan in scans:
        vol_tmp, _, _, _ = _load_one(scan)
        raw_thresholds.append(otsu_threshold(vol_tmp.ravel()) * float(args.threshold_fraction))
        del vol_tmp

    if args.method == "classical" and args.smooth_window > 1 and len(raw_thresholds) > 1:
        effective_thresholds = [float(x) for x in smooth_thresholds(raw_thresholds, window=args.smooth_window)]
        outlier_flags = [bool(x) for x in flag_outlier_frames(raw_thresholds, window=args.smooth_window)]
    else:
        effective_thresholds = list(raw_thresholds)
        outlier_flags = [False] * len(raw_thresholds)

    summaries: list[dict[str, object]] = []
    for idx, scan in enumerate(scans):
        volume, source, ds_path, cal = _load_one(scan)
        scan_out = outdir / scan.scan_name  # type: ignore[attr-defined]
        scan_out.mkdir(parents=True, exist_ok=True)
        raw_thr = raw_thresholds[idx]
        eff_thr = effective_thresholds[idx]

        if args.method == "dino":
            result = segment_dino(
                volume,
                backend=args.dino_backend,
                model_name=args.dino_model,
                n_components_pca=args.dino_pca_components,
                min_cluster_size=args.dino_min_cluster_size,
                min_samples=args.dino_min_samples,
                threshold_fraction=float(args.threshold_fraction),
                min_overlap_fraction=args.dino_min_overlap,
            )
        else:
            result = segment_classical(
                volume,
                threshold=eff_thr,
                blur_passes=max(1, args.blur_passes),
                h_value=float(args.h_value),
                min_seed_separation=max(1, args.seed_separation),
                seed_peak_fraction=float(args.seed_peak_fraction),
                seed_response_percentile=float(args.seed_response_percentile),
            )

        labels = remove_small_objects(result.labeled_volume, min_size=max(1, args.min_size))
        binary = labels > 0
        binary = fill_holes_binary(binary)
        labels = np.where(binary, labels, 0)
        if args.merge_distance > 0:
            labels = merge_nearby_labels(labels, volume, max_centroid_distance=args.merge_distance)
        labels = relabel_sequential(labels)

        active_cal = cal if cal.calibrated else None
        table = extract_instance_table(labels, volume, calibration=active_cal)
        write_csv(scan_out / "features.csv", table)
        np.savez_compressed(scan_out / "labels.npz", labels=labels.astype(np.int32, copy=False))
        (scan_out / "summary.json").write_text(
            json.dumps(
                {
                    "scan": scan.scan_name,  # type: ignore[attr-defined]
                    "file": str(scan.path),  # type: ignore[attr-defined]
                    "source": source,
                    "dataset_path": ds_path,
                    "method": args.method,
                    "raw_threshold": raw_thr,
                    "threshold": result.threshold,
                    "threshold_fraction": args.threshold_fraction,
                    "effective_threshold": result.threshold,
                    "smooth_window": args.smooth_window if args.method == "classical" else 1,
                    "is_outlier_frame": outlier_flags[idx],
                    "merge_distance": args.merge_distance,
                    "seed_count": result.seed_count,
                    "component_count": len(table),
                    "calibration": cal.to_dict(),
                    "schema_version": "segmentation.v1",
                    "labels_archive": str(scan_out / "labels.npz"),
                },
                indent=2,
            )
        )

        summaries.append(
            {
                "scan": scan.scan_name,  # type: ignore[attr-defined]
                "file": str(scan.path),  # type: ignore[attr-defined]
                "source": source,
                "dataset_path": ds_path,
                "component_count": len(table),
                "feature_rows": len(table),
                "calibrated": cal.calibrated,
                "summary": str(scan_out / "summary.json"),
                "features": str(scan_out / "features.csv"),
                "labels_archive": str(scan_out / "labels.npz"),
                "schema_version": "segmentation.v1",
            }
        )

    (outdir / "segmentation_summary.json").write_text(json.dumps(summaries, indent=2))
    write_csv(outdir / "segmentation_summary.csv", summaries)
    write_qc_notebook(
        outdir / "qc" / "segmentation_visual_qc.ipynb",
        title="Segmentation Visual QC",
        code_source=_SEGMENTATION_QC_CODE,
    )

    print(json.dumps(summaries, indent=2))
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
