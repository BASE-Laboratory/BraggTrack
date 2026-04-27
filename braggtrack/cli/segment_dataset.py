"""Run classical segmentation over discovered scan files and write artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from braggtrack.io import (
    MissingH5DependencyError,
    discover_operando_scans,
    load_primary_volume,
    resolve_dataset_root,
)
from braggtrack.segmentation import (
    extract_instance_table,
    fill_holes_binary,
    otsu_threshold,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root with scan folders (default: data/sample_operando if present, else .)",
    )
    parser.add_argument("--outdir", default="artifacts/week2", help="Output artifact directory")
    parser.add_argument("--blur-passes", type=int, default=1)
    parser.add_argument("--seed-separation", type=int, default=1)
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
    return parser


def _synth_volume_from_file(path: Path, size: int = 24) -> np.ndarray:
    """Generate a deterministic synthetic volume with Gaussian blobs."""
    digest = hashlib.sha256(path.read_bytes()[:4096]).digest()
    seed_vals = [b for b in digest[:12]]
    volume = np.ones((size, size, size), dtype=np.float64)
    centers = [
        (4 + seed_vals[0] % 8, 4 + seed_vals[1] % 8, 4 + seed_vals[2] % 8),
        (10 + seed_vals[3] % 8, 10 + seed_vals[4] % 8, 10 + seed_vals[5] % 8),
        (6 + seed_vals[6] % 10, 6 + seed_vals[7] % 10, 6 + seed_vals[8] % 10),
    ]
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    for cz, cy, cx in centers:
        amp = 10.0 + (seed_vals[(cz + cy + cx) % len(seed_vals)] % 20)
        sigma_blob = 1.5
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume += amp * np.exp(-d2 / (2.0 * sigma_blob ** 2))
    return volume


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notebook(path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Week 2 Visual QC\n",
                    "Loads per-scan feature tables and overlays up to 20 representative objects.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import csv, json\n",
                    "from pathlib import Path\n",
                    "import matplotlib.pyplot as plt\n",
                    "root = Path('artifacts/week2')\n",
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
                ],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=2))


def main() -> int:
    args = build_parser().parse_args()
    scans = discover_operando_scans(resolve_dataset_root(args.root))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for scan in scans:
        scan_out = outdir / scan.scan_name
        scan_out.mkdir(parents=True, exist_ok=True)

        source = "nexus"
        try:
            volume = load_primary_volume(scan.path)
            if not isinstance(volume, np.ndarray):
                volume = np.asarray(volume, dtype=np.float64)
        except MissingH5DependencyError:
            volume = _synth_volume_from_file(scan.path)
            source = "synthetic_fallback"
        except (KeyError, ValueError):
            volume = _synth_volume_from_file(scan.path)
            source = "synthetic_fallback"

        threshold = otsu_threshold(volume.ravel())
        result = segment_classical(
            volume,
            threshold=threshold,
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
        labels = relabel_sequential(labels)

        table = extract_instance_table(labels, volume)
        _write_csv(scan_out / "features.csv", table)
        np.savez_compressed(scan_out / "labels.npz", labels=labels.astype(np.int32, copy=False))
        (scan_out / "summary.json").write_text(
            json.dumps(
                {
                    "scan": scan.scan_name,
                    "file": str(scan.path),
                    "source": source,
                    "threshold": threshold,
                    "seed_count": result.seed_count,
                    "component_count": len(table),
                    "schema_version": "week2.v1",
                    "labels_archive": str(scan_out / "labels.npz"),
                },
                indent=2,
            )
        )

        summaries.append(
            {
                "scan": scan.scan_name,
                "file": str(scan.path),
                "source": source,
                "component_count": len(table),
                "feature_rows": len(table),
                "summary": str(scan_out / "summary.json"),
                "features": str(scan_out / "features.csv"),
                "labels_archive": str(scan_out / "labels.npz"),
                "schema_version": "week2.v1",
            }
        )

    (outdir / "segmentation_summary.json").write_text(json.dumps(summaries, indent=2))
    _write_csv(outdir / "segmentation_summary.csv", summaries)
    _write_notebook(outdir / "qc" / "week2_visual_qc.ipynb")

    print(json.dumps(summaries, indent=2))
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
