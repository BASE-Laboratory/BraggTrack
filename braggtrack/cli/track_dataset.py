"""Run physics-only tracking across segmented scan feature tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from braggtrack.tracking import (
    PositionShapeCost,
    build_tracks,
    compute_tracking_metrics,
    tracks_to_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indir", nargs="?", default="artifacts/week2",
                        help="Directory with per-scan feature CSVs (Week 2 output)")
    parser.add_argument("--outdir", default="artifacts/week3",
                        help="Output artifact directory")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=0.5)
    parser.add_argument("--gate-mu", type=float, default=float("inf"))
    parser.add_argument("--gate-chi", type=float, default=float("inf"))
    parser.add_argument("--gate-d", type=float, default=float("inf"))
    parser.add_argument("--max-cost", type=float, default=float("inf"))
    return parser


def _load_feature_csv(path: Path) -> list[dict[str, Any]]:
    """Load a Week 2 features.csv into a list of dicts with numeric types."""
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            typed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v)
                except ValueError:
                    try:
                        typed[k] = float(v)
                    except ValueError:
                        typed[k] = v
            rows.append(typed)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
                    "# Week 3 Tracking QC\n",
                    "Visualises spot trajectories across scans and highlights near-overlap cases.\n",
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
                    "\n",
                    "root = Path('artifacts/week3')\n",
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
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover per-scan feature tables.
    scan_dirs = sorted(d for d in indir.iterdir() if d.is_dir() and d.name.startswith("scan"))
    scan_tables: list[list[dict]] = []
    scan_names: list[str] = []
    for sd in scan_dirs:
        feat_path = sd / "features.csv"
        if feat_path.exists():
            scan_tables.append(_load_feature_csv(feat_path))
            scan_names.append(sd.name)

    if not scan_tables:
        print(json.dumps({"error": "No feature tables found", "indir": str(indir)}))
        return 1

    cost_fn = PositionShapeCost(
        position_weight=args.position_weight,
        shape_weight=args.shape_weight,
        gate_mu=args.gate_mu,
        gate_chi=args.gate_chi,
        gate_d=args.gate_d,
    )

    G = build_tracks(scan_tables, cost_fn=cost_fn, max_cost=args.max_cost)
    metrics = compute_tracking_metrics(G, n_scans=len(scan_tables))

    track_rows = tracks_to_table(G)
    _write_csv(outdir / "tracks.csv", track_rows)
    (outdir / "tracking_metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = {
        "scan_names": scan_names,
        "n_scans": len(scan_tables),
        "spots_per_scan": [len(t) for t in scan_tables],
        **metrics,
        "schema_version": "week3.v1",
    }
    (outdir / "tracking_summary.json").write_text(json.dumps(summary, indent=2))
    _write_notebook(outdir / "qc" / "week3_tracking_qc.ipynb")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
