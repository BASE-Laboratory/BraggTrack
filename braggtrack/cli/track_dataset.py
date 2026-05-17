"""Run tracking across segmented scan feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.cli._utils import load_feature_csv, write_csv, write_qc_notebook
from braggtrack.tracking import (
    GeometrySemanticCost,
    PositionShapeCost,
    build_tracks,
    compute_tracking_metrics,
    tracks_to_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", default="artifacts/segmentation", help="Directory with per-scan feature CSVs"
    )
    parser.add_argument("--outdir", default="artifacts/tracking", help="Output artifact directory")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=0.5)
    parser.add_argument("--gate-mu", type=float, default=float("inf"))
    parser.add_argument("--gate-chi", type=float, default=float("inf"))
    parser.add_argument("--gate-d", type=float, default=float("inf"))
    parser.add_argument("--max-cost", type=float, default=float("inf"))
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


def main() -> int:
    args = build_parser().parse_args()
    indir = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover per-scan feature tables.
    scan_dirs = sorted(d for d in indir.iterdir() if d.is_dir() and d.name.startswith("scan"))
    scan_tables: list[list[dict]] = []
    scan_names: list[str] = []
    for sd in scan_dirs:
        feat_path = sd / "features.csv"
        if feat_path.exists():
            scan_tables.append(load_feature_csv(feat_path))
            scan_names.append(sd.name)

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

    geo = PositionShapeCost(
        position_weight=args.position_weight,
        shape_weight=args.shape_weight,
        gate_mu=args.gate_mu,
        gate_chi=args.gate_chi,
        gate_d=args.gate_d,
    )
    if args.cost_beta != 0.0:
        cost_fn = GeometrySemanticCost(
            geo,
            cost_alpha=args.cost_alpha,
            cost_beta=args.cost_beta,
        )
    else:
        cost_fn = geo

    G = build_tracks(scan_tables, cost_fn=cost_fn, max_cost=args.max_cost)
    metrics = compute_tracking_metrics(G, n_scans=len(scan_tables))

    track_rows = tracks_to_table(G)
    for tr in track_rows:
        tr.pop("embedding", None)
    write_csv(outdir / "tracks.csv", track_rows)
    (outdir / "tracking_metrics.json").write_text(json.dumps(metrics, indent=2))

    schema_version = "tracking_semantic.v1" if args.cost_beta != 0.0 else "tracking.v1"
    summary = {
        "scan_names": scan_names,
        "n_scans": len(scan_tables),
        "spots_per_scan": [len(t) for t in scan_tables],
        **metrics,
        "schema_version": schema_version,
        "cost_alpha": args.cost_alpha,
        "cost_beta": args.cost_beta,
        "embedding_dir": str(emb_root) if emb_root else None,
    }
    (outdir / "tracking_summary.json").write_text(json.dumps(summary, indent=2))
    write_qc_notebook(outdir / "qc" / "tracking_qc.ipynb", title="Tracking QC", code_source=_TRACKING_QC_CODE)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
