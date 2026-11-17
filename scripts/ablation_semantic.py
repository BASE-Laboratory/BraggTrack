"""Grid search over ``cost_alpha`` and ``cost_beta`` for semantic tracking.

Loads feature tables once (and caches ``embeddings.npz`` per scan in memory),
deep-copies per grid point, optionally merges embeddings, runs
:class:`~braggtrack.tracking.lifecycle.build_tracks`, and records
:class:`~braggtrack.tracking.metrics.compute_tracking_metrics` for every
(α, β) pair.

Example:

.. code-block:: bash

   python scripts/ablation_semantic.py \\
     --indir artifacts/segmentation \\
     --embedding-dir artifacts/embedding \\
     --betas 0,0.25,0.5,1.0 \\
     --output artifacts/ablation/report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from braggtrack.cli._utils import load_feature_csv
from braggtrack.cli.track_dataset import _load_embeddings_npz, _merge_embeddings
from braggtrack.tracking import (
    GeometrySemanticCost,
    PositionShapeCost,
    build_tracks,
    compute_tracking_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", default="artifacts/segmentation", help="Directory with scan*/features.csv")
    p.add_argument(
        "--embedding-dir",
        default=None,
        help="Embeddings root (required if any beta > 0)",
    )
    p.add_argument(
        "--alphas",
        default="1.0",
        help="Comma-separated cost_alpha values",
    )
    p.add_argument(
        "--betas",
        default="0,0.25,0.5,1.0",
        help="Comma-separated cost_beta values (0 = geometry-only)",
    )
    p.add_argument("--output", default="artifacts/ablation/report.json", help="JSON output path")
    p.add_argument("--position-weight", type=float, default=1.0)
    p.add_argument("--shape-weight", type=float, default=0.5)
    p.add_argument("--gate-mu", type=float, default=math.inf)
    p.add_argument("--gate-chi", type=float, default=math.inf)
    p.add_argument("--gate-d", type=float, default=math.inf)
    p.add_argument("--max-cost", type=float, default=math.inf)
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Number of bootstrap resamples over tracks for 95%% CIs (0 disables)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap resampling")
    return p


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _load_scan_tables(indir: Path) -> tuple[list[list[dict[str, Any]]], list[str]]:
    scan_dirs = sorted(d for d in indir.iterdir() if d.is_dir() and d.name.startswith("scan"))
    tables: list[list[dict[str, Any]]] = []
    names: list[str] = []
    for sd in scan_dirs:
        feat = sd / "features.csv"
        if feat.exists():
            tables.append(load_feature_csv(feat))
            names.append(sd.name)
    return tables, names


def _bootstrap_metric_cis(
    G: nx.DiGraph,
    n_scans: int,
    n_samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap 95% confidence intervals by resampling track subgraphs with replacement."""
    by_tid: dict[int, list[str]] = {}
    for nid, data in G.nodes(data=True):
        by_tid.setdefault(int(data["track_id"]), []).append(nid)

    tids = sorted(by_tid)
    if not tids or n_samples <= 0:
        return {}

    rng = np.random.default_rng(seed)
    frag_vals: list[float] = []
    switch_vals: list[float] = []

    for _ in range(n_samples):
        sampled_tids = rng.choice(tids, size=len(tids), replace=True)
        H = nx.DiGraph()
        for copy_idx, tid in enumerate(sampled_tids):
            new_tid = copy_idx + 1
            node_map: dict[str, str] = {}
            for nid in by_tid[int(tid)]:
                new_nid = f"b{copy_idx}_{nid}"
                node_map[nid] = new_nid
                attrs = dict(G.nodes[nid])
                attrs["track_id"] = new_tid
                H.add_node(new_nid, **attrs)
            for u in by_tid[int(tid)]:
                for v in G.successors(u):
                    if v in node_map:
                        H.add_edge(node_map[u], node_map[v], **dict(G.edges[u, v]))
        m = compute_tracking_metrics(H, n_scans=n_scans)
        frag_vals.append(float(m["fragmentation_ratio"]))
        switch_vals.append(float(m["id_switch_rate"]))

    frag_lo, frag_hi = np.percentile(frag_vals, [2.5, 97.5])
    sw_lo, sw_hi = np.percentile(switch_vals, [2.5, 97.5])
    return {
        "fragmentation_ratio_ci95": [round(float(frag_lo), 4), round(float(frag_hi), 4)],
        "id_switch_rate_ci95": [round(float(sw_lo), 4), round(float(sw_hi), 4)],
    }


def _evaluate_tables(
    scan_tables: list[list[dict[str, Any]]],
    scan_names: list[str],
    *,
    cost_alpha: float,
    cost_beta: float,
    position_weight: float,
    shape_weight: float,
    gate_mu: float,
    gate_chi: float,
    gate_d: float,
    max_cost: float,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    geo = PositionShapeCost(
        position_weight=position_weight,
        shape_weight=shape_weight,
        gate_mu=gate_mu,
        gate_chi=gate_chi,
        gate_d=gate_d,
    )
    if cost_beta != 0.0:
        cost_fn = GeometrySemanticCost(geo, cost_alpha=cost_alpha, cost_beta=cost_beta)
    else:
        cost_fn = geo

    G = build_tracks(scan_tables, cost_fn=cost_fn, max_cost=max_cost)
    n_scans = len(scan_tables)
    metrics = compute_tracking_metrics(G, n_scans=n_scans)
    cis = _bootstrap_metric_cis(G, n_scans, bootstrap_samples, seed) if bootstrap_samples > 0 else {}
    return {
        "cost_alpha": cost_alpha,
        "cost_beta": cost_beta,
        "n_scans": n_scans,
        "scan_names": scan_names,
        **metrics,
        **cis,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    indir = Path(args.indir)
    emb_root = Path(args.embedding_dir) if args.embedding_dir else None
    alphas = _parse_float_list(args.alphas)
    betas = _parse_float_list(args.betas)

    if any(b != 0.0 for b in betas) and emb_root is None:
        print(json.dumps({"error": "--embedding-dir required when any beta > 0"}))
        return 1

    if not indir.is_dir():
        print(json.dumps({"error": "indir not found", "path": str(indir)}))
        return 1

    base_tables, scan_names = _load_scan_tables(indir)
    if not base_tables:
        print(json.dumps({"error": "No feature tables", "indir": str(indir)}))
        return 1

    emb_cache: dict[str, dict[int, Any]] = {}
    if emb_root is not None and any(b != 0.0 for b in betas):
        for sname in scan_names:
            npz = emb_root / sname / "embeddings.npz"
            if not npz.exists():
                print(json.dumps({"error": "Missing embeddings", "path": str(npz)}))
                return 1
            emb_cache[sname] = _load_embeddings_npz(npz)

    rows_out: list[dict[str, Any]] = []
    for a in alphas:
        for b in betas:
            tables = [copy.deepcopy(t) for t in base_tables]
            if b != 0.0:
                for rows, sname in zip(tables, scan_names):
                    _merge_embeddings(rows, emb_cache[sname])
            row = _evaluate_tables(
                tables,
                scan_names,
                cost_alpha=a,
                cost_beta=b,
                position_weight=args.position_weight,
                shape_weight=args.shape_weight,
                gate_mu=args.gate_mu,
                gate_chi=args.gate_chi,
                gate_d=args.gate_d,
                max_cost=args.max_cost,
                bootstrap_samples=max(0, int(args.bootstrap_samples)),
                seed=int(args.seed),
            )
            rows_out.append(row)

    report = {
        "schema_version": "ablation.v1",
        "indir": str(indir.resolve()),
        "embedding_dir": str(emb_root.resolve()) if emb_root else None,
        "grid": {"alphas": alphas, "betas": betas},
        "bootstrap_samples": max(0, int(args.bootstrap_samples)),
        "runs": rows_out,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"wrote": str(out_path.resolve()), "n_runs": len(rows_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
