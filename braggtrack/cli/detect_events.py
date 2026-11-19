"""Detect split/merge lineage events and rank persistent 4D fracture precursors."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.cli._utils import load_feature_csv, synth_volume_from_file, write_csv
from braggtrack.cli.track_dataset import _load_embeddings_npz, _merge_embeddings
from braggtrack.io import (
    BeamlineAdapter,
    LazyVolume,
    MissingH5DependencyError,
    discover_operando_scans,
    open_lazy_volume,
    resolve_dataset_root,
)
from braggtrack.semantic import DEFAULT_DINO_MODEL, make_patch_encoder
from braggtrack.semantic.dino import _requested_backend, _resolve_backend
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
    compute_tracking_metrics,
    events_to_table,
    precursors_to_table,
    tracks_to_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Optional dataset root with scan folders for 3D spot ROI sub-peak + DINO patch dispersion",
    )
    parser.add_argument(
        "--segdir",
        default="artifacts/segmentation",
        help="Segmentation root containing scanXXXX/features.csv and optional labels.npz",
    )
    parser.add_argument(
        "--embedding-dir",
        default=None,
        help="Optional embedding root containing scanXXXX/embeddings.npz",
    )
    parser.add_argument("--outdir", default="artifacts/events", help="Output directory for event/precursor reports")
    parser.add_argument(
        "--backend",
        choices=("none", "auto", "mock", "torch"),
        default="none",
        help="Optional ViT patch-encoder backend for ROI dispersion ('none' uses classical 3D sub-peaks only)",
    )
    parser.add_argument("--model", default=DEFAULT_DINO_MODEL, help="HF model ID when backend=torch")
    parser.add_argument(
        "--cost-model",
        choices=("position_shape", "wasserstein_bures"),
        default="position_shape",
        help="Geometric association cost model ('position_shape' or 3D Gaussian 'wasserstein_bures')",
    )
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=0.5)
    parser.add_argument("--bures-weight", type=float, default=1.0)
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
    parser.add_argument("--max-gap", type=int, default=0, help="Max missing intermediate scans to bridge (0 disables)")
    parser.add_argument("--gap-max-cost", type=float, default=25.0, help="Max penalized cost for gap-closing bridges")
    parser.add_argument("--cost-alpha", type=float, default=1.0)
    parser.add_argument("--cost-beta", type=float, default=0.0)
    parser.add_argument("--split-max-distance", type=float, default=25.0)
    parser.add_argument("--min-intensity-ratio", type=float, default=0.4)
    parser.add_argument("--max-intensity-ratio", type=float, default=1.6)
    parser.add_argument("--min-fragment-share", type=float, default=0.1)
    parser.add_argument("--precursor-threshold", type=float, default=0.40)
    parser.add_argument("--precursor-persistence", type=int, default=2)
    parser.add_argument("--precursor-baseline-reference", action="store_true")
    parser.add_argument("--precursor-baseline-scans", type=int, default=2)
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow deterministic synthetic volume fallback if an HDF5 file lacks a valid 3D dataset",
    )
    return parser


def _try_load_roi_inputs(
    stack: ExitStack,
    root_arg: str | None,
    scan_dirs: list[Path],
    scan_names: list[str],
    *,
    allow_synthetic_fallback: bool,
    load_volumes_without_labels: bool = False,
) -> tuple[dict[int, Any], dict[int, np.ndarray], dict[int, str | None]]:
    volumes: dict[int, Any] = {}
    label_volumes: dict[int, np.ndarray] = {}
    timestamps: dict[int, str | None] = {}

    for idx, sd in enumerate(scan_dirs):
        lab_path = sd / "labels.npz"
        if lab_path.exists():
            label_volumes[idx] = np.load(lab_path)["labels"]

    try:
        root_path = resolve_dataset_root(root_arg)
        scans_fs = discover_operando_scans(root_path)
    except Exception:
        return {}, label_volumes, timestamps

    if not scans_fs:
        return {}, label_volumes, timestamps

    scan_by_name = {s.scan_name: s for s in scans_fs}
    try:
        seq = BeamlineAdapter(root_path).build_sequence()
        meta_by_name = {m.scan_name: m for m in seq.scans}
        for idx, sname in enumerate(scan_names):
            m = meta_by_name.get(sname)
            if m and m.start_time:
                timestamps[idx] = m.start_time.isoformat()
    except Exception:
        pass

    for idx, sname in enumerate(scan_names):
        sf = scan_by_name.get(sname)
        if sf is None or (idx not in label_volumes and not load_volumes_without_labels):
            continue
        try:
            vol: LazyVolume = stack.enter_context(open_lazy_volume(sf.path))
        except MissingH5DependencyError:
            vol = LazyVolume.from_array(synth_volume_from_file(sf.path))
        except (KeyError, ValueError):
            if not allow_synthetic_fallback:
                continue
            vol = LazyVolume.from_array(synth_volume_from_file(sf.path))
        if idx not in label_volumes or vol.shape == label_volumes[idx].shape:
            volumes[idx] = vol

    return volumes, label_volumes, timestamps


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    segdir = Path(args.segdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not segdir.is_dir():
        print(json.dumps({"error": "Segmentation directory not found", "segdir": str(segdir)}))
        return 1

    scan_dirs = sorted(d for d in segdir.iterdir() if d.is_dir() and d.name.startswith("scan"))
    scan_tables: list[list[dict[str, Any]]] = []
    scan_names: list[str] = []
    used_scan_dirs: list[Path] = []
    for sd in scan_dirs:
        feat_path = sd / "features.csv"
        if feat_path.exists():
            scan_tables.append(load_feature_csv(feat_path))
            scan_names.append(sd.name)
            used_scan_dirs.append(sd)

    if not scan_tables:
        print(json.dumps({"error": "No feature tables found", "segdir": str(segdir)}))
        return 1

    emb_root = Path(args.embedding_dir) if args.embedding_dir else None
    if args.cost_beta != 0.0 and emb_root is None:
        print(json.dumps({"error": "--cost-beta > 0 requires --embedding-dir"}))
        return 1

    if emb_root is not None:
        for rows, sname in zip(scan_tables, scan_names):
            npz = emb_root / sname / "embeddings.npz"
            if npz.exists():
                _merge_embeddings(rows, _load_embeddings_npz(npz))
            elif args.cost_beta != 0.0:
                print(json.dumps({"error": "Missing embeddings.npz", "path": str(npz)}))
                return 1

    with ExitStack() as stack:
        volumes, label_volumes, scan_timestamps = _try_load_roi_inputs(
            stack,
            args.root,
            used_scan_dirs,
            scan_names,
            allow_synthetic_fallback=bool(args.allow_synthetic_fallback),
            load_volumes_without_labels=args.drift_mode in ("fft", "fft_kalman"),
        )

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
        cost_fn: CostFunction = (
            GeometrySemanticCost(geo, cost_alpha=args.cost_alpha, cost_beta=args.cost_beta)
            if args.cost_beta != 0.0
            else geo
        )
        gap_cfg = GapClosingConfig(max_gap=args.max_gap, max_cost=args.gap_max_cost) if args.max_gap > 0 else None
        sm_cfg = SplitMergeConfig(
            max_distance=args.split_max_distance,
            gate_mu=args.gate_mu,
            gate_chi=args.gate_chi,
            gate_d=args.gate_d,
            min_intensity_ratio=args.min_intensity_ratio,
            max_intensity_ratio=args.max_intensity_ratio,
            min_fragment_share=args.min_fragment_share,
        )

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
        lineage_events = list(G.graph.get("lineage_events", []))
        track_rows = tracks_to_table(G)
        for tr in track_rows:
            tr.pop("embedding", None)

        has_roi_labels = bool(volumes and label_volumes)
        roi_volumes = {k: v for k, v in volumes.items() if k in label_volumes} if has_roi_labels else None
        resolved_backend = (
            None if args.backend == "none" else _resolve_backend(_requested_backend(args.backend))  # type: ignore[arg-type]
        )
        patch_encoder = (
            make_patch_encoder(resolved_backend, model_name=args.model)
            if (resolved_backend is not None and roi_volumes and label_volumes)
            else None
        )

        prec_cfg = PrecursorConfig(
            score_threshold=args.precursor_threshold,
            min_persistence=max(1, args.precursor_persistence),
            baseline_reference=bool(args.precursor_baseline_reference),
            baseline_scans=max(1, int(args.precursor_baseline_scans)),
        )
        candidates, obs_scores = compute_fracture_precursors(
            track_rows,
            prec_cfg,
            volumes=roi_volumes,
            label_volumes=label_volumes if roi_volumes else None,
            patch_encoder=patch_encoder,
            lineage_events=lineage_events,
        )

    event_rows = events_to_table(lineage_events)
    prec_rows = precursors_to_table(candidates)
    obs_rows = [
        {
            "track_id": s.track_id,
            "scan_idx": s.scan_idx,
            "label": s.label,
            "anisotropy": s.anisotropy,
            "anisotropy_term": s.anisotropy_term,
            "spread_growth": s.spread_growth,
            "spread_term": s.spread_term,
            "roi_subpeak_count": s.roi_subpeak_count,
            "roi_dispersion_term": s.roi_dispersion_term,
            "precursor_score": s.precursor_score,
        }
        for s in obs_scores
    ]

    write_csv(outdir / "tracks.csv", track_rows)
    write_csv(outdir / "events.csv", event_rows)
    (outdir / "events.json").write_text(json.dumps(event_rows, indent=2))
    write_csv(outdir / "precursors.csv", prec_rows)
    write_csv(outdir / "precursor_observations.csv", obs_rows)

    summary = {
        "schema_version": "events.v1",
        "scan_names": scan_names,
        "n_scans": len(scan_tables),
        "cost_model": args.cost_model,
        "drift_mode": args.drift_mode,
        "total_tracks": metrics["total_tracks"],
        "gap_closed_count": metrics["gap_closed_count"],
        "split_count": metrics["split_count"],
        "merge_count": metrics["merge_count"],
        "event_count": len(event_rows),
        "precursor_candidate_count": len(prec_rows),
        "roi_dispersion_enabled": bool(roi_volumes and label_volumes),
        "backend": resolved_backend if patch_encoder is not None else None,
        "model": (args.model if resolved_backend == "torch" else "mock-hash") if patch_encoder is not None else None,
        "top_precursor_track_id": candidates[0].track_id if candidates else None,
    }
    (outdir / "events_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
