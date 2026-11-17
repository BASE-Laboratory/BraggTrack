"""One-command reproducible end-to-end BraggTrack pipeline with manifest generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import braggtrack
from braggtrack.cli import detect_events, embed_dataset, report_dataset, segment_dataset, track_dataset
from braggtrack.io import BeamlineAdapter, discover_operando_scans, resolve_dataset_root, validate_sequence
from braggtrack.semantic import DEFAULT_DINO_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root with scan folders (default: data/sample_operando if present, else .)",
    )
    parser.add_argument("--outdir", default="artifacts/pipeline", help="Root output directory for pipeline run")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file overriding pipeline stage parameters",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mock", "torch"),
        default="auto",
        help="Semantic/patch encoder backend (auto, mock, or torch)",
    )
    parser.add_argument("--model", default=DEFAULT_DINO_MODEL, help="HuggingFace model ID when backend=torch")
    parser.add_argument(
        "--method",
        choices=("classical", "dino"),
        default="classical",
        help="Segmentation method (classical 3D LoG+watershed or experimental dino)",
    )
    parser.add_argument("--smooth-window", type=int, default=5, help="Cross-scan rolling median threshold window")
    parser.add_argument("--calibration-json", default=None, help="Optional AxisCalibration JSON path")
    parser.add_argument(
        "--detector-json", default=None, help="Optional DetectorGeometry JSON path for (2theta, eta, Q)"
    )
    parser.add_argument("--cost-alpha", type=float, default=1.0, help="Geometry weight in semantic cost")
    parser.add_argument("--cost-beta", type=float, default=0.25, help="Semantic embedding weight in tracking cost")
    parser.add_argument("--max-gap", type=int, default=0, help="Max missing intermediate scans to bridge (0 disables)")
    parser.add_argument(
        "--split-max-distance", type=float, default=25.0, help="Max centroid distance for splits/merges"
    )
    parser.add_argument("--precursor-threshold", type=float, default=0.40, help="Fracture precursor score threshold")
    parser.add_argument("--precursor-persistence", type=int, default=2, help="Contiguous scans required for precursor")
    parser.add_argument(
        "--detect-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run split/merge lineage and fracture precursor detection stage (default: enabled)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate static publication figures and interactive HTML 4D explorer under <outdir>/report",
    )
    parser.add_argument("--export-imaged11", default=None, help="Optional ImageD11 .flt export path")
    parser.add_argument("--export-nexus", default=None, help="Optional NeXus NXprocess .nxs/.h5 export path")
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow deterministic synthetic volume fallback if an HDF5 file lacks a valid 3D dataset",
    )
    return parser


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _resolve_git_sha() -> str | None:
    repo_dir = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def _merge_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    effective: dict[str, Any] = {
        "backend": args.backend,
        "model": args.model,
        "method": args.method,
        "smooth_window": args.smooth_window,
        "calibration_json": args.calibration_json,
        "detector_json": args.detector_json,
        "cost_alpha": args.cost_alpha,
        "cost_beta": args.cost_beta,
        "max_gap": args.max_gap,
        "split_max_distance": args.split_max_distance,
        "precursor_threshold": args.precursor_threshold,
        "precursor_persistence": args.precursor_persistence,
        "detect_events": bool(args.detect_events),
        "report": bool(args.report),
        "export_imaged11": args.export_imaged11,
        "export_nexus": args.export_nexus,
        "allow_synthetic_fallback": bool(args.allow_synthetic_fallback),
        "segmentation_extra_args": [],
        "tracking_extra_args": [],
    }
    if args.config:
        cfg_data = json.loads(Path(args.config).read_text())
        if not isinstance(cfg_data, dict):
            raise ValueError("--config must point to a JSON object")
        effective.update(cfg_data)
    return effective


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _merge_config_overrides(args)

    root = resolve_dataset_root(cfg.get("root") or args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seg_dir = outdir / "segmentation"
    emb_dir = outdir / "embedding"
    trk_dir = outdir / "tracking"
    evt_dir = outdir / "events"
    rep_dir = outdir / "report"

    timings: dict[str, float] = {}
    t_total0 = time.perf_counter()

    # 1. Validate dataset
    t0 = time.perf_counter()
    adapter = BeamlineAdapter(root)
    sequence = adapter.build_sequence()
    issues = validate_sequence(sequence)
    val_payload = {
        "root": str(root.resolve()),
        "scan_count": len(sequence.scans),
        "is_monotonic": sequence.is_monotonic(),
        "issues": [issue.__dict__ for issue in issues],
    }
    (outdir / "validation_summary.json").write_text(json.dumps(val_payload, indent=2))
    timings["validate"] = round(time.perf_counter() - t0, 4)
    if any(issue.level == "error" for issue in issues):
        print(json.dumps({"error": "Dataset validation failed", "issues": val_payload["issues"]}))
        return 1

    # Compute input checksums
    discovered = discover_operando_scans(root)
    input_files = [
        {
            "scan": sf.scan_name,
            "path": str(sf.path.resolve()),
            "size_bytes": sf.path.stat().st_size,
            "sha256": _sha256_file(sf.path),
        }
        for sf in discovered
    ]

    # 2. Segment dataset
    t0 = time.perf_counter()
    seg_argv = [
        str(root),
        "--outdir",
        str(seg_dir),
        "--method",
        str(cfg["method"]),
        "--smooth-window",
        str(cfg["smooth_window"]),
        "--dino-backend",
        str(cfg["backend"]),
        "--dino-model",
        str(cfg["model"]),
    ]
    if cfg.get("calibration_json"):
        seg_argv.extend(["--calibration-json", str(cfg["calibration_json"])])
    if cfg.get("allow_synthetic_fallback"):
        seg_argv.append("--allow-synthetic-fallback")
    seg_argv.extend(str(x) for x in cfg.get("segmentation_extra_args", []))
    rc_seg = segment_dataset.main(seg_argv)
    timings["segment"] = round(time.perf_counter() - t0, 4)
    if rc_seg != 0:
        return rc_seg

    # 3. Embed dataset (whenever cost_beta != 0)
    need_embed = float(cfg["cost_beta"]) != 0.0
    if need_embed:
        t0 = time.perf_counter()
        emb_argv = [
            str(root),
            "--segdir",
            str(seg_dir),
            "--outdir",
            str(emb_dir),
            "--backend",
            str(cfg["backend"]),
            "--model",
            str(cfg["model"]),
        ]
        if cfg.get("allow_synthetic_fallback"):
            emb_argv.append("--allow-synthetic-fallback")
        rc_emb = embed_dataset.main(emb_argv)
        timings["embed"] = round(time.perf_counter() - t0, 4)
        if rc_emb != 0:
            return rc_emb

    # 4. Track dataset + kinematics (+ split/merge if detect_events)
    t0 = time.perf_counter()
    trk_argv = [
        str(seg_dir),
        "--outdir",
        str(trk_dir),
        "--cost-alpha",
        str(cfg["cost_alpha"]),
        "--cost-beta",
        str(cfg["cost_beta"]),
        "--max-gap",
        str(cfg["max_gap"]),
        "--split-max-distance",
        str(cfg["split_max_distance"]),
        "--precursor-threshold",
        str(cfg["precursor_threshold"]),
        "--precursor-persistence",
        str(cfg["precursor_persistence"]),
    ]
    if need_embed:
        trk_argv.extend(["--embedding-dir", str(emb_dir)])
    if cfg.get("detect_events"):
        trk_argv.append("--detect-events")
    if cfg.get("detector_json"):
        trk_argv.extend(["--detector-json", str(cfg["detector_json"])])
    if cfg.get("export_imaged11"):
        trk_argv.extend(["--export-imaged11", str(cfg["export_imaged11"])])
    if cfg.get("export_nexus") and not cfg.get("detect_events"):
        trk_argv.extend(["--export-nexus", str(cfg["export_nexus"])])
    trk_argv.extend(str(x) for x in cfg.get("tracking_extra_args", []))
    rc_trk = track_dataset.main(trk_argv)
    timings["track"] = round(time.perf_counter() - t0, 4)
    if rc_trk != 0:
        return rc_trk

    # 5. Full ROI fracture precursor + lineage analysis
    if cfg.get("detect_events"):
        t0 = time.perf_counter()
        evt_argv = [
            str(root),
            "--segdir",
            str(seg_dir),
            "--outdir",
            str(evt_dir),
            "--backend",
            str(cfg["backend"]),
            "--model",
            str(cfg["model"]),
            "--cost-alpha",
            str(cfg["cost_alpha"]),
            "--cost-beta",
            str(cfg["cost_beta"]),
            "--max-gap",
            str(cfg["max_gap"]),
            "--split-max-distance",
            str(cfg["split_max_distance"]),
            "--precursor-threshold",
            str(cfg["precursor_threshold"]),
            "--precursor-persistence",
            str(cfg["precursor_persistence"]),
        ]
        if need_embed:
            evt_argv.extend(["--embedding-dir", str(emb_dir)])
        if cfg.get("allow_synthetic_fallback"):
            evt_argv.append("--allow-synthetic-fallback")
        rc_evt = detect_events.main(evt_argv)
        timings["events"] = round(time.perf_counter() - t0, 4)
        if rc_evt != 0:
            return rc_evt

    if cfg.get("export_nexus") and cfg.get("detect_events"):
        from braggtrack.io import DetectorGeometry, export_nexus_process

        dgeom = DetectorGeometry.from_json(cfg["detector_json"]) if cfg.get("detector_json") else None
        export_nexus_process(outdir, cfg["export_nexus"], geometry=dgeom)

    manifest_path = outdir / "run_manifest.json"

    def _build_manifest(cur_total_sec: float) -> dict[str, Any]:
        return {
            "schema_version": "manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "braggtrack_version": braggtrack.__version__,
            "git_sha": _resolve_git_sha(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "dataset_root": str(root.resolve()),
            "outdir": str(outdir.resolve()),
            "config": cfg,
            "inputs": input_files,
            "stage_timings_sec": timings,
            "total_duration_sec": cur_total_sec,
            "artifacts": {
                "validation_summary": str((outdir / "validation_summary.json").resolve()),
                "segmentation_summary": str((seg_dir / "segmentation_summary.json").resolve()),
                "embedding_summary": str((emb_dir / "embedding_summary.json").resolve()) if need_embed else None,
                "tracking_summary": str((trk_dir / "tracking_summary.json").resolve()),
                "kinematics_csv": str((trk_dir / "kinematics.csv").resolve()),
                "kinematics_summary": str((trk_dir / "kinematics_summary.json").resolve()),
                "events_summary": str((evt_dir / "events_summary.json").resolve())
                if cfg.get("detect_events")
                else None,
                "events_csv": str((evt_dir / "events.csv").resolve()) if cfg.get("detect_events") else None,
                "precursors_csv": str((evt_dir / "precursors.csv").resolve()) if cfg.get("detect_events") else None,
                "html_report": str((rep_dir / "report.html").resolve()) if cfg.get("report") else None,
            },
        }

    total_sec = round(time.perf_counter() - t_total0, 4)
    manifest_path.write_text(json.dumps(_build_manifest(total_sec), indent=2))

    # 6. Optional interactive HTML & static figure report
    if cfg.get("report"):
        t0 = time.perf_counter()
        rc_rep = report_dataset.main([str(outdir), "--dataset-root", str(root), "--outdir", str(rep_dir)])
        timings["report"] = round(time.perf_counter() - t0, 4)
        if rc_rep != 0:
            return rc_rep
        total_sec = round(time.perf_counter() - t_total0, 4)
        manifest_path.write_text(json.dumps(_build_manifest(total_sec), indent=2))

    print(json.dumps({"manifest": str(manifest_path.resolve()), "total_duration_sec": total_sec}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
