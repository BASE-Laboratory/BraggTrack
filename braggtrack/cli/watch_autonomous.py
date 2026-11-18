"""Autonomous DFXM "Scout-and-Zoom" beamline watcher, FAL Staged Certification, and causal replay CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from braggtrack.autonomous import (
    CausalStreamingTracker,
    OperationalDesignDomain,
    ZoomTriggerPolicy,
    evaluate_causal_replay,
)
from braggtrack.io import (
    AxisCalibration,
    BeamlineAdapter,
    DetectorGeometry,
    load_operando_log,
    resolve_dataset_root,
    synchronize_to_scans,
)
from braggtrack.io.nexus import _open_h5_readonly, _require_h5py, _resolve_volume_dataset
from braggtrack.semantic import DEFAULT_DINO_MODEL
from braggtrack.tracking.precursors import PrecursorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root containing scan subdirectories (default: data/sample_operando if present, else .)",
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "shadow", "watch", "active"),
        default="replay",
        help="FAL certification stage: 'replay' (offline), 'shadow'/'watch' (live non-actuating), or 'active' (closed-loop)",
    )
    parser.add_argument(
        "--hypothesis",
        choices=("mechanical_fracture", "phase_nucleation", "intragranular_disorder"),
        default="mechanical_fracture",
        help="FAL Level 3 experimental hypothesis governing heuristic target prioritization",
    )
    parser.add_argument("--outdir", default="artifacts/autonomous", help="Output directory for zoom triggers & logs")
    parser.add_argument(
        "--backend",
        choices=("auto", "mock", "torch"),
        default="auto",
        help="Semantic/patch encoder backend (auto, mock, or torch)",
    )
    parser.add_argument("--model", default=DEFAULT_DINO_MODEL, help="HuggingFace model ID when backend=torch")
    parser.add_argument(
        "--operando-log",
        default=None,
        help="Optional CSV/TSV/Biologic .mpt operando log to attach potential & dQ/dV to zoom triggers",
    )
    parser.add_argument("--calibration-json", default=None, help="Optional AxisCalibration JSON file")
    parser.add_argument("--detector-json", default=None, help="Optional DetectorGeometry JSON file")
    parser.add_argument("--min-precursor-score", type=float, default=0.40, help="Precursor trigger threshold")
    parser.add_argument("--min-persistence", type=int, default=2, help="Required contiguous precursor scans")
    parser.add_argument("--cooldown-scans", type=int, default=3, help="Per-lineage cooldown scans after firing")
    parser.add_argument("--max-triggers-per-scan", type=int, default=2, help="Maximum zoom targets emitted per scan")
    parser.add_argument("--lead-scans", type=float, default=1.0, help="Velocity extrapolation lead (in scan steps)")
    parser.add_argument("--window-sigma", type=float, default=3.5, help="Scan half-width sigma multiplier")
    parser.add_argument(
        "--mu-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional ODD motor soft-limits for rocking angle mu",
    )
    parser.add_argument(
        "--chi-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional ODD motor soft-limits for roll angle chi",
    )
    parser.add_argument(
        "--d-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional ODD motor soft-limits for d / 2theta / obpitch",
    )
    parser.add_argument(
        "--voltage-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("V_MIN", "V_MAX"),
        help="Optional ODD electrochemical voltage window (V); halts actuation on excursion",
    )
    parser.add_argument(
        "--max-zooms-per-lineage",
        type=int,
        default=None,
        help="Optional ODD per-grain maximum high-res zoom count to cap X-ray dose",
    )
    parser.add_argument(
        "--scan-cadence-sec",
        type=float,
        default=None,
        help="Expected beamline scan cadence t_decision (s) for FAL latency budget verification",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=24,
        help="Bounded causal tracking window size in scans (default: 24 for O(1) latency)",
    )
    parser.add_argument(
        "--zoom-budget",
        type=int,
        default=None,
        help="Optional cap on total zoom acquisitions during --mode replay baseline comparison",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Directory poll interval (s) in live modes")
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.03,
        help="File size/mtime quiescence window (s) before reading newly created HDF5 scans",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Stop watching after ingesting N scans (useful for bounded live runs or tests)",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=20,
        help="Stop live watcher after this many consecutive polls with no ready scans (0 = run indefinitely)",
    )
    return parser


def _load_optional_calibration(path_str: str | None) -> AxisCalibration | None:
    if not path_str:
        return None
    raw = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return AxisCalibration.from_dict(raw)


def _is_hdf5_quiescent_and_readable(path: Path, *, settle_sec: float = 0.03) -> bool:
    """Verify file size/mtime stability and SWMR HDF5 dataset readability before ingestion."""
    try:
        st0 = path.stat()
        if st0.st_size <= 0:
            return False
        if settle_sec > 0.0:
            time.sleep(settle_sec)
            st1 = path.stat()
            if st1.st_size != st0.st_size or st1.st_mtime_ns != st0.st_mtime_ns:
                return False
        h5py = _require_h5py()
        with _open_h5_readonly(h5py, path, swmr=True) as handle:
            ds_path = _resolve_volume_dataset(handle, h5py, auto_discover=True)
            ds = handle[ds_path]
            if ds.ndim != 3 or any(int(dim) <= 0 for dim in ds.shape):
                return False
            # Probe trailing frame voxel to catch incomplete LIMA chunk flushes
            _ = ds[-1, 0:1, 0:1]
        return True
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_dataset_root(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    calibration = _load_optional_calibration(args.calibration_json)
    detector = DetectorGeometry.from_json(args.detector_json) if args.detector_json else None
    prec_cfg = PrecursorConfig(
        score_threshold=float(args.min_precursor_score),
        min_persistence=int(args.min_persistence),
    )
    odd = OperationalDesignDomain(
        mu_bounds=(float(args.mu_bounds[0]), float(args.mu_bounds[1])) if args.mu_bounds else None,
        chi_bounds=(float(args.chi_bounds[0]), float(args.chi_bounds[1])) if args.chi_bounds else None,
        d_bounds=(float(args.d_bounds[0]), float(args.d_bounds[1])) if args.d_bounds else None,
        voltage_bounds_v=(float(args.voltage_bounds[0]), float(args.voltage_bounds[1]))
        if args.voltage_bounds
        else None,
        max_zooms_per_lineage=args.max_zooms_per_lineage,
        scan_cadence_sec=args.scan_cadence_sec,
    )
    policy = ZoomTriggerPolicy(
        min_precursor_score=float(args.min_precursor_score),
        min_persistence=int(args.min_persistence),
        cooldown_scans=int(args.cooldown_scans),
        max_triggers_per_scan=int(args.max_triggers_per_scan),
        lead_scans=float(args.lead_scans),
        window_sigma=float(args.window_sigma),
        hypothesis=args.hypothesis,
        execution_mode=args.mode,
        odd=odd,
    )

    triggers_jsonl = outdir / "zoom_triggers.jsonl"
    audit_jsonl = outdir / "audit_trajectory.jsonl"
    steps_json = outdir / "autonomous_steps.json"

    if args.mode == "replay":
        seq = BeamlineAdapter(root).build_sequence()
        if not seq.scans:
            raise FileNotFoundError(f"No valid scan folders discovered under {root}")
        scan_files = [sc.file_path for sc in seq.scans]
        scan_names = [sc.scan_name for sc in seq.scans]
        if args.max_scans is not None:
            scan_files = scan_files[: int(args.max_scans)]
            scan_names = scan_names[: int(args.max_scans)]

        stimuli = None
        if args.operando_log:
            op_log = load_operando_log(args.operando_log)
            stimuli = synchronize_to_scans(op_log, seq)[: len(scan_files)]

        bench = evaluate_causal_replay(
            scan_files,
            scan_names=scan_names,
            stimuli=stimuli,
            calibration=calibration,
            detector=detector,
            backend=args.backend,
            precursor_config=prec_cfg,
            trigger_policy=policy,
            zoom_budget=args.zoom_budget,
        )

        with triggers_jsonl.open("w", encoding="utf-8") as fh:
            for tgt in bench.zoom_targets:
                fh.write(json.dumps(tgt.to_bliss_command()) + "\n")

        with audit_jsonl.open("w", encoding="utf-8") as fh:
            for rec in bench.audit_records:
                fh.write(json.dumps(rec.to_dict()) + "\n")

        steps_json.write_text(json.dumps([s.to_dict() for s in bench.steps], indent=2), encoding="utf-8")
        summary_path = outdir / "replay_summary.json"
        summary_path.write_text(json.dumps(bench.to_dict(), indent=2), encoding="utf-8")

        bt_m = bench.policy_comparison["braggtrack_autonomous"]
        gap_msg = (
            f", mean Delta_gap={bench.mean_sim_to_real_gap:+.3f}" if bench.mean_sim_to_real_gap is not None else ""
        )
        print(
            f"[braggtrack-watch:replay] Processed {bench.n_scans} scans "
            f"(mean {bench.mean_step_latency_sec:.3f}s/scan) -> {len(bench.zoom_targets)} zoom trigger(s){gap_msg}"
        )
        print(
            f"  Fracture capture recall={bt_m.fracture_capture_recall:.2f}, "
            f"precision={bt_m.fracture_capture_precision:.2f}, "
            f"mean lead={bt_m.mean_lead_scans_before_split:.2f} scans, "
            f"next-scan hit rate={bt_m.setpoint_hit_rate_next_scan:.2f}"
        )
        return 0

    # Live directory polling modes: "shadow", "watch" (non-actuating), or "active" (FAL Level 3 actuation)
    tracker = CausalStreamingTracker(
        backend=args.backend,
        model_name=args.model,
        calibration=calibration,
        detector=detector,
        precursor_config=prec_cfg,
        trigger_policy=policy,
        sliding_window_scans=int(args.sliding_window) if args.sliding_window and args.sliding_window > 0 else None,
        swmr=True,
    )
    seen_files: set[Path] = set()
    step_dicts: list[dict] = []
    idle_polls = 0

    triggers_jsonl.write_text("", encoding="utf-8")
    audit_jsonl.write_text("", encoding="utf-8")
    while True:
        try:
            seq = BeamlineAdapter(root).build_sequence()
        except OSError:
            seq = None

        stim_by_name = {}
        if seq is not None and args.operando_log and seq.scans:
            try:
                op_log = load_operando_log(args.operando_log)
                for sp in synchronize_to_scans(op_log, seq):
                    stim_by_name[sp.scan_name] = sp
            except (OSError, ValueError):
                pass

        candidates = [sc for sc in (seq.scans if seq is not None else []) if sc.file_path.resolve() not in seen_files]
        ingested_any = False
        for sc in candidates:
            if not _is_hdf5_quiescent_and_readable(sc.file_path, settle_sec=max(0.0, float(args.settle_sec))):
                continue
            ts_iso = sc.start_time.isoformat() if sc.start_time is not None else None
            try:
                res = tracker.step(
                    sc.file_path,
                    scan_name=sc.scan_name,
                    timestamp=ts_iso,
                    stimulus=stim_by_name.get(sc.scan_name),
                )
            except (OSError, KeyError, ValueError):
                # Transient partial HDF5 write or lock — retry cleanly on next poll without losing the scan
                continue

            seen_files.add(sc.file_path.resolve())
            ingested_any = True
            step_dicts.append(res.to_dict())
            if res.zoom_targets:
                with triggers_jsonl.open("a", encoding="utf-8") as fh:
                    for tgt in res.zoom_targets:
                        fh.write(json.dumps(tgt.to_bliss_command()) + "\n")
                    fh.flush()
            if res.audit_records:
                with audit_jsonl.open("a", encoding="utf-8") as fh:
                    for rec in res.audit_records:
                        fh.write(json.dumps(rec.to_dict()) + "\n")
                    fh.flush()
            steps_json.write_text(json.dumps(step_dicts, indent=2), encoding="utf-8")
            print(
                f"[braggtrack-watch:{args.mode}] scan={res.scan_name} spots={res.n_spots} "
                f"fal={res.fal_level_active} zoom_triggers={len(res.zoom_targets)} ({res.step_latency_sec:.3f}s)"
            )
            if args.max_scans is not None and tracker.n_scans_processed >= int(args.max_scans):
                return 0

        if ingested_any:
            idle_polls = 0
        else:
            idle_polls += 1
            if args.max_idle_polls > 0 and idle_polls >= int(args.max_idle_polls):
                break
            time.sleep(max(0.01, float(args.poll_interval)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
