"""Autonomous DFXM "Scout-and-Zoom" beamline watcher and causal replay CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from braggtrack.autonomous import (
    CausalStreamingTracker,
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
        choices=("replay", "watch"),
        default="replay",
        help="'replay' benchmarks causal closed-loop triggers on an existing sequence; 'watch' polls for new scans",
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
        "--zoom-budget",
        type=int,
        default=None,
        help="Optional cap on total zoom acquisitions during --mode replay baseline comparison",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Directory poll interval (s) in --mode watch")
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
        help="Stop --mode watch after this many consecutive polls with no new scans (0 = run indefinitely)",
    )
    return parser


def _load_optional_calibration(path_str: str | None) -> AxisCalibration | None:
    if not path_str:
        return None
    raw = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return AxisCalibration.from_dict(raw)


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
    policy = ZoomTriggerPolicy(
        min_precursor_score=float(args.min_precursor_score),
        min_persistence=int(args.min_persistence),
        cooldown_scans=int(args.cooldown_scans),
        max_triggers_per_scan=int(args.max_triggers_per_scan),
        lead_scans=float(args.lead_scans),
        window_sigma=float(args.window_sigma),
    )

    triggers_jsonl = outdir / "zoom_triggers.jsonl"
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

        steps_json.write_text(json.dumps([s.to_dict() for s in bench.steps], indent=2), encoding="utf-8")
        summary_path = outdir / "replay_summary.json"
        summary_path.write_text(json.dumps(bench.to_dict(), indent=2), encoding="utf-8")

        bt_m = bench.policy_comparison["braggtrack_autonomous"]
        print(
            f"[braggtrack-watch:replay] Processed {bench.n_scans} scans "
            f"(mean {bench.mean_step_latency_sec:.3f}s/scan) -> {len(bench.zoom_targets)} zoom trigger(s)"
        )
        print(
            f"  Fracture capture recall={bt_m.fracture_capture_recall:.2f}, "
            f"precision={bt_m.fracture_capture_precision:.2f}, "
            f"mean lead={bt_m.mean_lead_scans_before_split:.2f} scans, "
            f"next-scan hit rate={bt_m.setpoint_hit_rate_next_scan:.2f}"
        )
        return 0

    # Live directory polling mode ("watch")
    tracker = CausalStreamingTracker(
        backend=args.backend,
        model_name=args.model,
        calibration=calibration,
        detector=detector,
        precursor_config=prec_cfg,
        trigger_policy=policy,
    )
    seen_files: set[Path] = set()
    step_dicts: list[dict] = []
    idle_polls = 0

    triggers_jsonl.write_text("", encoding="utf-8")
    while True:
        seq = BeamlineAdapter(root).build_sequence()
        stim_by_name = {}
        if args.operando_log and seq.scans:
            op_log = load_operando_log(args.operando_log)
            for sp in synchronize_to_scans(op_log, seq):
                stim_by_name[sp.scan_name] = sp

        new_scans = [sc for sc in seq.scans if sc.file_path.resolve() not in seen_files]
        if new_scans:
            idle_polls = 0
            for sc in new_scans:
                seen_files.add(sc.file_path.resolve())
                ts_iso = sc.start_time.isoformat() if sc.start_time is not None else None
                res = tracker.step(
                    sc.file_path,
                    scan_name=sc.scan_name,
                    timestamp=ts_iso,
                    stimulus=stim_by_name.get(sc.scan_name),
                )
                step_dicts.append(res.to_dict())
                if res.zoom_targets:
                    with triggers_jsonl.open("a", encoding="utf-8") as fh:
                        for tgt in res.zoom_targets:
                            fh.write(json.dumps(tgt.to_bliss_command()) + "\n")
                        fh.flush()
                steps_json.write_text(json.dumps(step_dicts, indent=2), encoding="utf-8")
                print(
                    f"[braggtrack-watch:live] scan={res.scan_name} spots={res.n_spots} "
                    f"zoom_triggers={len(res.zoom_targets)} ({res.step_latency_sec:.3f}s)"
                )
                if args.max_scans is not None and tracker.n_scans_processed >= int(args.max_scans):
                    return 0
        else:
            idle_polls += 1
            if args.max_idle_polls > 0 and idle_polls >= int(args.max_idle_polls):
                break
            time.sleep(max(0.01, float(args.poll_interval)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
