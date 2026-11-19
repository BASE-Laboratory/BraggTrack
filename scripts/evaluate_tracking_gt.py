#!/usr/bin/env python3
"""CLI wrapper to score BraggTrack trajectory outputs against blind human ground-truth link annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from braggtrack.tracking.gt_eval import (
    extract_predicted_edges_from_graph,
    extract_predicted_edges_from_tables,
    score_graph_against_gt,
    score_predictions_against_gt,
)

__all__ = [
    "extract_predicted_edges_from_graph",
    "extract_predicted_edges_from_tables",
    "score_graph_against_gt",
    "score_predictions_against_gt",
    "main",
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score BraggTrack outputs against blind human ground-truth links.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground_truth_links.json")
    parser.add_argument("--tracks", type=Path, required=True, help="Path to predicted tracks.csv")
    parser.add_argument("--events", type=Path, default=None, help="Optional path to predicted events.csv")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args(argv)

    gt_payload = json.loads(args.gt.read_text(encoding="utf-8"))
    tracks_rows = _read_csv_rows(args.tracks)
    events_rows = _read_csv_rows(args.events) if args.events and args.events.exists() else None
    pred_1to1, pred_events = extract_predicted_edges_from_tables(tracks_rows, events_rows)
    summary = score_predictions_against_gt(pred_1to1, pred_events, gt_payload)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ov = summary["overall_1to1"]
    cr = summary["crowded_1to1"]
    iso = summary["isolated_1to1"]
    print(
        f"Overall 1-to-1: F1={ov['f1']:.4f} (P={ov['precision']:.4f}, R={ov['recall']:.4f}, "
        f"TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']})"
    )
    print(f"  Isolated: F1={iso['f1']:.4f} (TP={iso['tp']}, FP={iso['fp']}, FN={iso['fn']})")
    print(f"  Crowded : F1={cr['f1']:.4f} (TP={cr['tp']}, FP={cr['fp']}, FN={cr['fn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
