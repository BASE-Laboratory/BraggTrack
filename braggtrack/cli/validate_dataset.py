"""Validate BraggTrack dataset contracts for operando scan sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from braggtrack.io import BeamlineAdapter, resolve_dataset_root, validate_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root with scan folders (default: data/sample_operando if present, else .)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_dataset_root(args.root)
    adapter = BeamlineAdapter(root)
    sequence = adapter.build_sequence()
    issues = validate_sequence(sequence)

    payload = {
        "root": str(root.resolve()),
        "scan_count": len(sequence.scans),
        "is_monotonic": sequence.is_monotonic(),
        "scans": [
            {
                "scan": scan.scan_name,
                "index": scan.sequence_index,
                "file": str(scan.file_path),
                "start_time": scan.start_time.isoformat() if scan.start_time else None,
                "end_time": scan.end_time.isoformat() if scan.end_time else None,
                "sample_name": scan.sample_name,
                "title": scan.title,
                "extras": scan.extras,
            }
            for scan in sequence.scans
        ],
        "issues": [issue.__dict__ for issue in issues],
    }

    print(json.dumps(payload, indent=2))
    return 1 if any(issue.level == "error" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
