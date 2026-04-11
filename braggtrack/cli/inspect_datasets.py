"""Inspect local operando scan datasets and summarize metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from braggtrack.io import (
    MissingH5DependencyError,
    discover_operando_scans,
    extract_scan_metadata,
    resolve_dataset_root,
)


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
    scans = discover_operando_scans(resolve_dataset_root(args.root))

    if not scans:
        print("No scan directories found.")
        return 1

    payload = []
    for scan in scans:
        item = {"scan": scan.scan_name, "file": str(scan.path)}
        try:
            item["metadata"] = extract_scan_metadata(scan.path)
        except MissingH5DependencyError as exc:
            item["metadata_error"] = str(exc)
        payload.append(item)

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
