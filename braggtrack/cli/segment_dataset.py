"""Run classical segmentation over discovered scan files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from braggtrack.io import MissingH5DependencyError, discover_operando_scans, load_primary_volume
from braggtrack.segmentation import otsu_threshold, segment_classical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Dataset root containing scan folders")
    parser.add_argument("--blur-passes", type=int, default=1, help="Number of lightweight Gaussian blur passes")
    parser.add_argument("--seed-separation", type=int, default=1, help="Minimum seed separation in voxels")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scans = discover_operando_scans(Path(args.root))

    payload: list[dict[str, object]] = []
    errors = 0

    for scan in scans:
        item: dict[str, object] = {"scan": scan.scan_name, "file": str(scan.path)}
        try:
            volume = load_primary_volume(scan.path)
            flat = [v for plane in volume for row in plane for v in row]
            threshold = otsu_threshold(flat)
            result = segment_classical(
                volume,
                threshold=threshold,
                blur_passes=max(1, args.blur_passes),
                min_seed_separation=max(1, args.seed_separation),
            )
            item["threshold"] = result.threshold
            item["seed_count"] = result.seed_count
            item["component_count"] = result.component_count
        except (MissingH5DependencyError, KeyError, ValueError) as exc:
            item["error"] = str(exc)
            errors += 1

        payload.append(item)

    print(json.dumps(payload, indent=2))
    return 1 if errors == len(payload) and payload else 0


if __name__ == "__main__":
    raise SystemExit(main())
