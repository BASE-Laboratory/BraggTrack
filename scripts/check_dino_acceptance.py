"""DINO segmentation acceptance checks on bundled sample data."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braggtrack.io import resolve_dataset_root

OUTDIR = Path("artifacts/dino")
DATASET_ROOT = resolve_dataset_root(None)


def main() -> int:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "braggtrack.cli.segment_dataset",
            str(DATASET_ROOT),
            "--outdir",
            str(OUTDIR),
            "--method",
            "dino",
            "--dino-backend",
            "mock",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else []

    failures: list[str] = []

    if proc.returncode != 0:
        failures.append(f"segment_dataset exited with code {proc.returncode}: {proc.stderr.strip()}")

    if len(payload) != 3:
        failures.append(f"Expected 3 scans in output, found {len(payload)}")

    for row in payload:
        scan = row.get("scan", "?")
        if row.get("component_count", 0) <= 0:
            failures.append(f"{scan}: component_count must be > 0")
        if row.get("schema_version") != "segmentation.v1":
            failures.append(f"{scan}: schema_version mismatch")

    summary_csv = OUTDIR / "segmentation_summary.csv"
    if not summary_csv.exists():
        failures.append("Missing segmentation_summary.csv")
    else:
        with summary_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 3:
            failures.append(f"segmentation_summary.csv expected 3 rows, found {len(rows)}")

    for scan_dir in sorted(OUTDIR.glob("scan*")):
        summary_path = scan_dir / "summary.json"
        if not summary_path.exists():
            failures.append(f"{scan_dir.name}: missing summary.json")
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("method") != "dino":
            failures.append(f"{scan_dir.name}: method should be 'dino', got {summary.get('method')}")
        if not (scan_dir / "features.csv").exists():
            failures.append(f"{scan_dir.name}: missing features.csv")
        if not (scan_dir / "labels.npz").exists():
            failures.append(f"{scan_dir.name}: missing labels.npz")

    report = {
        "method": "dino",
        "scan_count": len(payload),
        "non_empty_components": sum(1 for r in payload if r.get("component_count", 0) > 0),
        "schema_consistent": all(r.get("schema_version") == "segmentation.v1" for r in payload),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
