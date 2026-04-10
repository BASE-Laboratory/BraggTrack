"""Week 2 acceptance checks for segmentation artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


OUTDIR = Path("artifacts/week2")


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "braggtrack.cli.segment_dataset", ".", "--outdir", str(OUTDIR)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else []

    failures: list[str] = []

    if len(payload) != 3:
        failures.append(f"Expected 3 scans in output, found {len(payload)}")

    for row in payload:
        if row.get("component_count", 0) <= 0:
            failures.append(f"{row.get('scan')}: component_count must be > 0")
        if row.get("schema_version") != "week2.v1":
            failures.append(f"{row.get('scan')}: schema_version mismatch")

    summary_csv = OUTDIR / "segmentation_summary.csv"
    if not summary_csv.exists():
        failures.append("Missing segmentation_summary.csv")
    else:
        with summary_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 3:
            failures.append(f"segmentation_summary.csv expected 3 rows, found {len(rows)}")

    report = {
        "scan_count": len(payload),
        "non_empty_components": sum(1 for r in payload if r.get("component_count", 0) > 0),
        "schema_consistent": all(r.get("schema_version") == "week2.v1" for r in payload),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
