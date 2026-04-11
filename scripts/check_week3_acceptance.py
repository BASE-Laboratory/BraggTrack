"""Week 3 acceptance checks for tracking artifacts.

Acceptance criteria:
  1. Tracks are generated across all three scans.
  2. ID-switch rate and track fragmentation are computed.
  3. Tracking metrics and trajectory table are produced.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braggtrack.io import resolve_dataset_root

OUTDIR = Path("artifacts/week3")
DATASET_ROOT = resolve_dataset_root(None)


def main() -> int:
    # Step 1 — ensure segmentation artifacts exist (run Week 2 if needed).
    seg_dir = Path("artifacts/week2")
    if not (seg_dir / "segmentation_summary.json").exists():
        subprocess.run(
            [sys.executable, "-m", "braggtrack.cli.segment_dataset", str(DATASET_ROOT), "--outdir", str(seg_dir)],
            check=False, capture_output=True, text=True,
        )

    # Step 2 — run tracking.
    proc = subprocess.run(
        [sys.executable, "-m", "braggtrack.cli.track_dataset", str(seg_dir), "--outdir", str(OUTDIR)],
        check=False, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}

    failures: list[str] = []

    # Check tracking ran successfully.
    if proc.returncode != 0:
        failures.append(f"track_dataset exit code {proc.returncode}: {proc.stderr.strip()}")

    # Check scans processed.
    n_scans = payload.get("n_scans", 0)
    if n_scans != 3:
        failures.append(f"Expected 3 scans, got {n_scans}")

    # Check tracks exist.
    total_tracks = payload.get("total_tracks", 0)
    if total_tracks <= 0:
        failures.append(f"total_tracks must be > 0, got {total_tracks}")

    # Check metrics are present.
    for key in ("fragmentation_ratio", "id_switch_rate", "born_count",
                "continued_count", "terminated_count"):
        if key not in payload:
            failures.append(f"Missing metric: {key}")

    # Check artifact files.
    for fname in ("tracks.csv", "tracking_metrics.json", "tracking_summary.json"):
        if not (OUTDIR / fname).exists():
            failures.append(f"Missing artifact: {fname}")

    # Check schema version.
    if payload.get("schema_version") != "week3.v1":
        failures.append(f"schema_version mismatch: {payload.get('schema_version')}")

    report = {
        "n_scans": n_scans,
        "total_tracks": total_tracks,
        "metrics_present": all(k in payload for k in ("fragmentation_ratio", "id_switch_rate")),
        "schema_consistent": payload.get("schema_version") == "week3.v1",
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
