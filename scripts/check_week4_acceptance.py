"""Week 4 acceptance: segmentation labels, mock embeddings, semantic tracking."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

OUT_EMB = Path("artifacts/week4")
OUT_TRACK = Path("artifacts/week4_track")
SEG = Path("artifacts/week2")


def main() -> int:
    seg_dir = SEG
    need_seg = not (seg_dir / "segmentation_summary.json").exists()
    if seg_dir.is_dir():
        for sub in seg_dir.iterdir():
            if sub.is_dir() and sub.name.startswith("scan") and not (sub / "labels.npz").exists():
                need_seg = True
                break
    if need_seg:
        subprocess.run(
            [sys.executable, "-m", "braggtrack.cli.segment_dataset", ".", "--outdir", str(seg_dir)],
            check=False,
            capture_output=True,
            text=True,
        )

    env = {**os.environ, "BRAGGTRACK_DINO_BACKEND": "mock"}
    proc_e = subprocess.run(
        [
            sys.executable,
            "-m",
            "braggtrack.cli.embed_dataset",
            ".",
            "--segdir",
            str(seg_dir),
            "--outdir",
            str(OUT_EMB),
            "--backend",
            "mock",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc_e.returncode != 0:
        print(json.dumps({"failures": [f"embed_dataset failed: {proc_e.stderr.strip()}"], "stdout": proc_e.stdout}))
        return 1

    proc_t = subprocess.run(
        [
            sys.executable,
            "-m",
            "braggtrack.cli.track_dataset",
            str(seg_dir),
            "--outdir",
            str(OUT_TRACK),
            "--embedding-dir",
            str(OUT_EMB),
            "--cost-beta",
            "0.25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc_t.stdout) if proc_t.stdout.strip() else {}

    failures: list[str] = []
    if proc_t.returncode != 0:
        failures.append(f"track_dataset exit {proc_t.returncode}: {proc_t.stderr.strip()}")

    if payload.get("n_scans") != 3:
        failures.append(f"Expected 3 scans, got {payload.get('n_scans')}")

    if payload.get("schema_version") != "week4.v1":
        failures.append(f"schema_version expected week4.v1, got {payload.get('schema_version')}")

    for fname in ("tracks.csv", "tracking_metrics.json", "tracking_summary.json"):
        if not (OUT_TRACK / fname).exists():
            failures.append(f"Missing {fname}")

    for scan in ("scan0001", "scan0002", "scan0003"):
        if not (OUT_EMB / scan / "embeddings.npz").exists():
            failures.append(f"Missing embeddings for {scan}")

    report = {"failures": failures, "n_scans": payload.get("n_scans"), "schema": payload.get("schema_version")}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
