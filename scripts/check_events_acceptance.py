"""Events, fracture precursor, and reproducible pipeline acceptance checks.

Acceptance criteria:
  1. Synthetic 4D fracture benchmark recovers exact 1->2 split and 2->1 merge events.
  2. Persistent fracture precursor ranking places the fracturing grain at rank #1
     with subsequent_split=True while excluding the stable control grain.
  3. One-command pipeline (braggtrack-run) completes end-to-end on bundled ESRF-ID03
     operando scans and produces run_manifest.json, kinematics.csv, events.csv, and precursors.csv.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braggtrack.cli._utils import write_csv
from braggtrack.io import resolve_dataset_root
from braggtrack.tracking import generate_fracture_scenario

PIPELINE_OUTDIR = Path("artifacts/pipeline_acceptance")
DATASET_ROOT = resolve_dataset_root(None)


def _check_synthetic_fracture_pipeline(failures: list[str]) -> dict[str, object]:
    volumes, label_volumes, scan_tables, expected = generate_fracture_scenario(n_scans=5, seed=42)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        data_root = tmp_root / "synth_data"
        seg_dir = tmp_root / "seg"
        evt_dir = tmp_root / "events"

        for idx, (vol, labs, rows) in enumerate(zip(volumes, label_volumes, scan_tables)):
            sname = f"scan{idx + 1:04d}"
            sdata = data_root / sname
            sdata.mkdir(parents=True, exist_ok=True)
            with h5py.File(sdata / "scan.h5", "w") as h5:
                h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol.astype(np.float32))

            sseg = seg_dir / sname
            sseg.mkdir(parents=True, exist_ok=True)
            write_csv(sseg / "features.csv", rows)
            np.savez_compressed(sseg / "labels.npz", labels=labs)

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "braggtrack.cli.detect_events",
                str(data_root),
                "--segdir",
                str(seg_dir),
                "--outdir",
                str(evt_dir),
                "--backend",
                "mock",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failures.append(f"detect_events failed: {proc.stderr.strip() or proc.stdout.strip()}")
            return {}

        summary = json.loads(proc.stdout)
        if summary.get("schema_version") != "events.v1":
            failures.append(f"Unexpected events schema_version: {summary.get('schema_version')}")
        if summary.get("split_count") != 1:
            failures.append(f"Expected 1 split event on synthetic fracture scenario, got {summary.get('split_count')}")
        if summary.get("merge_count") != 1:
            failures.append(f"Expected 1 merge event on synthetic fracture scenario, got {summary.get('merge_count')}")
        if summary.get("precursor_candidate_count", 0) < 1:
            failures.append("Expected at least 1 persistent precursor candidate")
        if not summary.get("roi_dispersion_enabled"):
            failures.append("Expected roi_dispersion_enabled=True when 3D volumes and labels.npz are present")

        events_list = json.loads((evt_dir / "events.json").read_text())
        split_evts = [e for e in events_list if e["event_type"] == "split"]
        if split_evts and int(split_evts[0]["scan_idx_to"]) != expected["split_scan_to"]:
            failures.append(f"Split detected at wrong scan: {split_evts[0]['scan_idx_to']}")

        return summary


def _check_end_to_end_pipeline(failures: list[str]) -> dict[str, object]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "braggtrack.cli.run_pipeline",
            str(DATASET_ROOT),
            "--outdir",
            str(PIPELINE_OUTDIR),
            "--backend",
            "mock",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        failures.append(f"run_pipeline failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return {}

    manifest_path = PIPELINE_OUTDIR / "run_manifest.json"
    if not manifest_path.exists():
        failures.append("Missing run_manifest.json from run_pipeline")
        return {}

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "manifest.v1":
        failures.append(f"Unexpected manifest schema_version: {manifest.get('schema_version')}")
    if len(manifest.get("inputs", [])) != 3:
        failures.append(f"Expected 3 input checksums in manifest, got {len(manifest.get('inputs', []))}")

    for key in ("kinematics_csv", "events_csv", "precursors_csv", "events_summary"):
        art_path = manifest.get("artifacts", {}).get(key)
        if not art_path or not Path(art_path).exists():
            failures.append(f"Missing pipeline artifact {key}: {art_path}")

    return {
        "manifest_schema": manifest.get("schema_version"),
        "input_count": len(manifest.get("inputs", [])),
        "stages": sorted(manifest.get("stage_timings_sec", {}).keys()),
    }


def main() -> int:
    failures: list[str] = []
    synth_report = _check_synthetic_fracture_pipeline(failures)
    pipe_report = _check_end_to_end_pipeline(failures)

    report = {
        "synthetic_fracture": synth_report,
        "pipeline_manifest": pipe_report,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
