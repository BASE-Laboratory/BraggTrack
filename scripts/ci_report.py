"""Human-readable CI report for BraggTrack checks.

Runs unit tests and key acceptance/smoke commands with explicit pass/fail criteria.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


def _filter_suite(suite: unittest.TestSuite, skip_module: str | None) -> unittest.TestSuite:
    if skip_module is None:
        return suite
    filtered = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            nested = _filter_suite(test, skip_module)
            if nested.countTestCases() > 0:
                filtered.addTest(nested)
        else:
            test_id = getattr(test, "id", lambda: "")()
            if skip_module not in test_id:
                filtered.addTest(test)
    return filtered


def run_unit_tests() -> bool:
    print("\n=== Unit Tests ===")
    suite = unittest.defaultTestLoader.discover("tests")
    skip_module = os.environ.get("BRAGGTRACK_SKIP_TEST_MODULE")
    suite = _filter_suite(suite, skip_module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"unit_tests: ran={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}")
    return result.wasSuccessful()


def run_cmd_json(cmd: list[str], label: str) -> tuple[bool, dict | list | None, str]:
    print(f"\n=== {label} ===")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())

    payload = None
    import contextlib

    with contextlib.suppress(json.JSONDecodeError):
        payload = json.loads(proc.stdout) if proc.stdout.strip() else None

    ok = proc.returncode == 0
    print(f"{label.lower().replace(' ', '_')}: returncode={proc.returncode}")
    return ok, payload, proc.stdout


def evaluate_acceptance(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        print("acceptance_evaluation: FAIL (no JSON payload)")
        return False

    checks = {
        "scan_count==3": payload.get("scan_count") == 3,
        "is_monotonic": payload.get("is_monotonic") is True,
        "error_count==0": payload.get("error_count") == 0,
        "no_failures": payload.get("failures") == [],
    }
    for name, passed in checks.items():
        print(f" - {name}: {'PASS' if passed else 'FAIL'}")
    overall = all(checks.values())
    print(f"acceptance_evaluation: {'PASS' if overall else 'FAIL'}")
    return overall


def evaluate_smoke(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        print("smoke_evaluation: FAIL (no JSON payload)")
        return False
    checks = {
        "method==otsu": payload.get("method") == "otsu",
        "component_count>=1": isinstance(payload.get("component_count"), int)
        and payload.get("component_count", 0) >= 1,
        "voxel_count>=1": isinstance(payload.get("voxel_count"), int) and payload.get("voxel_count", 0) >= 1,
    }
    for name, passed in checks.items():
        print(f" - {name}: {'PASS' if passed else 'FAIL'}")
    overall = all(checks.values())
    print(f"smoke_evaluation: {'PASS' if overall else 'FAIL'}")
    return overall


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    unit_ok = run_unit_tests()

    acc_ok_cmd, acc_payload, _ = run_cmd_json([sys.executable, "scripts/check_io_acceptance.py"], "IO Acceptance")
    acc_ok = acc_ok_cmd and evaluate_acceptance(acc_payload if isinstance(acc_payload, dict) else None)

    smoke_ok_cmd, smoke_payload, _ = run_cmd_json(
        [sys.executable, "-m", "braggtrack.cli.segment_synthetic"], "Segmentation Smoke"
    )
    smoke_ok = smoke_ok_cmd and evaluate_smoke(smoke_payload if isinstance(smoke_payload, dict) else None)

    seg_ok_cmd, seg_payload, _ = run_cmd_json(
        [sys.executable, "scripts/check_segmentation_acceptance.py"], "Segmentation Acceptance"
    )
    seg_ok = seg_ok_cmd and isinstance(seg_payload, dict) and seg_payload.get("failures") == []

    trk_ok_cmd, trk_payload, _ = run_cmd_json(
        [sys.executable, "scripts/check_tracking_acceptance.py"], "Tracking Acceptance"
    )
    trk_ok = trk_ok_cmd and isinstance(trk_payload, dict) and trk_payload.get("failures") == []

    emb_ok_cmd, emb_payload, _ = run_cmd_json(
        [sys.executable, "scripts/check_embedding_acceptance.py"], "Embedding Acceptance"
    )
    emb_ok = emb_ok_cmd and isinstance(emb_payload, dict) and emb_payload.get("failures") == []

    dino_ok_cmd, dino_payload, _ = run_cmd_json([sys.executable, "scripts/check_dino_acceptance.py"], "DINO Acceptance")
    dino_ok = dino_ok_cmd and isinstance(dino_payload, dict) and dino_payload.get("failures") == []

    all_ok = unit_ok and acc_ok and smoke_ok and seg_ok and trk_ok and emb_ok and dino_ok
    print("\n=== Summary ===")
    print(f"unit_tests={'PASS' if unit_ok else 'FAIL'}")
    print(f"io_acceptance={'PASS' if acc_ok else 'FAIL'}")
    print(f"smoke={'PASS' if smoke_ok else 'FAIL'}")
    print(f"segmentation_acceptance={'PASS' if seg_ok else 'FAIL'}")
    print(f"tracking_acceptance={'PASS' if trk_ok else 'FAIL'}")
    print(f"embedding_acceptance={'PASS' if emb_ok else 'FAIL'}")
    print(f"dino_acceptance={'PASS' if dino_ok else 'FAIL'}")
    print(f"overall={'PASS' if all_ok else 'FAIL'}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
