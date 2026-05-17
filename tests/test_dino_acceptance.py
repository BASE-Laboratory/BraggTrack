"""Acceptance test: DINO segmentation pipeline on bundled sample data."""

import json
import subprocess
import sys
import unittest


class DinoAcceptanceTests(unittest.TestCase):
    def test_dino_acceptance_script(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/check_dino_acceptance.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["method"], "dino")
        self.assertEqual(payload["scan_count"], 3)
        self.assertEqual(payload["non_empty_components"], 3)
        self.assertEqual(payload["failures"], [])


if __name__ == "__main__":
    unittest.main()
