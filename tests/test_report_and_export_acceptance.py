from __future__ import annotations

import json
import subprocess
import sys
import unittest


class ReportAndExportAcceptanceTests(unittest.TestCase):
    def test_report_and_export_acceptance_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/check_report_and_export_acceptance.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["failures"], [])
        self.assertEqual(payload["gap_closing"]["gap_closed_count"], 1)
        self.assertGreater(payload["report_and_exports"]["html_report_bytes"], 5_000)
        self.assertGreater(payload["report_and_exports"]["flt_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
