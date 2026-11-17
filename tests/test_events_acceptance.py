import json
import subprocess
import sys
import unittest


class TestEventsAcceptance(unittest.TestCase):
    def test_events_acceptance_script(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/check_events_acceptance.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["failures"], [])
        self.assertEqual(payload["synthetic_fracture"]["split_count"], 1)
        self.assertEqual(payload["synthetic_fracture"]["merge_count"], 1)
        self.assertEqual(payload["pipeline_manifest"]["manifest_schema"], "manifest.v1")


if __name__ == "__main__":
    unittest.main()
