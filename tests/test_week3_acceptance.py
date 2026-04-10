import json
import subprocess
import sys
import unittest


class Week3AcceptanceTests(unittest.TestCase):
    def test_week3_acceptance_script(self) -> None:
        proc = subprocess.run(
            [sys.executable, 'scripts/check_week3_acceptance.py'],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload['n_scans'], 3)
        self.assertGreater(payload['total_tracks'], 0)
        self.assertTrue(payload['metrics_present'])
        self.assertEqual(payload['failures'], [])


if __name__ == '__main__':
    unittest.main()
