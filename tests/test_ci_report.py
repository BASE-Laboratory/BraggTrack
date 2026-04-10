import os
import subprocess
import sys
import unittest


class CiReportTests(unittest.TestCase):
    def test_ci_report_runs_and_reports_summary(self) -> None:
        env = os.environ.copy()
        env['PYTHONPATH'] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
        env['BRAGGTRACK_SKIP_TEST_MODULE'] = 'test_ci_report'
        proc = subprocess.run(
            [sys.executable, 'scripts/ci_report.py'],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn('=== Summary ===', proc.stdout)
        self.assertIn('overall=PASS', proc.stdout)


if __name__ == '__main__':
    unittest.main()
