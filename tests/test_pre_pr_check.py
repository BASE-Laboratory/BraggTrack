import subprocess
import sys
import unittest


class PrePrCheckTests(unittest.TestCase):
    def test_pre_pr_check_runs(self) -> None:
        proc = subprocess.run(
            [sys.executable, 'scripts/pre_pr_check.py'],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn('pre_pr_check=PASS', proc.stdout)


if __name__ == '__main__':
    unittest.main()
