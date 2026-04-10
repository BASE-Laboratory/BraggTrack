import os
import subprocess
import sys
import unittest


class AcceptanceScriptTests(unittest.TestCase):
    def test_acceptance_script_runs_successfully(self) -> None:
        env = os.environ.copy()
        env['PYTHONPATH'] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(
            [sys.executable, 'scripts/check_acceptance.py'],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn('"is_monotonic": true', proc.stdout)


if __name__ == '__main__':
    unittest.main()
