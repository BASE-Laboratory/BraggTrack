import json
import subprocess
import sys
import unittest


class SegmentDatasetCliTests(unittest.TestCase):
    def test_segment_dataset_reports_per_scan_errors_without_h5py(self) -> None:
        proc = subprocess.run(
            [sys.executable, '-m', 'braggtrack.cli.segment_dataset', '.'],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 1)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload), 3)
        self.assertTrue(all('error' in item for item in payload))


if __name__ == '__main__':
    unittest.main()
