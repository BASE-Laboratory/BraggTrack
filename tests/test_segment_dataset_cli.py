import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class SegmentDatasetCliTests(unittest.TestCase):
    def test_segment_dataset_writes_week2_artifacts(self) -> None:
        outdir = Path('artifacts/test_week2_cli')
        if outdir.exists():
            shutil.rmtree(outdir)

        proc = subprocess.run(
            [sys.executable, '-m', 'braggtrack.cli.segment_dataset', '.', '--outdir', str(outdir)],
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
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload), 3)
        self.assertTrue(all(item['component_count'] > 0 for item in payload))
        self.assertTrue((outdir / 'segmentation_summary.csv').exists())
        self.assertEqual(proc.returncode, 1)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload), 3)
        self.assertTrue(all('error' in item for item in payload))


if __name__ == '__main__':
    unittest.main()
