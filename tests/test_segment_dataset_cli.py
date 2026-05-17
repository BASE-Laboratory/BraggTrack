"""CLI integration test for segment_dataset with proper temp dir cleanup."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from braggtrack.io import sample_operando_root


class SegmentDatasetCliTests(unittest.TestCase):
    def test_segment_dataset_writes_week2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "week2"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "braggtrack.cli.segment_dataset",
                    str(sample_operando_root()),
                    "--outdir",
                    str(outdir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(len(payload), 3)
            self.assertTrue(all(item["component_count"] > 0 for item in payload))
            self.assertTrue((outdir / "segmentation_summary.csv").exists())
            self.assertTrue((outdir / "segmentation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
