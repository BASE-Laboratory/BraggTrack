import unittest
from pathlib import Path

from braggtrack.io import BeamlineAdapter, validate_sequence
from braggtrack.io.models import ExperimentSequence, ScanVolumeMeta


class Week1ValidationTests(unittest.TestCase):
    def test_beamline_adapter_builds_three_scan_sequence(self) -> None:
        adapter = BeamlineAdapter(Path('.'))
        sequence = adapter.build_sequence()
        self.assertEqual(len(sequence.scans), 3)
        self.assertTrue(sequence.is_monotonic())
        self.assertEqual([scan.sequence_index for scan in sequence.scans], [1, 2, 3])

    def test_validate_sequence_flags_missing_metadata(self) -> None:
        seq = ExperimentSequence(
            scans=(
                ScanVolumeMeta(scan_name='scan0001', file_path=Path('a.h5'), sequence_index=1),
                ScanVolumeMeta(scan_name='scan0002', file_path=Path('b.h5'), sequence_index=2),
            )
        )
        issues = validate_sequence(seq)
        self.assertTrue(any(issue.code == 'missing_metadata' for issue in issues))


if __name__ == '__main__':
    unittest.main()
