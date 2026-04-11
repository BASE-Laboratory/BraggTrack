import unittest
from unittest import mock

from braggtrack.io import sample_operando_root
from braggtrack.io.nexus import MissingH5DependencyError, extract_scan_metadata


class NexusDependencyTests(unittest.TestCase):
    @mock.patch(
        "braggtrack.io.nexus._require_h5py",
        side_effect=MissingH5DependencyError("h5py unavailable"),
    )
    def test_extract_scan_metadata_requires_h5py(self, _m: object) -> None:
        with self.assertRaises(MissingH5DependencyError):
            extract_scan_metadata(
                sample_operando_root() / "scan0001" / "pco_nf_0000_cropped.h5"
            )


if __name__ == '__main__':
    unittest.main()
