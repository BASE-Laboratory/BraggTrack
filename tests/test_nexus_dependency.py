import unittest

from braggtrack.io.nexus import MissingH5DependencyError, extract_scan_metadata


class NexusDependencyTests(unittest.TestCase):
    def test_extract_scan_metadata_requires_h5py(self) -> None:
        with self.assertRaises(MissingH5DependencyError):
            extract_scan_metadata('scan0001/pco_nf_0000_cropped.h5')


if __name__ == '__main__':
    unittest.main()
