import unittest

from braggtrack.io import discover_operando_scans, sample_operando_root


class DiscoveryTests(unittest.TestCase):
    def test_discover_operando_scans_repo_samples(self) -> None:
        scans = discover_operando_scans(sample_operando_root())
        names = [item.scan_name for item in scans]
        self.assertEqual(names, ['scan0001', 'scan0002', 'scan0003'])
        self.assertTrue(all(item.path.suffix == '.h5' for item in scans))


if __name__ == '__main__':
    unittest.main()
