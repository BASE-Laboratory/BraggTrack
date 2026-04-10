import unittest
from pathlib import Path

from braggtrack.io import discover_operando_scans


class DiscoveryTests(unittest.TestCase):
    def test_discover_operando_scans_repo_samples(self) -> None:
        scans = discover_operando_scans(Path('.'))
        names = [item.scan_name for item in scans]
        self.assertEqual(names, ['scan0001', 'scan0002', 'scan0003'])
        self.assertTrue(all(item.path.suffix == '.h5' for item in scans))


if __name__ == '__main__':
    unittest.main()
