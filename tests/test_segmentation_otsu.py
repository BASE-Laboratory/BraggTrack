import unittest

import numpy as np

from braggtrack.segmentation import connected_components_3d, otsu_threshold, segment_volume


class OtsuSegmentationTests(unittest.TestCase):
    def test_otsu_threshold_separates_bimodal_signal(self) -> None:
        values = [1.0] * 100 + [20.0] * 100
        threshold = otsu_threshold(values)
        self.assertGreater(threshold, 1.0)
        self.assertLess(threshold, 20.0)

    def test_segment_volume_detects_two_components(self) -> None:
        size = 8
        volume = np.ones((size, size, size))

        # blob 1
        volume[1, 1, 1] = 20.0
        volume[1, 1, 2] = 20.0
        volume[1, 2, 1] = 20.0

        # blob 2 (far away)
        volume[6, 6, 6] = 20.0
        volume[6, 6, 5] = 20.0
        volume[5, 6, 6] = 20.0

        result = segment_volume(volume, method='otsu')
        self.assertEqual(result.component_count, 2)
        self.assertGreater(result.voxel_count, 0)

    def test_connected_components_empty(self) -> None:
        self.assertEqual(connected_components_3d(np.array([[[False]]])), 0)


if __name__ == '__main__':
    unittest.main()
