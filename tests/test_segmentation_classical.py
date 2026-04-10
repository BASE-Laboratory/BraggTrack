import unittest

import numpy as np

from braggtrack.segmentation import h_maxima_seeds, local_maxima_seeds, log_enhance_3d, segment_classical


class ClassicalSegmentationTests(unittest.TestCase):
    def test_log_enhance_preserves_shape(self) -> None:
        volume = np.ones((4, 4, 4))
        out = log_enhance_3d(volume, blur_passes=1)
        self.assertEqual(out.shape, (4, 4, 4))

    def test_h_maxima_detects_peak(self) -> None:
        volume = np.zeros((5, 5, 5))
        volume[2, 2, 2] = 10.0
        seeds = h_maxima_seeds(volume, min_value=5.0, h=0.1, min_separation=1)
        self.assertIn((2, 2, 2), seeds)

    def test_local_maxima_detects_peak(self) -> None:
        volume = np.zeros((5, 5, 5))
        volume[2, 2, 2] = 10.0
        seeds = local_maxima_seeds(volume, min_value=5.0, min_separation=1)
        self.assertIn((2, 2, 2), seeds)

    def test_segment_classical_two_blobs(self) -> None:
        size = 10
        volume = np.ones((size, size, size))
        for z, y, x in [(2, 2, 2), (2, 2, 3), (7, 7, 7), (7, 7, 6)]:
            volume[z, y, x] = 25.0
        result = segment_classical(volume, threshold=0.01, blur_passes=1, h_value=0.0, min_seed_separation=1)
        self.assertGreaterEqual(result.seed_count, 2)
        self.assertGreaterEqual(result.component_count, 2)


if __name__ == '__main__':
    unittest.main()
