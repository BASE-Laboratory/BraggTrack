import unittest

from braggtrack.segmentation import h_maxima_seeds, log_enhance_3d, segment_classical


class ClassicalSegmentationTests(unittest.TestCase):
    def test_log_enhance_preserves_shape(self) -> None:
        volume = [[[1.0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        out = log_enhance_3d(volume, blur_passes=1)
        self.assertEqual(len(out), 4)
        self.assertEqual(len(out[0]), 4)
        self.assertEqual(len(out[0][0]), 4)

    def test_h_maxima_detects_peak(self) -> None:
        volume = [[[0.0 for _ in range(5)] for _ in range(5)] for _ in range(5)]
        volume[2][2][2] = 10.0
        seeds = h_maxima_seeds(volume, min_value=5.0, h=0.1, min_separation=1)
        self.assertIn((2, 2, 2), seeds)

    def test_segment_classical_two_blobs(self) -> None:
        size = 10
        volume = [[[1.0 for _ in range(size)] for _ in range(size)] for _ in range(size)]
        for z, y, x in [(2, 2, 2), (2, 2, 3), (7, 7, 7), (7, 7, 6)]:
            volume[z][y][x] = 25.0
        result = segment_classical(volume, threshold=0.01, blur_passes=1, h_value=0.0, min_seed_separation=1)
        self.assertGreaterEqual(result.seed_count, 2)
        self.assertGreaterEqual(result.component_count, 2)


if __name__ == '__main__':
    unittest.main()
