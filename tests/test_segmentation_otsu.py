import unittest

import numpy as np

from braggtrack.segmentation import (
    connected_components_3d,
    flag_outlier_frames,
    otsu_threshold,
    segment_volume,
    smooth_thresholds,
)


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


class SmoothThresholdTests(unittest.TestCase):
    def test_stable_sequence_unchanged(self) -> None:
        raw = [180.0, 181.0, 180.5, 179.8, 180.2]
        smoothed = smooth_thresholds(raw, window=3)
        self.assertEqual(len(smoothed), 5)
        for r, s in zip(raw, smoothed):
            self.assertAlmostEqual(s, r, delta=1.5)

    def test_single_outlier_suppressed(self) -> None:
        raw = [180.0, 180.0, 500.0, 180.0, 180.0]
        smoothed = smooth_thresholds(raw, window=3)
        self.assertAlmostEqual(float(smoothed[2]), 180.0, delta=1.0)

    def test_gradual_drift_tracked(self) -> None:
        raw = list(np.linspace(180, 200, 20))
        smoothed = smooth_thresholds(raw, window=5)
        self.assertAlmostEqual(float(smoothed[0]), 180.0, delta=3.0)
        self.assertAlmostEqual(float(smoothed[-1]), 200.0, delta=3.0)

    def test_flag_outlier_detects_flash(self) -> None:
        raw = [180.0] * 10
        raw[5] = 500.0
        flags = flag_outlier_frames(raw, window=5)
        self.assertTrue(flags[5])
        self.assertFalse(any(flags[i] for i in range(10) if i != 5))

    def test_single_frame(self) -> None:
        smoothed = smooth_thresholds([182.0], window=5)
        self.assertEqual(len(smoothed), 1)
        self.assertAlmostEqual(float(smoothed[0]), 182.0)

    def test_two_frames(self) -> None:
        smoothed = smooth_thresholds([180.0, 184.0], window=5)
        self.assertEqual(len(smoothed), 2)


if __name__ == '__main__':
    unittest.main()
