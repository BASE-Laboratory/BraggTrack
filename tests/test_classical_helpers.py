"""Tests for braggtrack.segmentation.classical (gaussian_blur, laplacian, LoG)."""

import unittest

import numpy as np

from braggtrack.segmentation.classical import gaussian_blur_3d, laplacian_3d, log_enhance_3d


class GaussianBlur3DTests(unittest.TestCase):
    def test_preserves_shape(self) -> None:
        volume = np.random.RandomState(0).rand(8, 8, 8)
        out = gaussian_blur_3d(volume, passes=2, sigma=1.0)
        self.assertEqual(out.shape, (8, 8, 8))

    def test_reduces_variance(self) -> None:
        volume = np.random.RandomState(1).rand(10, 10, 10)
        blurred = gaussian_blur_3d(volume, passes=3, sigma=1.5)
        self.assertLess(blurred.var(), volume.var())

    def test_uniform_unchanged(self) -> None:
        volume = np.full((6, 6, 6), 5.0)
        blurred = gaussian_blur_3d(volume, passes=2)
        np.testing.assert_allclose(blurred, 5.0, atol=1e-10)

    def test_multiple_passes_smoother(self) -> None:
        volume = np.random.RandomState(2).rand(8, 8, 8)
        b1 = gaussian_blur_3d(volume, passes=1)
        b3 = gaussian_blur_3d(volume, passes=3)
        self.assertLessEqual(b3.var(), b1.var())


class Laplacian3DTests(unittest.TestCase):
    def test_constant_field_zero(self) -> None:
        volume = np.full((6, 6, 6), 3.0)
        lap = laplacian_3d(volume)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)

    def test_peak_has_negative_laplacian(self) -> None:
        volume = np.zeros((7, 7, 7))
        volume[3, 3, 3] = 10.0
        lap = laplacian_3d(volume)
        self.assertLess(lap[3, 3, 3], 0)

    def test_shape_preserved(self) -> None:
        volume = np.ones((5, 6, 7))
        lap = laplacian_3d(volume)
        self.assertEqual(lap.shape, (5, 6, 7))


class LogEnhance3DTests(unittest.TestCase):
    def test_peak_enhanced(self) -> None:
        volume = np.zeros((9, 9, 9))
        volume[4, 4, 4] = 10.0
        enhanced = log_enhance_3d(volume, blur_passes=1, sigma=1.0)
        self.assertGreater(enhanced[4, 4, 4], enhanced[0, 0, 0])

    def test_flat_volume_near_zero(self) -> None:
        volume = np.full((6, 6, 6), 7.0)
        enhanced = log_enhance_3d(volume, blur_passes=1)
        np.testing.assert_allclose(enhanced, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
