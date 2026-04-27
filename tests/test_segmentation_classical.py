import unittest

import numpy as np

from braggtrack.segmentation import (
    h_maxima_seeds,
    local_maxima_seeds,
    log_enhance_3d,
    otsu_threshold,
    segment_classical,
    watershed_from_seeds,
)


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
        result = segment_classical(volume, threshold=0.01, blur_passes=1, h_value=0.0, min_seed_separation=1,
                                   seed_response_percentile=99.5)
        self.assertGreaterEqual(result.seed_count, 2)
        self.assertGreaterEqual(result.component_count, 2)

    def test_foreground_mask_is_intensity_domain(self) -> None:
        """Regression: watershed mask must come from the raw intensity,
        not from the LoG response. On a bimodal volume the intensity mask
        covers a meaningful fraction; a response-domain mask would be tiny."""
        rng = np.random.RandomState(0)
        volume = rng.normal(loc=100.0, scale=5.0, size=(30, 30, 30))
        zz, yy, xx = np.mgrid[0:30, 0:30, 0:30]
        for cz, cy, cx in [(8, 8, 8), (22, 22, 22)]:
            volume += 200.0 * np.exp(
                -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 3.0 ** 2)
            )

        thr = otsu_threshold(volume.ravel())
        result = segment_classical(volume, threshold=thr, seed_response_percentile=99.5)
        foreground_mask = volume >= thr

        # Every labelled voxel lies inside the intensity foreground.
        self.assertTrue(np.all(foreground_mask[result.labeled_volume > 0]))
        # The foreground is non-trivially large on this bimodal volume.
        self.assertGreater(foreground_mask.mean(), 0.005)
        # Both blobs are recovered.
        self.assertGreaterEqual(result.component_count, 2)

    def test_seed_floor_uses_response_distribution(self) -> None:
        """The seed floor is derived from the LoG response *inside* the
        foreground, not from the intensity threshold itself."""
        volume = np.ones((12, 12, 12))
        volume[3, 3, 3] = 50.0
        volume[9, 9, 9] = 50.0
        # A tiny intensity threshold admits almost all voxels; the seed
        # floor must still be tight enough on the response to pick only
        # the two peaks, not every flat voxel.
        result = segment_classical(volume, threshold=0.5, h_value=0.0, seed_response_percentile=99.5)
        self.assertEqual(result.seed_count, 2)

    def test_watershed_from_seeds_takes_mask(self) -> None:
        response = np.zeros((6, 6, 6))
        response[2, 2, 2] = 1.0
        mask = np.ones_like(response, dtype=bool)
        labels = watershed_from_seeds(response, seeds=[(2, 2, 2)], mask=mask)
        self.assertEqual(labels.shape, response.shape)
        self.assertEqual(int(labels[2, 2, 2]), 1)


if __name__ == '__main__':
    unittest.main()
