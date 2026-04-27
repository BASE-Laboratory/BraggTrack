"""Label-by-intensity projection + MIP floor helpers."""

import unittest

import numpy as np

from braggtrack.segmentation import (
    label_projection_by_intensity,
    otsu_floor_from_mip,
)


class LabelProjectionTests(unittest.TestCase):
    def test_picks_label_of_brightest_voxel(self) -> None:
        # Along axis 0, the brightest voxel at (x, y) = (1, 1) is z=2 (val=5),
        # which has label 7. A naive labels.max(axis=0) would return 9 from
        # (z=0, y=1, x=1) even though that voxel is dim (val=1).
        intensity = np.zeros((3, 3, 3), dtype=np.float64)
        labels = np.zeros((3, 3, 3), dtype=np.int32)

        intensity[2, 1, 1] = 5.0
        labels[2, 1, 1] = 7

        intensity[0, 1, 1] = 1.0
        labels[0, 1, 1] = 9  # higher id, but dim

        proj = label_projection_by_intensity(intensity, labels, axis=0)
        self.assertEqual(int(proj[1, 1]), 7)
        self.assertEqual(int(labels.max(axis=0)[1, 1]), 9)  # the old, wrong behaviour

    def test_mip_floor_hides_dim_rays(self) -> None:
        intensity = np.zeros((4, 4, 4), dtype=np.float64)
        labels = np.zeros((4, 4, 4), dtype=np.int32)
        intensity[1, 0, 0] = 100.0
        labels[1, 0, 0] = 3
        intensity[1, 3, 3] = 1.0  # noise-level
        labels[1, 3, 3] = 4

        proj_no_floor = label_projection_by_intensity(intensity, labels, axis=0)
        self.assertEqual(int(proj_no_floor[0, 0]), 3)
        self.assertEqual(int(proj_no_floor[3, 3]), 4)

        proj_floored = label_projection_by_intensity(intensity, labels, axis=0, mip_floor=50.0)
        self.assertEqual(int(proj_floored[0, 0]), 3)
        self.assertEqual(int(proj_floored[3, 3]), 0)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            label_projection_by_intensity(np.zeros((2, 2, 2)), np.zeros((3, 3, 3)))

    def test_otsu_floor_from_mip_between_min_and_max(self) -> None:
        rng = np.random.RandomState(1)
        background = rng.normal(loc=100.0, scale=5.0, size=(10, 20, 20))
        zz, yy, xx = np.mgrid[0:10, 0:20, 0:20]
        volume = background + 500.0 * np.exp(-((zz - 5) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) / 2.0)
        mip = volume.max(axis=0)
        floor = otsu_floor_from_mip(volume, axis=0)
        self.assertGreater(floor, mip.min())
        self.assertLess(floor, mip.max())


if __name__ == '__main__':
    unittest.main()
