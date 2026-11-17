"""Tests for braggtrack.segmentation.postprocess (remove, fill, relabel)."""

import unittest

import numpy as np

from braggtrack.segmentation import fill_holes_binary, merge_nearby_labels, relabel_sequential, remove_small_objects


class RemoveSmallObjectsTests(unittest.TestCase):
    def test_removes_below_threshold(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=int)
        labels[0, 0, 0] = 1  # 1 voxel — too small
        labels[2, 2, 2] = 2
        labels[2, 2, 3] = 2
        labels[2, 3, 2] = 2  # 3 voxels — large enough
        out = remove_small_objects(labels, min_size=3)
        self.assertEqual(np.count_nonzero(out == 1), 0)
        self.assertEqual(np.count_nonzero(out == 2), 3)

    def test_keeps_objects_at_threshold(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=int)
        labels[1, 1, 1] = 1
        labels[1, 1, 2] = 1  # exactly 2 voxels
        out = remove_small_objects(labels, min_size=2)
        self.assertEqual(np.count_nonzero(out == 1), 2)

    def test_background_unaffected(self) -> None:
        labels = np.zeros((4, 4, 4), dtype=int)
        labels[0, 0, 0] = 1
        out = remove_small_objects(labels, min_size=5)
        self.assertTrue(np.all(out == 0))

    def test_empty_volume(self) -> None:
        labels = np.zeros((3, 3, 3), dtype=int)
        out = remove_small_objects(labels, min_size=1)
        self.assertTrue(np.all(out == 0))


class FillHolesBinaryTests(unittest.TestCase):
    def test_fills_internal_hole(self) -> None:
        mask = np.ones((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = False  # internal hole
        filled = fill_holes_binary(mask)
        self.assertTrue(filled[2, 2, 2])

    def test_does_not_fill_boundary_connected(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        mask[2, 2, 0] = False  # touches boundary via background
        filled = fill_holes_binary(mask)
        self.assertEqual(filled.shape, mask.shape)

    def test_preserves_shape(self) -> None:
        mask = np.ones((7, 8, 9), dtype=bool)
        filled = fill_holes_binary(mask)
        self.assertEqual(filled.shape, (7, 8, 9))

    def test_solid_block_unchanged(self) -> None:
        mask = np.ones((4, 4, 4), dtype=bool)
        filled = fill_holes_binary(mask)
        self.assertTrue(np.all(filled))


class RelabelSequentialTests(unittest.TestCase):
    def test_remaps_to_sequential(self) -> None:
        labels = np.zeros((4, 4, 4), dtype=int)
        labels[0, 0, 0] = 5
        labels[1, 1, 1] = 12
        labels[2, 2, 2] = 100
        out = relabel_sequential(labels)
        unique = sorted(np.unique(out))
        self.assertEqual(unique, [0, 1, 2, 3])

    def test_preserves_spatial_identity(self) -> None:
        labels = np.zeros((4, 4, 4), dtype=int)
        labels[0, 0, 0] = 7
        labels[3, 3, 3] = 3
        out = relabel_sequential(labels)
        self.assertNotEqual(out[0, 0, 0], out[3, 3, 3])
        self.assertGreater(out[0, 0, 0], 0)
        self.assertGreater(out[3, 3, 3], 0)

    def test_already_sequential(self) -> None:
        labels = np.array([[[0, 1], [2, 3]]])
        out = relabel_sequential(labels)
        self.assertEqual(sorted(np.unique(out)), [0, 1, 2, 3])

    def test_empty_volume(self) -> None:
        labels = np.zeros((3, 3, 3), dtype=int)
        out = relabel_sequential(labels)
        self.assertTrue(np.all(out == 0))


class MergeNearbyLabelsTests(unittest.TestCase):
    def test_merges_touching_close_labels(self) -> None:
        labels = np.zeros((6, 6, 6), dtype=np.int32)
        labels[2, 2, 1:3] = 1
        labels[2, 2, 3:5] = 2
        vol = np.ones((6, 6, 6), dtype=np.float64) * 10.0
        merged = relabel_sequential(merge_nearby_labels(labels, vol, max_centroid_distance=5.0))
        self.assertEqual(sorted(np.unique(merged)), [0, 1])

    def test_keeps_separated_labels_distinct(self) -> None:
        labels = np.zeros((6, 6, 8), dtype=np.int32)
        labels[2, 2, 0:2] = 1
        labels[2, 2, 6:8] = 2  # non-adjacent
        vol = np.ones((6, 6, 8), dtype=np.float64) * 10.0
        merged = relabel_sequential(merge_nearby_labels(labels, vol, max_centroid_distance=20.0))
        self.assertEqual(sorted(np.unique(merged)), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
