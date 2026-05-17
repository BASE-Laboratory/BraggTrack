"""Tests for braggtrack.segmentation.features (extract_instance_table)."""

import unittest

import numpy as np

from braggtrack.segmentation import extract_instance_table


class ExtractInstanceTableTests(unittest.TestCase):
    def _single_blob_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """A 10x10x10 volume with one labelled blob at centre."""
        labels = np.zeros((10, 10, 10), dtype=int)
        intensity = np.ones((10, 10, 10), dtype=np.float64)
        labels[4:7, 4:7, 4:7] = 1  # 3x3x3 = 27 voxels
        intensity[4:7, 4:7, 4:7] = 10.0
        return labels, intensity

    def test_single_blob_fields(self) -> None:
        labels, intensity = self._single_blob_labels()
        table = extract_instance_table(labels, intensity)
        self.assertEqual(len(table), 1)
        row = table[0]
        self.assertEqual(row["label"], 1)
        self.assertEqual(row["voxel_count"], 27)
        self.assertAlmostEqual(row["integrated_intensity"], 27 * 10.0)

    def test_centroid_position(self) -> None:
        labels, intensity = self._single_blob_labels()
        table = extract_instance_table(labels, intensity)
        row = table[0]
        self.assertAlmostEqual(row["centroid_mu"], 5.0, places=5)
        self.assertAlmostEqual(row["centroid_d"], 5.0, places=5)
        self.assertAlmostEqual(row["centroid_chi"], 5.0, places=5)

    def test_bbox_bounds(self) -> None:
        labels, intensity = self._single_blob_labels()
        table = extract_instance_table(labels, intensity)
        row = table[0]
        self.assertEqual(row["bbox_min_z"], 4)
        self.assertEqual(row["bbox_max_z"], 6)
        self.assertEqual(row["bbox_min_y"], 4)
        self.assertEqual(row["bbox_max_y"], 6)
        self.assertEqual(row["bbox_min_x"], 4)
        self.assertEqual(row["bbox_max_x"], 6)

    def test_eigenvalues_symmetric_blob(self) -> None:
        labels, intensity = self._single_blob_labels()
        table = extract_instance_table(labels, intensity)
        row = table[0]
        self.assertAlmostEqual(row["eig_1"], row["eig_2"], places=5)
        self.assertAlmostEqual(row["eig_2"], row["eig_3"], places=5)

    def test_multiple_blobs(self) -> None:
        labels = np.zeros((10, 10, 10), dtype=int)
        intensity = np.ones((10, 10, 10), dtype=np.float64) * 5.0
        labels[1, 1, 1] = 1
        labels[8, 8, 8] = 2
        intensity[1, 1, 1] = 20.0
        intensity[8, 8, 8] = 30.0
        table = extract_instance_table(labels, intensity)
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]["label"], 1)
        self.assertEqual(table[1]["label"], 2)

    def test_zero_intensity_fallback(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=int)
        intensity = np.zeros((5, 5, 5), dtype=np.float64)
        labels[1:3, 1:3, 1:3] = 1
        table = extract_instance_table(labels, intensity)
        self.assertEqual(len(table), 1)
        row = table[0]
        self.assertAlmostEqual(row["centroid_mu"], 1.5, places=5)
        self.assertAlmostEqual(row["centroid_d"], 1.5, places=5)
        self.assertAlmostEqual(row["centroid_chi"], 1.5, places=5)

    def test_empty_labels_returns_empty(self) -> None:
        labels = np.zeros((4, 4, 4), dtype=int)
        intensity = np.ones((4, 4, 4), dtype=np.float64)
        table = extract_instance_table(labels, intensity)
        self.assertEqual(table, [])

    def test_weighted_centroid_shifts_toward_bright_voxel(self) -> None:
        labels = np.zeros((10, 10, 10), dtype=int)
        intensity = np.ones((10, 10, 10), dtype=np.float64)
        labels[3, 5, 5] = 1
        labels[4, 5, 5] = 1
        labels[5, 5, 5] = 1
        intensity[3, 5, 5] = 1.0
        intensity[4, 5, 5] = 1.0
        intensity[5, 5, 5] = 100.0  # much brighter
        table = extract_instance_table(labels, intensity)
        row = table[0]
        self.assertGreater(row["centroid_mu"], 4.0)


if __name__ == "__main__":
    unittest.main()
