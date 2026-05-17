"""Invariant assertions — properties that must hold for any valid input."""

import unittest

import numpy as np

from braggtrack.segmentation import (
    extract_instance_table,
    fill_holes_binary,
    otsu_threshold,
    relabel_sequential,
    remove_small_objects,
    smooth_thresholds,
)
from braggtrack.semantic import make_multiview_encoder, orthogonal_mips


class RelabelInvariants(unittest.TestCase):
    """relabel_sequential output always has labels {0} ∪ {1..N} with no gaps."""

    def _check_sequential(self, labels: np.ndarray) -> None:
        out = relabel_sequential(labels)
        unique = np.unique(out)
        positive = unique[unique > 0]
        if len(positive) == 0:
            return
        expected = np.arange(1, len(positive) + 1)
        np.testing.assert_array_equal(positive, expected)

    def test_random_sparse_labels(self) -> None:
        rng = np.random.RandomState(0)
        labels = np.zeros((8, 8, 8), dtype=int)
        for lbl in [3, 7, 15, 42, 100]:
            z, y, x = rng.randint(0, 8, size=3)
            labels[z, y, x] = lbl
        self._check_sequential(labels)

    def test_already_sequential(self) -> None:
        labels = np.zeros((4, 4, 4), dtype=int)
        labels[0, 0, 0] = 1
        labels[1, 1, 1] = 2
        labels[2, 2, 2] = 3
        self._check_sequential(labels)

    def test_single_label(self) -> None:
        labels = np.zeros((3, 3, 3), dtype=int)
        labels[1, 1, 1] = 99
        out = relabel_sequential(labels)
        self.assertEqual(int(out[1, 1, 1]), 1)

    def test_all_background(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=int)
        out = relabel_sequential(labels)
        self.assertTrue(np.all(out == 0))


class FillHolesInvariants(unittest.TestCase):
    """fill_holes_binary output is always a superset of the input mask."""

    def test_superset_random_mask(self) -> None:
        rng = np.random.RandomState(1)
        mask = rng.rand(10, 10, 10) > 0.7
        filled = fill_holes_binary(mask)
        self.assertTrue(np.all(filled[mask]))

    def test_superset_solid_with_hole(self) -> None:
        mask = np.ones((7, 7, 7), dtype=bool)
        mask[3, 3, 3] = False
        filled = fill_holes_binary(mask)
        self.assertTrue(np.all(filled[mask]))
        self.assertTrue(np.all(filled))

    def test_empty_mask_stays_empty(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        filled = fill_holes_binary(mask)
        self.assertTrue(np.all(~filled))


class RemoveSmallObjectsInvariants(unittest.TestCase):
    """Removing small objects never introduces new labels or enlarges existing ones."""

    def test_output_labels_are_subset_of_input(self) -> None:
        rng = np.random.RandomState(2)
        labels = np.zeros((8, 8, 8), dtype=int)
        for lbl in range(1, 6):
            n_voxels = rng.randint(1, 20)
            coords = rng.randint(0, 8, size=(n_voxels, 3))
            for c in coords:
                labels[c[0], c[1], c[2]] = lbl

        out = remove_small_objects(labels, min_size=5)
        out_labels = set(np.unique(out)) - {0}
        in_labels = set(np.unique(labels)) - {0}
        self.assertTrue(out_labels.issubset(in_labels))

    def test_kept_label_voxel_count_unchanged(self) -> None:
        labels = np.zeros((6, 6, 6), dtype=int)
        labels[0:3, 0:3, 0:3] = 1  # 27 voxels
        labels[5, 5, 5] = 2  # 1 voxel
        out = remove_small_objects(labels, min_size=5)
        self.assertEqual(np.count_nonzero(out == 1), 27)


class EigenvalueInvariants(unittest.TestCase):
    """Eigenvalues from extract_instance_table are always non-negative and descending."""

    def test_random_blobs(self) -> None:
        rng = np.random.RandomState(3)
        labels = np.zeros((20, 20, 20), dtype=int)
        intensity = rng.rand(20, 20, 20) * 10

        for lbl in range(1, 5):
            center = rng.randint(3, 17, size=3)
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        labels[center[0] + dz, center[1] + dy, center[2] + dx] = lbl

        table = extract_instance_table(labels, intensity)
        for row in table:
            self.assertGreaterEqual(row["eig_1"], 0.0, f"Negative eig_1 for label {row['label']}")
            self.assertGreaterEqual(row["eig_2"], 0.0, f"Negative eig_2 for label {row['label']}")
            self.assertGreaterEqual(row["eig_3"], 0.0, f"Negative eig_3 for label {row['label']}")
            self.assertGreaterEqual(row["eig_1"], row["eig_2"])
            self.assertGreaterEqual(row["eig_2"], row["eig_3"])


class OtsuInvariants(unittest.TestCase):
    """Otsu threshold always lies within the range of input values."""

    def test_threshold_within_range(self) -> None:
        rng = np.random.RandomState(4)
        values = list(rng.rand(200) * 100)
        thr = otsu_threshold(values)
        self.assertGreaterEqual(thr, min(values))
        self.assertLessEqual(thr, max(values))

    def test_bimodal_between_modes(self) -> None:
        values = [1.0] * 50 + [100.0] * 50
        thr = otsu_threshold(values)
        self.assertGreater(thr, 1.0)
        self.assertLess(thr, 100.0)


class SmoothThresholdsInvariants(unittest.TestCase):
    """Smoothed thresholds have lower variance than raw when outliers are present."""

    def test_variance_reduced_with_outliers(self) -> None:
        raw = [180.0] * 20
        raw[5] = 500.0
        raw[15] = 10.0
        smoothed = smooth_thresholds(raw, window=5)
        self.assertLess(float(np.var(smoothed)), float(np.var(raw)))

    def test_output_length_equals_input(self) -> None:
        for n in [1, 2, 5, 20, 100]:
            raw = list(np.linspace(100, 200, n))
            smoothed = smooth_thresholds(raw, window=7)
            self.assertEqual(len(smoothed), n)


class MockEncoderInvariants(unittest.TestCase):
    """Mock encoder always produces unit-norm 384-d vectors."""

    def test_unit_norm_various_inputs(self) -> None:
        enc = make_multiview_encoder("mock")
        rng = np.random.RandomState(5)
        for _ in range(10):
            shape = (rng.randint(2, 20), rng.randint(2, 20))
            m1 = rng.rand(*shape).astype(np.float32)
            m2 = rng.rand(*shape).astype(np.float32)
            m3 = rng.rand(*shape).astype(np.float32)
            vec = enc.embed(m1, m2, m3)
            self.assertEqual(vec.shape, (384,))
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_zero_input_still_unit_norm(self) -> None:
        enc = make_multiview_encoder("mock")
        z = np.zeros((4, 4), dtype=np.float32)
        vec = enc.embed(z, z, z)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
