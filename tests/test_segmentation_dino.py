"""Tests for DINO-based segmentation (mock backend, no GPU)."""

import unittest

import numpy as np

from braggtrack.segmentation.dino_segment import (
    DinoSegmentationResult,
    _cluster_feature_map,
    _stitch_slices_3d,
    _upsample_labels,
    segment_dino,
)
from braggtrack.semantic.dino import MockPatchEncoder, make_patch_encoder


class TestMockPatchEncoder(unittest.TestCase):
    def test_output_shape(self) -> None:
        enc = MockPatchEncoder()
        img = np.random.default_rng(42).standard_normal((56, 56))
        features = enc.extract_patch_features(img)
        self.assertEqual(features.shape, (4, 4, 384))

    def test_deterministic(self) -> None:
        enc = MockPatchEncoder()
        img = np.ones((28, 28))
        a = enc.extract_patch_features(img)
        b = enc.extract_patch_features(img)
        np.testing.assert_array_equal(a, b)

    def test_l2_normalized(self) -> None:
        enc = MockPatchEncoder()
        img = np.random.default_rng(7).standard_normal((42, 42))
        features = enc.extract_patch_features(img)
        norms = np.linalg.norm(features, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestMakePatchEncoder(unittest.TestCase):
    def test_mock_backend(self) -> None:
        enc = make_patch_encoder("mock")
        self.assertIsInstance(enc, MockPatchEncoder)
        self.assertEqual(enc.patch_size, 14)
        self.assertEqual(enc.feature_dim, 384)


class TestClusterFeatureMap(unittest.TestCase):
    def test_basic_clustering(self) -> None:
        rng = np.random.default_rng(99)
        features = np.zeros((6, 6, 32), dtype=np.float32)
        features[:3, :, :] = rng.standard_normal((3, 6, 32)) + 5.0
        features[3:, :, :] = rng.standard_normal((3, 6, 32)) - 5.0
        labels = _cluster_feature_map(features, n_components_pca=8, min_cluster_size=3, min_samples=1)
        self.assertEqual(labels.shape, (6, 6))
        self.assertGreaterEqual(len(np.unique(labels[labels > 0])), 1)

    def test_tiny_input_single_region(self) -> None:
        features = np.ones((1, 1, 4), dtype=np.float32)
        labels = _cluster_feature_map(features, n_components_pca=2)
        self.assertEqual(labels.shape, (1, 1))
        self.assertEqual(labels[0, 0], 1)


class TestUpsampleLabels(unittest.TestCase):
    def test_basic_upsample(self) -> None:
        patch_labels = np.array([[1, 2], [3, 0]], dtype=np.int32)
        out = _upsample_labels(patch_labels, target_shape=(28, 28), patch_size=14)
        self.assertEqual(out.shape, (28, 28))
        self.assertEqual(out[0, 0], 1)
        self.assertEqual(out[0, 14], 2)
        self.assertEqual(out[14, 0], 3)
        self.assertEqual(out[14, 14], 0)

    def test_edge_handling(self) -> None:
        patch_labels = np.array([[1]], dtype=np.int32)
        out = _upsample_labels(patch_labels, target_shape=(10, 10), patch_size=14)
        self.assertEqual(out.shape, (10, 10))
        self.assertTrue(np.all(out == 1))


class TestStitchSlices3D(unittest.TestCase):
    def test_empty_input(self) -> None:
        result = _stitch_slices_3d([])
        self.assertEqual(result.shape, (0, 0, 0))

    def test_single_slice(self) -> None:
        sl = np.array([[1, 0], [0, 2]], dtype=np.int32)
        result = _stitch_slices_3d([sl])
        self.assertEqual(result.shape, (1, 2, 2))
        self.assertEqual(len(np.unique(result[result > 0])), 2)

    def test_overlapping_slices_merge(self) -> None:
        sl1 = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)
        sl2 = np.array([[2, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.int32)
        result = _stitch_slices_3d([sl1, sl2], min_overlap_fraction=0.3)
        self.assertEqual(result.shape, (2, 3, 3))
        labels_s0 = result[0][result[0] > 0]
        labels_s1 = result[1][result[1] > 0]
        self.assertEqual(len(np.unique(labels_s0)), 1)
        self.assertEqual(len(np.unique(labels_s1)), 1)
        self.assertEqual(np.unique(labels_s0)[0], np.unique(labels_s1)[0])

    def test_non_overlapping_stay_separate(self) -> None:
        sl1 = np.array([[1, 0], [0, 0]], dtype=np.int32)
        sl2 = np.array([[0, 0], [0, 2]], dtype=np.int32)
        result = _stitch_slices_3d([sl1, sl2], min_overlap_fraction=0.3)
        labels_s0 = result[0][result[0] > 0]
        labels_s1 = result[1][result[1] > 0]
        self.assertTrue(len(labels_s0) > 0)
        self.assertTrue(len(labels_s1) > 0)
        self.assertNotEqual(np.unique(labels_s0)[0], np.unique(labels_s1)[0])


class TestSegmentDino(unittest.TestCase):
    def test_mock_backend_runs(self) -> None:
        rng = np.random.default_rng(42)
        volume = rng.standard_normal((6, 28, 28)).astype(np.float64)
        volume[2:4, 10:18, 10:18] += 10.0
        result = segment_dino(volume, backend="mock")
        self.assertIsInstance(result, DinoSegmentationResult)
        self.assertEqual(result.labeled_volume.shape, volume.shape)
        self.assertEqual(result.response.size, 0)
        self.assertGreater(result.threshold, 0)

    def test_result_field_compatible(self) -> None:
        volume = np.random.default_rng(0).standard_normal((4, 28, 28)).astype(np.float64)
        volume += 5.0
        result = segment_dino(volume, backend="mock")
        self.assertIsInstance(result.threshold, float)
        self.assertIsInstance(result.seed_count, int)
        self.assertIsInstance(result.component_count, int)
        self.assertIsInstance(result.labeled_volume, np.ndarray)
        self.assertIsInstance(result.response, np.ndarray)

    def test_foreground_mask_applied(self) -> None:
        rng = np.random.default_rng(123)
        volume = rng.standard_normal((4, 28, 28)).astype(np.float64)
        volume[:, :14, :14] += 20.0
        result = segment_dino(volume, backend="mock")
        below_threshold = volume < result.threshold
        self.assertTrue(np.all(result.labeled_volume[below_threshold] == 0))

    def test_labels_are_nonnegative(self) -> None:
        volume = np.random.default_rng(55).standard_normal((4, 28, 28)).astype(np.float64)
        volume += 3.0
        result = segment_dino(volume, backend="mock")
        self.assertTrue(np.all(result.labeled_volume >= 0))


if __name__ == "__main__":
    unittest.main()
