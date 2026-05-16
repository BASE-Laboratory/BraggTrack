"""Tests for braggtrack.semantic (MIPs, mock encoder, embed pipeline)."""

import unittest

import numpy as np

from braggtrack.semantic import crop_spot_cube, embed_multiview_mips, make_multiview_encoder, orthogonal_mips


class OrthogonalMIPsTests(unittest.TestCase):
    def test_output_shapes(self) -> None:
        vol = np.random.RandomState(0).rand(5, 6, 7)
        mip_mu, mip_chi, mip_d = orthogonal_mips(vol)
        self.assertEqual(mip_mu.shape, (6, 7))
        self.assertEqual(mip_chi.shape, (5, 6))
        self.assertEqual(mip_d.shape, (5, 7))

    def test_mip_is_max_projection(self) -> None:
        vol = np.zeros((4, 4, 4))
        vol[2, 1, 3] = 99.0
        mip_mu, mip_chi, mip_d = orthogonal_mips(vol)
        self.assertEqual(float(mip_mu[1, 3]), 99.0)
        self.assertEqual(float(mip_chi[2, 1]), 99.0)
        self.assertEqual(float(mip_d[2, 3]), 99.0)

    def test_rejects_non_3d(self) -> None:
        with self.assertRaises(ValueError):
            orthogonal_mips(np.zeros((4, 4)))


class CropSpotCubeTests(unittest.TestCase):
    def test_crop_extracts_correct_region(self) -> None:
        volume = np.random.RandomState(0).rand(20, 20, 20)
        labels = np.zeros((20, 20, 20), dtype=int)
        labels[5:8, 10:13, 2:5] = 1
        bbox = {
            "bbox_min_z": 5, "bbox_max_z": 7,
            "bbox_min_y": 10, "bbox_max_y": 12,
            "bbox_min_x": 2, "bbox_max_x": 4,
        }
        masked, mask = crop_spot_cube(volume, labels, label_id=1, bbox=bbox, margin=1)
        self.assertEqual(masked.shape, mask.shape)
        self.assertTrue(np.any(mask > 0))

    def test_margin_clamps_to_boundary(self) -> None:
        volume = np.ones((10, 10, 10))
        labels = np.zeros((10, 10, 10), dtype=int)
        labels[0:2, 0:2, 0:2] = 1
        bbox = {
            "bbox_min_z": 0, "bbox_max_z": 1,
            "bbox_min_y": 0, "bbox_max_y": 1,
            "bbox_min_x": 0, "bbox_max_x": 1,
        }
        masked, mask = crop_spot_cube(volume, labels, label_id=1, bbox=bbox, margin=5)
        self.assertGreater(mask.sum(), 0)

    def test_other_labels_masked_out(self) -> None:
        volume = np.ones((10, 10, 10)) * 5.0
        labels = np.zeros((10, 10, 10), dtype=int)
        labels[3, 3, 3] = 1
        labels[4, 4, 4] = 2
        bbox = {
            "bbox_min_z": 3, "bbox_max_z": 4,
            "bbox_min_y": 3, "bbox_max_y": 4,
            "bbox_min_x": 3, "bbox_max_x": 4,
        }
        masked, mask = crop_spot_cube(volume, labels, label_id=1, bbox=bbox, margin=1)
        self.assertEqual(float(mask[mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2]), 0.0)


class MockEncoderTests(unittest.TestCase):
    def test_produces_unit_vector(self) -> None:
        mip = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        vec = embed_multiview_mips(mip, mip, mip, backend="mock")
        self.assertEqual(vec.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_deterministic(self) -> None:
        mip = np.random.RandomState(42).rand(6, 6).astype(np.float32)
        v1 = embed_multiview_mips(mip, mip, mip, backend="mock")
        v2 = embed_multiview_mips(mip, mip, mip, backend="mock")
        np.testing.assert_array_equal(v1, v2)

    def test_different_inputs_different_vectors(self) -> None:
        rng = np.random.RandomState(7)
        m1 = rng.rand(5, 5).astype(np.float32)
        m2 = rng.rand(5, 5).astype(np.float32)
        v1 = embed_multiview_mips(m1, m1, m1, backend="mock")
        v2 = embed_multiview_mips(m2, m2, m2, backend="mock")
        self.assertFalse(np.allclose(v1, v2))

    def test_make_encoder_mock(self) -> None:
        enc = make_multiview_encoder(backend="mock")
        mip = np.ones((4, 4), dtype=np.float32)
        vec = enc.embed(mip, mip, mip)
        self.assertEqual(vec.shape, (384,))


if __name__ == "__main__":
    unittest.main()
