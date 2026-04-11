"""Week 4 semantic MIPs and association cost."""

import math
import unittest

import numpy as np

from braggtrack.semantic import crop_spot_cube, make_multiview_encoder, orthogonal_mips
from braggtrack.tracking.assignment import associate_frames
from braggtrack.tracking.cost import GeometrySemanticCost, PositionShapeCost


class TestOrthogonalMips(unittest.TestCase):
    def test_mip_shapes(self) -> None:
        vol = np.random.RandomState(0).rand(5, 4, 6)
        m_mu, m_chi, m_d = orthogonal_mips(vol)
        self.assertEqual(m_mu.shape, (4, 6))
        self.assertEqual(m_chi.shape, (5, 4))
        self.assertEqual(m_d.shape, (5, 6))


class TestCropSpotCube(unittest.TestCase):
    def test_crop_masked(self) -> None:
        vol = np.ones((10, 10, 10), dtype=np.float64) * 3.0
        lab = np.zeros((10, 10, 10), dtype=np.int32)
        lab[2:5, 3:6, 4:7] = 7
        bbox = {
            "bbox_min_z": 2,
            "bbox_max_z": 4,
            "bbox_min_y": 3,
            "bbox_max_y": 5,
            "bbox_min_x": 4,
            "bbox_max_x": 6,
        }
        masked, mask = crop_spot_cube(vol, lab, 7, bbox, margin=0)
        self.assertTrue(np.all(masked[mask > 0] == 3.0))
        self.assertAlmostEqual(float(masked.max()), 3.0)


class TestMockEncoder(unittest.TestCase):
    def test_unit_norm(self) -> None:
        enc = make_multiview_encoder("mock")
        a = np.zeros((3, 4), dtype=np.float64)
        b = np.zeros((3, 5), dtype=np.float64)
        c = np.zeros((3, 6), dtype=np.float64)
        v = enc.embed(a, b, c)
        self.assertEqual(v.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)


class TestGeometrySemanticCost(unittest.TestCase):
    def test_beta_zero_matches_scaled_geometry(self) -> None:
        geo = PositionShapeCost(position_weight=2.0, shape_weight=0.0)
        combo = GeometrySemanticCost(geo, cost_alpha=0.5, cost_beta=0.0)
        s = {"centroid_mu": 1.0, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1.0, "eig_2": 1.0, "eig_3": 1.0}
        t = {"centroid_mu": 2.0, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1.0, "eig_2": 1.0, "eig_3": 1.0}
        self.assertAlmostEqual(combo(s, t), 0.5 * geo(s, t))

    def test_semantic_term(self) -> None:
        geo = PositionShapeCost(position_weight=0.0, shape_weight=0.0)
        combo = GeometrySemanticCost(geo, cost_alpha=1.0, cost_beta=1.0)
        e0 = np.array([1.0, 0.0], dtype=np.float64)
        e1 = np.array([1.0, 0.0], dtype=np.float64)
        s = {**_min_spot(), "embedding": e0}
        t = {**_min_spot(), "embedding": e1}
        self.assertAlmostEqual(combo(s, t), 0.0)

    def test_missing_embedding_inf(self) -> None:
        geo = PositionShapeCost(position_weight=1.0, shape_weight=0.0)
        combo = GeometrySemanticCost(geo, cost_alpha=1.0, cost_beta=0.5)
        s = _min_spot()
        t = _min_spot()
        self.assertEqual(combo(s, t), math.inf)


def _min_spot() -> dict:
    return {
        "centroid_mu": 0.0,
        "centroid_chi": 0.0,
        "centroid_d": 0.0,
        "eig_1": 0.5,
        "eig_2": 0.5,
        "eig_3": 0.5,
    }


class TestSemanticAssignment(unittest.TestCase):
    def test_embedding_disambiguates(self) -> None:
        """Near-overlap crossing: geometry favours the wrong identity pairing."""
        e0 = np.array([1.0, 0.0], dtype=np.float64)
        e1 = np.array([0.0, 1.0], dtype=np.float64)
        g = 20.0
        spots_t = [
            {**_min_spot(), "centroid_mu": 0.0, "embedding": e0},
            {**_min_spot(), "centroid_mu": 0.2, "embedding": e1},
        ]
        spots_t1 = [
            {**_min_spot(), "centroid_mu": 0.1, "embedding": e1},
            {**_min_spot(), "centroid_mu": 0.3, "embedding": e0},
        ]
        geo = PositionShapeCost(position_weight=1.0, shape_weight=0.0, gate_mu=g, gate_chi=g, gate_d=g)
        cost_g = GeometrySemanticCost(geo, cost_alpha=1.0, cost_beta=1.0)
        matches, _, _ = associate_frames(spots_t, spots_t1, cost_g)
        self.assertEqual(dict(matches), {0: 1, 1: 0})


if __name__ == "__main__":
    unittest.main()
