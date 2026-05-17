"""Error-path and edge-case tests for modules that raise exceptions."""

import unittest

import numpy as np

from braggtrack.io.beamline import BeamlineAdapter
from braggtrack.segmentation import otsu_threshold, segment_volume, smooth_thresholds
from braggtrack.segmentation.classical import segment_classical
from braggtrack.segmentation.features import extract_instance_table
from braggtrack.segmentation.postprocess import fill_holes_binary, relabel_sequential
from braggtrack.semantic import orthogonal_mips
from braggtrack.tracking import PositionShapeCost, build_tracks


class OtsuErrorTests(unittest.TestCase):
    def test_empty_values_raises(self) -> None:
        with self.assertRaises(ValueError):
            otsu_threshold([])

    def test_single_value_returns_that_value(self) -> None:
        thr = otsu_threshold([5.0])
        self.assertAlmostEqual(thr, 5.0)

    def test_constant_values_returns_that_value(self) -> None:
        thr = otsu_threshold([3.0, 3.0, 3.0, 3.0])
        self.assertAlmostEqual(thr, 3.0)


class SegmentVolumeErrorTests(unittest.TestCase):
    def test_empty_volume_raises(self) -> None:
        with self.assertRaises(ValueError):
            segment_volume(np.array([]), method="otsu")

    def test_unsupported_method_raises(self) -> None:
        vol = np.ones((4, 4, 4))
        with self.assertRaises(ValueError):
            segment_volume(vol, method="kmeans")


class SegmentClassicalEdgeTests(unittest.TestCase):
    def test_flat_volume_zero_seeds(self) -> None:
        vol = np.full((10, 10, 10), 5.0)
        result = segment_classical(vol, threshold=1.0, seed_response_percentile=99.5)
        self.assertEqual(result.seed_count, 0)
        self.assertEqual(result.component_count, 0)

    def test_single_bright_voxel(self) -> None:
        vol = np.ones((8, 8, 8))
        vol[4, 4, 4] = 50.0
        result = segment_classical(vol, threshold=0.5, h_value=0.0, seed_response_percentile=99.5)
        self.assertGreaterEqual(result.seed_count, 1)

    def test_all_foreground_still_works(self) -> None:
        vol = np.random.RandomState(99).rand(8, 8, 8) * 10 + 100
        vol[4, 4, 4] = 500.0
        result = segment_classical(vol, threshold=0.0, seed_response_percentile=99.5)
        self.assertGreaterEqual(result.seed_count, 1)


class SmoothThresholdsEdgeTests(unittest.TestCase):
    def test_empty_sequence(self) -> None:
        out = smooth_thresholds([])
        self.assertEqual(len(out), 0)

    def test_window_exceeds_length(self) -> None:
        out = smooth_thresholds([100.0, 101.0], window=50)
        self.assertEqual(len(out), 2)

    def test_all_identical_values(self) -> None:
        out = smooth_thresholds([42.0] * 20, window=5)
        np.testing.assert_allclose(out, 42.0)


class BeamlineAdapterErrorTests(unittest.TestCase):
    def test_unparseable_scan_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            BeamlineAdapter._parse_scan_index("no_digits_here")

    def test_parse_scan_index_extracts_digits(self) -> None:
        self.assertEqual(BeamlineAdapter._parse_scan_index("scan0042"), 42)
        self.assertEqual(BeamlineAdapter._parse_scan_index("scan123abc456"), 123456)


class BuildTracksEdgeTests(unittest.TestCase):
    def _make_spot(self, mu: float = 0.0, chi: float = 0.0, d: float = 0.0) -> dict:
        return {
            "label": 1, "voxel_count": 10, "integrated_intensity": 100.0,
            "centroid_mu": mu, "centroid_chi": chi, "centroid_d": d,
            "eig_1": 0.5, "eig_2": 0.5, "eig_3": 0.5,
        }

    def test_empty_scan_tables(self) -> None:
        G = build_tracks([], cost_fn=PositionShapeCost())
        self.assertEqual(len(G.nodes), 0)

    def test_single_frame(self) -> None:
        feats = [[self._make_spot(1.0, 2.0, 3.0), self._make_spot(5.0, 6.0, 7.0)]]
        G = build_tracks(feats, cost_fn=PositionShapeCost())
        self.assertEqual(len(G.nodes), 2)

    def test_empty_middle_frame(self) -> None:
        feats = [
            [self._make_spot(0.0, 0.0, 0.0)],
            [],
            [self._make_spot(0.1, 0.1, 0.1)],
        ]
        G = build_tracks(feats, cost_fn=PositionShapeCost())
        self.assertGreater(len(G.nodes), 0)

    def test_all_empty_frames(self) -> None:
        feats = [[], [], []]
        G = build_tracks(feats, cost_fn=PositionShapeCost())
        self.assertEqual(len(G.nodes), 0)


class ExtractInstanceTableEdgeTests(unittest.TestCase):
    def test_negative_intensity_falls_back_to_geometric_centroid(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=int)
        intensity = np.full((5, 5, 5), -10.0)
        labels[1:4, 1:4, 1:4] = 1
        table = extract_instance_table(labels, intensity)
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table[0]["centroid_mu"], 2.0, places=3)

    def test_single_voxel_zero_covariance(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=int)
        intensity = np.ones((5, 5, 5))
        labels[2, 3, 4] = 1
        intensity[2, 3, 4] = 10.0
        table = extract_instance_table(labels, intensity)
        row = table[0]
        self.assertAlmostEqual(row["cov_zz"], 0.0)
        self.assertAlmostEqual(row["cov_yy"], 0.0)
        self.assertAlmostEqual(row["cov_xx"], 0.0)
        self.assertAlmostEqual(row["eig_1"], 0.0)


class OrthogonalMipsEdgeTests(unittest.TestCase):
    def test_single_voxel_volume(self) -> None:
        vol = np.array([[[5.0]]])
        mip_mu, mip_chi, mip_d = orthogonal_mips(vol)
        self.assertEqual(float(mip_mu[0, 0]), 5.0)
        self.assertEqual(float(mip_chi[0, 0]), 5.0)
        self.assertEqual(float(mip_d[0, 0]), 5.0)

    def test_4d_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            orthogonal_mips(np.zeros((2, 2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
