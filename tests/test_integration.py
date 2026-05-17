"""End-to-end integration test exercising the full pipeline on bundled data.

Replaces fragile subprocess-based acceptance wrappers with a single fast
in-process test that verifies the pipeline produces sensible results.
"""

import unittest

import h5py
import numpy as np

from braggtrack.io import discover_operando_scans, sample_operando_root
from braggtrack.segmentation import (
    extract_instance_table,
    fill_holes_binary,
    otsu_threshold,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
    smooth_thresholds,
)
from braggtrack.semantic import crop_spot_cube, make_multiview_encoder, orthogonal_mips
from braggtrack.tracking import (
    GeometrySemanticCost,
    PositionShapeCost,
    build_tracks,
    compute_tracking_metrics,
    tracks_to_table,
)


def _load_3d_volume(h5_path) -> np.ndarray:
    candidates = []
    with h5py.File(h5_path, "r") as f:

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and np.issubdtype(obj.dtype, np.number):
                candidates.append((name, obj.shape))

        f.visititems(_visit)
        if not candidates:
            raise KeyError(f"No 3D numeric dataset in {h5_path}")
        name = max(candidates, key=lambda t: int(np.prod(t[1])))[0]
        return np.asarray(f[name][...], dtype=np.float64)


def _segment(volume: np.ndarray, threshold: float) -> np.ndarray:
    res = segment_classical(
        volume,
        threshold=threshold,
        blur_passes=1,
        h_value=0.1,
        min_seed_separation=2,
        seed_peak_fraction=0.2,
        seed_response_percentile=99.95,
    )
    labels = remove_small_objects(res.labeled_volume, min_size=8)
    binary = fill_holes_binary(labels > 0)
    return relabel_sequential(np.where(binary, labels, 0))


class EndToEndPipelineTest(unittest.TestCase):
    """Full pipeline on bundled scans: discover → segment → track → embed → semantic track."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scans = discover_operando_scans(sample_operando_root())
        cls.volumes = [_load_3d_volume(s.path) for s in cls.scans]
        raw_thresholds = [otsu_threshold(v.ravel()) for v in cls.volumes]
        cls.smoothed = smooth_thresholds(raw_thresholds, window=5)
        cls.all_labels = [_segment(v, float(t)) for v, t in zip(cls.volumes, cls.smoothed)]
        cls.all_features = [
            extract_instance_table(l, v) for l, v in zip(cls.all_labels, cls.volumes)
        ]

    def test_discovers_three_scans(self) -> None:
        self.assertEqual(len(self.scans), 3)

    def test_volumes_are_3d_nondegenerate(self) -> None:
        for v in self.volumes:
            self.assertEqual(v.ndim, 3)
            self.assertGreater(v.size, 1000)

    def test_spot_counts_are_stable(self) -> None:
        counts = [len(f) for f in self.all_features]
        spread = max(counts) - min(counts)
        self.assertLess(spread, 15, f"Spot counts {counts} have excessive spread")
        for c in counts:
            self.assertGreater(c, 5, "Too few spots detected")
            self.assertLess(c, 100, "Too many spots — likely over-segmentation")

    def test_feature_tables_have_required_columns(self) -> None:
        required = {
            "label", "voxel_count", "integrated_intensity",
            "centroid_mu", "centroid_chi", "centroid_d",
            "eig_1", "eig_2", "eig_3",
            "bbox_min_z", "bbox_max_z", "bbox_min_y", "bbox_max_y",
            "bbox_min_x", "bbox_max_x",
        }
        for feats in self.all_features:
            for row in feats:
                self.assertTrue(required.issubset(row.keys()), f"Missing keys: {required - row.keys()}")

    def test_eigenvalues_are_non_negative(self) -> None:
        for feats in self.all_features:
            for row in feats:
                self.assertGreaterEqual(row["eig_1"], 0.0)
                self.assertGreaterEqual(row["eig_2"], 0.0)
                self.assertGreaterEqual(row["eig_3"], 0.0)

    def test_eigenvalues_are_descending(self) -> None:
        for feats in self.all_features:
            for row in feats:
                self.assertGreaterEqual(row["eig_1"], row["eig_2"])
                self.assertGreaterEqual(row["eig_2"], row["eig_3"])

    def test_physics_tracking_produces_sane_metrics(self) -> None:
        cost = PositionShapeCost(position_weight=1.0, shape_weight=0.5)
        G = build_tracks(self.all_features, cost_fn=cost)
        metrics = compute_tracking_metrics(G, n_scans=len(self.scans))

        self.assertGreater(metrics["total_tracks"], 0)
        self.assertGreater(metrics["full_length_tracks"], 0)
        self.assertLess(metrics["fragmentation_ratio"], 0.8)
        self.assertGreaterEqual(metrics["continued_count"], 1)

    def test_tracks_to_table_has_all_observations(self) -> None:
        cost = PositionShapeCost(position_weight=1.0, shape_weight=0.5)
        G = build_tracks(self.all_features, cost_fn=cost)
        rows = tracks_to_table(G)

        total_spots = sum(len(f) for f in self.all_features)
        self.assertEqual(len(rows), total_spots)

    def test_semantic_embedding_and_tracking(self) -> None:
        encoder = make_multiview_encoder("mock")
        enriched = []
        for v, l, feats in zip(self.volumes, self.all_labels, self.all_features):
            enriched_scan = []
            for row in feats:
                row_copy = dict(row)
                masked, _ = crop_spot_cube(v, l, int(row["label"]), row, margin=3)
                m_mu, m_chi, m_d = orthogonal_mips(masked)
                row_copy["embedding"] = encoder.embed(m_mu, m_chi, m_d)
                enriched_scan.append(row_copy)
            enriched.append(enriched_scan)

        geo = PositionShapeCost(position_weight=1.0, shape_weight=0.5)
        cost = GeometrySemanticCost(geo, cost_alpha=1.0, cost_beta=0.5)
        G = build_tracks(enriched, cost_fn=cost)
        metrics = compute_tracking_metrics(G, n_scans=len(self.scans))

        self.assertGreater(metrics["total_tracks"], 0)
        self.assertLess(metrics["fragmentation_ratio"], 1.0)

    def test_embeddings_are_unit_norm(self) -> None:
        encoder = make_multiview_encoder("mock")
        for v, l, feats in zip(self.volumes, self.all_labels, self.all_features):
            for row in feats[:5]:
                masked, _ = crop_spot_cube(v, l, int(row["label"]), row, margin=3)
                m_mu, m_chi, m_d = orthogonal_mips(masked)
                vec = encoder.embed(m_mu, m_chi, m_d)
                self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
