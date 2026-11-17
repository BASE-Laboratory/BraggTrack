"""Unit tests for multi-scan gap closing (beam-dropout & rocking-clip recovery)."""

from __future__ import annotations

import unittest

import numpy as np

from braggtrack.tracking import (
    GapClosingConfig,
    GeometrySemanticCost,
    PositionShapeCost,
    SplitMergeConfig,
    build_tracks,
    compute_tracking_metrics,
    tracks_to_table,
)


def _spot(
    label: int,
    mu: float,
    chi: float,
    d: float,
    intensity: float = 1000.0,
    eig1: float = 2.0,
    eig2: float = 1.5,
    eig3: float = 1.2,
    embedding: np.ndarray | None = None,
) -> dict:
    row = {
        "label": label,
        "centroid_mu": mu,
        "centroid_chi": chi,
        "centroid_d": d,
        "integrated_intensity": intensity,
        "voxel_count": 25,
        "eig_1": eig1,
        "eig_2": eig2,
        "eig_3": eig3,
    }
    if embedding is not None:
        row["embedding"] = embedding
    return row


class GapClosingTests(unittest.TestCase):
    def test_one_scan_beam_dropout_reconnected(self) -> None:
        """A reflection missing at scan 2 (0, 1, -, 3, 4) bridges into 1 end-to-end track."""
        scans = [
            [_spot(1, 10.0, 20.0, 30.0)],
            [_spot(1, 10.0, 20.0, 31.0)],
            [],  # beam dropout at scan 2
            [_spot(1, 10.0, 20.0, 33.0)],
            [_spot(1, 10.0, 20.0, 34.0)],
        ]

        G_no_gap = build_tracks(scans)
        m_no_gap = compute_tracking_metrics(G_no_gap, n_scans=5)
        self.assertEqual(m_no_gap["total_tracks"], 2)
        self.assertEqual(m_no_gap["gap_closed_count"], 0)
        self.assertEqual(m_no_gap["fragmentation_ratio"], 1.0)

        G_gap = build_tracks(scans, gap_closing_config=GapClosingConfig(max_gap=2, max_cost=20.0))
        m_gap = compute_tracking_metrics(G_gap, n_scans=5)
        self.assertEqual(m_gap["total_tracks"], 1)
        self.assertEqual(m_gap["gap_closed_count"], 1)
        self.assertEqual(m_gap["full_length_tracks"], 1)
        self.assertEqual(m_gap["fragmentation_ratio"], 0.0)

        gap_edges = [(u, v, d) for u, v, d in G_gap.edges(data=True) if d.get("edge_type") == "gap_close"]
        self.assertEqual(len(gap_edges), 1)
        self.assertEqual(gap_edges[0][2]["gap_scans"], 1)

    def test_two_scan_dropout_respects_max_gap_limit(self) -> None:
        """A 2-scan gap is ignored when max_gap=1 and bridged when max_gap=2."""
        scans = [
            [_spot(1, 5.0, 5.0, 10.0)],
            [],
            [],
            [_spot(1, 5.0, 5.0, 11.5)],
        ]

        G_g1 = build_tracks(scans, gap_closing_config=GapClosingConfig(max_gap=1, max_cost=20.0))
        self.assertEqual(compute_tracking_metrics(G_g1, n_scans=4)["total_tracks"], 2)

        G_g2 = build_tracks(scans, gap_closing_config=GapClosingConfig(max_gap=2, max_cost=20.0))
        m_g2 = compute_tracking_metrics(G_g2, n_scans=4)
        self.assertEqual(m_g2["total_tracks"], 1)
        self.assertEqual(m_g2["gap_closed_count"], 1)

    def test_chained_multi_dropouts_preserve_single_track_id(self) -> None:
        """Repeated dropouts (0 -> 2 -> 4) propagate the initial track_id chronologically."""
        scans = [
            [_spot(1, 12.0, 15.0, 20.0)],
            [],
            [_spot(1, 12.0, 15.0, 21.0)],
            [],
            [_spot(1, 12.0, 15.0, 22.0)],
        ]
        G = build_tracks(scans, gap_closing_config=GapClosingConfig(max_gap=1, max_cost=20.0))
        rows = tracks_to_table(G)
        self.assertEqual({r["track_id"] for r in rows}, {1})
        self.assertEqual(compute_tracking_metrics(G, n_scans=5)["gap_closed_count"], 2)

    def test_semantic_cosine_gate_blocks_impostor_and_severs_stray_jump(self) -> None:
        """Semantic embeddings block dissimilar gap matches and sever stray intermediate noise jumps."""
        e_grain = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e_noise = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Grain 1 at (10, 10, 10) drops out at scan 1 while a distant noise spot appears at (80, 80, 80).
        scans = [
            [_spot(1, 10.0, 10.0, 10.0, embedding=e_grain)],
            [_spot(1, 80.0, 80.0, 80.0, embedding=e_noise)],
            [_spot(1, 10.0, 10.0, 10.5, embedding=e_grain)],
        ]
        cost_fn = GeometrySemanticCost(PositionShapeCost(), cost_alpha=1.0, cost_beta=2.0)
        G = build_tracks(
            scans,
            cost_fn=cost_fn,
            gap_closing_config=GapClosingConfig(max_gap=1, max_cost=15.0, min_cosine_similarity=0.8),
        )
        m = compute_tracking_metrics(G, n_scans=3)
        self.assertEqual(m["gap_closed_count"], 1)
        self.assertEqual(m["total_tracks"], 2)  # Grain 1 (scans 0, 2) + isolated noise (scan 1)

    def test_gap_closing_composes_cleanly_before_split_resolution(self) -> None:
        """A track that recovers from a 1-scan dropout and later splits 1->2 resolves both events."""
        scans = [
            [_spot(1, 20.0, 20.0, 20.0, intensity=1000.0)],
            [],  # dropout at scan 1
            [_spot(1, 20.0, 20.0, 20.5, intensity=1000.0)],
            [
                _spot(1, 20.0, 18.0, 21.0, intensity=520.0),
                _spot(2, 20.0, 22.0, 21.0, intensity=480.0),
            ],
        ]
        G = build_tracks(
            scans,
            gap_closing_config=GapClosingConfig(max_gap=1, max_cost=20.0),
            split_merge_config=SplitMergeConfig(max_distance=15.0),
        )
        m = compute_tracking_metrics(G, n_scans=4)
        self.assertEqual(m["gap_closed_count"], 1)
        self.assertEqual(m["split_count"], 1)
        ev = G.graph["lineage_events"][0]
        self.assertEqual(ev.parent_track_ids, (1,))


if __name__ == "__main__":
    unittest.main()
