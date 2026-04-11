"""Unit tests for the Week 3 tracking module."""

import math
import unittest

import numpy as np

from braggtrack.tracking.cost import PositionShapeCost
from braggtrack.tracking.assignment import associate_frames
from braggtrack.tracking.lifecycle import TrackEvent, build_tracks, tracks_to_table
from braggtrack.tracking.metrics import compute_tracking_metrics
from braggtrack.tracking.synthetic import generate_crossing_scenario


def _spot(mu: float, chi: float, d: float,
          eig: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> dict:
    return {
        "label": 1, "voxel_count": 10, "integrated_intensity": 100.0,
        "centroid_mu": mu, "centroid_chi": chi, "centroid_d": d,
        "eig_1": eig[0], "eig_2": eig[1], "eig_3": eig[2],
    }


class TestPositionShapeCost(unittest.TestCase):
    def test_pairwise_matrix_matches_scalar(self) -> None:
        rng = np.random.RandomState(7)
        spots_t = [_spot(float(rng.randn()), float(rng.randn()), float(rng.randn())) for _ in range(6)]
        spots_t1 = [_spot(float(rng.randn()), float(rng.randn()), float(rng.randn())) for _ in range(8)]
        fn = PositionShapeCost(
            position_weight=1.1,
            shape_weight=0.4,
            gate_mu=2.5,
            gate_chi=1.8,
            gate_d=3.2,
        )
        mat = fn.pairwise_cost_matrix(spots_t, spots_t1)
        self.assertEqual(mat.shape, (6, 8))
        for i, si in enumerate(spots_t):
            for j, sj in enumerate(spots_t1):
                c = fn(si, sj)
                if math.isfinite(c):
                    self.assertAlmostEqual(float(mat[i, j]), c, places=10)
                else:
                    self.assertFalse(math.isfinite(float(mat[i, j])))

    def test_identical_spots_zero_cost(self) -> None:
        cost_fn = PositionShapeCost()
        s = _spot(1.0, 2.0, 3.0)
        self.assertAlmostEqual(cost_fn(s, s), 0.0)

    def test_position_distance(self) -> None:
        cost_fn = PositionShapeCost(position_weight=1.0, shape_weight=0.0)
        a = _spot(0.0, 0.0, 0.0)
        b = _spot(3.0, 4.0, 0.0)
        self.assertAlmostEqual(cost_fn(a, b), 25.0)  # 9 + 16

    def test_gate_rejects(self) -> None:
        cost_fn = PositionShapeCost(gate_mu=2.0)
        a = _spot(0.0, 0.0, 0.0)
        b = _spot(5.0, 0.0, 0.0)
        self.assertEqual(cost_fn(a, b), math.inf)

    def test_gate_accepts(self) -> None:
        cost_fn = PositionShapeCost(gate_mu=10.0)
        a = _spot(0.0, 0.0, 0.0)
        b = _spot(5.0, 0.0, 0.0)
        self.assertTrue(math.isfinite(cost_fn(a, b)))

    def test_per_axis_gating(self) -> None:
        cost_fn = PositionShapeCost(gate_mu=10.0, gate_chi=1.0, gate_d=10.0)
        a = _spot(0.0, 0.0, 0.0)
        b = _spot(0.0, 5.0, 0.0)  # chi exceeds gate
        self.assertEqual(cost_fn(a, b), math.inf)


class TestAssignment(unittest.TestCase):
    def test_perfect_match(self) -> None:
        spots_t = [_spot(1, 1, 1), _spot(10, 10, 10)]
        spots_t1 = [_spot(1.1, 1.1, 1.1), _spot(10.1, 10.1, 10.1)]
        cost_fn = PositionShapeCost()
        matches, unmatched_t, unmatched_t1 = associate_frames(
            spots_t, spots_t1, cost_fn)
        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched_t, [])
        self.assertEqual(unmatched_t1, [])
        # Verify correct assignment (not swapped)
        match_dict = dict(matches)
        self.assertEqual(match_dict[0], 0)
        self.assertEqual(match_dict[1], 1)

    def test_birth_and_death(self) -> None:
        spots_t = [_spot(1, 1, 1), _spot(10, 10, 10)]
        spots_t1 = [_spot(1.1, 1.1, 1.1), _spot(50, 50, 50)]
        cost_fn = PositionShapeCost(gate_mu=5.0, gate_chi=5.0, gate_d=5.0)
        matches, unmatched_t, unmatched_t1 = associate_frames(
            spots_t, spots_t1, cost_fn)
        self.assertEqual(len(matches), 1)
        self.assertIn(1, unmatched_t)    # terminated
        self.assertIn(1, unmatched_t1)   # born

    def test_empty_frames(self) -> None:
        cost_fn = PositionShapeCost()
        matches, ut, ut1 = associate_frames([], [_spot(1, 1, 1)], cost_fn)
        self.assertEqual(len(matches), 0)
        self.assertEqual(ut1, [0])

        matches, ut, ut1 = associate_frames([_spot(1, 1, 1)], [], cost_fn)
        self.assertEqual(len(matches), 0)
        self.assertEqual(ut, [0])

    def test_max_cost_gate(self) -> None:
        spots_t = [_spot(0, 0, 0)]
        spots_t1 = [_spot(100, 100, 100)]
        cost_fn = PositionShapeCost()
        matches, ut, ut1 = associate_frames(spots_t, spots_t1, cost_fn, max_cost=10.0)
        self.assertEqual(len(matches), 0)


class TestLifecycle(unittest.TestCase):
    def test_three_frame_continuity(self) -> None:
        tables = [
            [_spot(1, 1, 1)],
            [_spot(1.1, 1.1, 1.1)],
            [_spot(1.2, 1.2, 1.2)],
        ]
        cost_fn = PositionShapeCost()
        G = build_tracks(tables, cost_fn)
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 2)
        # Single track across 3 scans
        track_ids = {G.nodes[n]["track_id"] for n in G.nodes}
        self.assertEqual(len(track_ids), 1)

    def test_birth_event(self) -> None:
        tables = [
            [_spot(1, 1, 1)],
            [_spot(1.1, 1.1, 1.1), _spot(50, 50, 50)],
        ]
        cost_fn = PositionShapeCost()
        G = build_tracks(tables, cost_fn)
        # Node for spot at (50,50,50) should be BORN
        born_nodes = [n for n in G.nodes if G.nodes[n]["event"] == TrackEvent.BORN
                      and G.nodes[n]["scan_idx"] == 1]
        self.assertEqual(len(born_nodes), 1)

    def test_termination_event(self) -> None:
        tables = [
            [_spot(1, 1, 1), _spot(50, 50, 50)],
            [_spot(1.1, 1.1, 1.1)],
        ]
        cost_fn = PositionShapeCost()
        G = build_tracks(tables, cost_fn)
        terminated = [n for n in G.nodes if G.nodes[n]["event"] == TrackEvent.TERMINATED]
        self.assertEqual(len(terminated), 1)

    def test_tracks_to_table_output(self) -> None:
        tables = [
            [_spot(1, 1, 1)],
            [_spot(1.1, 1.1, 1.1)],
        ]
        cost_fn = PositionShapeCost()
        G = build_tracks(tables, cost_fn)
        rows = tracks_to_table(G)
        self.assertEqual(len(rows), 2)
        self.assertIn("track_id", rows[0])
        self.assertIn("event", rows[0])


class TestMetrics(unittest.TestCase):
    def test_perfect_tracking_no_fragmentation(self) -> None:
        tables = [
            [_spot(1, 1, 1), _spot(10, 10, 10)],
            [_spot(1.1, 1.1, 1.1), _spot(10.1, 10.1, 10.1)],
            [_spot(1.2, 1.2, 1.2), _spot(10.2, 10.2, 10.2)],
        ]
        cost_fn = PositionShapeCost()
        G = build_tracks(tables, cost_fn)
        m = compute_tracking_metrics(G, n_scans=3)
        self.assertEqual(m["total_tracks"], 2)
        self.assertEqual(m["full_length_tracks"], 2)
        self.assertAlmostEqual(m["fragmentation_ratio"], 0.0)

    def test_fragmentation_from_birth(self) -> None:
        tables = [
            [_spot(1, 1, 1)],
            [_spot(1.1, 1.1, 1.1), _spot(50, 50, 50)],
            [_spot(1.2, 1.2, 1.2), _spot(50.1, 50.1, 50.1)],
        ]
        cost_fn = PositionShapeCost()
        G = build_tracks(tables, cost_fn)
        m = compute_tracking_metrics(G, n_scans=3)
        self.assertEqual(m["total_tracks"], 2)
        self.assertEqual(m["full_length_tracks"], 1)  # only track 1 spans all 3
        self.assertGreater(m["fragmentation_ratio"], 0.0)


class TestSyntheticScenario(unittest.TestCase):
    def test_crossing_scenario_structure(self) -> None:
        tables, gt = generate_crossing_scenario(n_scans=3, seed=42)
        self.assertEqual(len(tables), 3)
        # Scan 0: spots 1,2,3,5 = 4 spots
        self.assertEqual(len(tables[0]), 4)
        # Scan 1: spots 1,2,3,4 = 4 spots
        self.assertEqual(len(tables[1]), 4)
        # Scan 2: spots 1,2,3,4,6 = 5 spots
        self.assertEqual(len(tables[2]), 5)
        self.assertTrue(len(gt) > 0)

    def test_crossing_scenario_tracking(self) -> None:
        tables, gt = generate_crossing_scenario(n_scans=3, seed=42)
        cost_fn = PositionShapeCost(shape_weight=0.5)
        G = build_tracks(tables, cost_fn)
        m = compute_tracking_metrics(G, n_scans=3, ground_truth=gt)
        self.assertGreater(m["total_tracks"], 0)
        self.assertIn("id_switch_rate", m)

    def test_crossing_with_ground_truth_switches(self) -> None:
        """Spots 1 and 2 cross paths — verify we can measure ID switches."""
        tables, gt = generate_crossing_scenario(n_scans=3, seed=42)
        cost_fn = PositionShapeCost(position_weight=1.0, shape_weight=0.0)
        G = build_tracks(tables, cost_fn)
        m = compute_tracking_metrics(G, n_scans=3, ground_truth=gt)
        # With shape_weight=0 and crossing trajectories, there may be switches
        self.assertIsInstance(m["id_switch_count"], int)


if __name__ == '__main__':
    unittest.main()
