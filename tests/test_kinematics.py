"""Tests for per-grain kinematic time-series."""

import unittest

import numpy as np
from conftest import make_spot as _spot

from braggtrack.tracking.cost import PositionShapeCost
from braggtrack.tracking.kinematics import (
    compute_grain_kinematics,
    kinematics_to_table,
    summarize_kinematics,
)
from braggtrack.tracking.lifecycle import build_tracks, tracks_to_table


def _build_table(scan_tables: list[list[dict]]) -> list[dict]:
    cost_fn = PositionShapeCost()
    G = build_tracks(scan_tables, cost_fn)
    return tracks_to_table(G)


class TestComputeGrainKinematics(unittest.TestCase):
    def test_single_track_three_scans(self) -> None:
        table = _build_table(
            [
                [_spot(1.0, 2.0, 10.0, intensity=100, voxels=20)],
                [_spot(1.1, 2.2, 10.5, intensity=120, voxels=25)],
                [_spot(1.3, 2.5, 11.0, intensity=150, voxels=30)],
            ]
        )
        results = compute_grain_kinematics(table)
        self.assertEqual(len(results), 1)
        gk = results[0]
        self.assertEqual(len(gk.scan_indices), 3)
        self.assertEqual(gk.scan_indices, [0, 1, 2])

    def test_strain_from_d_spacing(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10.0)],
                [_spot(0, 0, 10.5)],
                [_spot(0, 0, 11.0)],
            ]
        )
        gk = compute_grain_kinematics(table)[0]
        np.testing.assert_allclose(gk.strain, [0.0, 0.05, 0.10], atol=1e-10)

    def test_misorientation_from_mu_chi(self) -> None:
        table = _build_table(
            [
                [_spot(0.0, 0.0, 10.0)],
                [_spot(3.0, 4.0, 10.0)],
            ]
        )
        gk = compute_grain_kinematics(table)[0]
        np.testing.assert_allclose(gk.misorientation_mu, [0.0, 3.0])
        np.testing.assert_allclose(gk.misorientation_chi, [0.0, 4.0])
        np.testing.assert_allclose(gk.misorientation_total, [0.0, 5.0])

    def test_relative_intensity_and_volume(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10, intensity=100, voxels=20)],
                [_spot(0, 0, 10, intensity=200, voxels=40)],
            ]
        )
        gk = compute_grain_kinematics(table)[0]
        np.testing.assert_allclose(gk.relative_intensity, [0.0, 1.0])
        np.testing.assert_allclose(gk.relative_volume, [0.0, 1.0])

    def test_anisotropy(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10, eig=(3.0, 2.0, 1.0))],
                [_spot(0, 0, 10, eig=(6.0, 2.0, 1.0))],
            ]
        )
        gk = compute_grain_kinematics(table)[0]
        np.testing.assert_allclose(gk.anisotropy, [3.0, 6.0])

    def test_covariance_trace(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10, eig=(3.0, 2.0, 1.0))],
            ]
        )
        gk = compute_grain_kinematics(table)[0]
        np.testing.assert_allclose(gk.covariance_trace, [6.0])

    def test_multiple_tracks(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10), _spot(50, 50, 50)],
                [_spot(0.1, 0.1, 10.1), _spot(50.1, 50.1, 50.1)],
            ]
        )
        results = compute_grain_kinematics(table)
        self.assertEqual(len(results), 2)
        tids = [gk.track_id for gk in results]
        self.assertEqual(len(set(tids)), 2)

    def test_birth_mid_sequence(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10)],
                [_spot(0.1, 0.1, 10.1), _spot(50, 50, 50)],
                [_spot(0.2, 0.2, 10.2), _spot(50.1, 50.1, 50.1)],
            ]
        )
        results = compute_grain_kinematics(table)
        self.assertEqual(len(results), 2)
        lengths = sorted(len(gk.scan_indices) for gk in results)
        self.assertEqual(lengths, [2, 3])

    def test_empty_table(self) -> None:
        results = compute_grain_kinematics([])
        self.assertEqual(results, [])

    def test_zero_d_spacing_no_crash(self) -> None:
        table = [
            {
                "track_id": 1,
                "scan_idx": 0,
                "centroid_mu": 0,
                "centroid_chi": 0,
                "centroid_d": 0.0,
                "eig_1": 1,
                "eig_2": 1,
                "eig_3": 1,
                "integrated_intensity": 100,
                "voxel_count": 10,
            },
        ]
        results = compute_grain_kinematics(table)
        self.assertEqual(len(results), 1)
        np.testing.assert_allclose(results[0].strain, [0.0])


class TestSummarizeKinematics(unittest.TestCase):
    def test_summary_fields(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10, intensity=100, voxels=20, eig=(3, 2, 1))],
                [_spot(0.5, 0.3, 10.2, intensity=150, voxels=30, eig=(4, 2, 1))],
            ]
        )
        gk = compute_grain_kinematics(table)
        summary = summarize_kinematics(gk, n_scans=2)
        self.assertEqual(summary.n_tracks, 1)
        self.assertEqual(summary.n_full_tracks, 1)
        self.assertEqual(summary.n_scans, 2)
        self.assertEqual(len(summary.max_strain), 1)
        self.assertGreater(summary.max_strain[0], 0)
        self.assertGreater(summary.total_misorientation[0], 0)
        self.assertGreater(summary.intensity_change_frac[0], 0)
        self.assertGreater(summary.volume_change_frac[0], 0)

    def test_full_track_count(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10)],
                [_spot(0.1, 0.1, 10.1), _spot(50, 50, 50)],
                [_spot(0.2, 0.2, 10.2), _spot(50.1, 50.1, 50.1)],
            ]
        )
        gk = compute_grain_kinematics(table)
        summary = summarize_kinematics(gk, n_scans=3)
        self.assertEqual(summary.n_full_tracks, 1)
        self.assertEqual(summary.n_tracks, 2)


class TestKinematicsToTable(unittest.TestCase):
    def test_row_count(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10), _spot(50, 50, 50)],
                [_spot(0.1, 0.1, 10.1), _spot(50.1, 50.1, 50.1)],
            ]
        )
        gk = compute_grain_kinematics(table)
        rows = kinematics_to_table(gk)
        self.assertEqual(len(rows), 4)

    def test_required_columns(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10)],
                [_spot(0.1, 0.1, 10.1)],
            ]
        )
        gk = compute_grain_kinematics(table)
        rows = kinematics_to_table(gk)
        expected = {
            "track_id",
            "scan_idx",
            "centroid_mu",
            "centroid_chi",
            "centroid_d",
            "strain",
            "misorientation_mu",
            "misorientation_chi",
            "misorientation_total",
            "integrated_intensity",
            "voxel_count",
            "relative_intensity",
            "relative_volume",
            "eig_1",
            "eig_2",
            "eig_3",
            "anisotropy",
            "covariance_trace",
        }
        self.assertEqual(set(rows[0].keys()), expected)

    def test_values_match_grain_kinematics(self) -> None:
        table = _build_table(
            [
                [_spot(0, 0, 10.0)],
                [_spot(0, 0, 10.5)],
            ]
        )
        gk = compute_grain_kinematics(table)
        rows = kinematics_to_table(gk)
        self.assertAlmostEqual(rows[1]["strain"], 0.05)


if __name__ == "__main__":
    unittest.main()
