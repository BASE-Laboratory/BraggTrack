"""Tests for analytical centroid/strain uncertainty, lineage strain, and assignment margins."""

from __future__ import annotations

import math

import numpy as np

from braggtrack.io.models import AxisCalibration
from braggtrack.segmentation.features import extract_instance_table
from braggtrack.tracking import (
    PositionShapeCost,
    SplitMergeConfig,
    build_tracks,
    compute_grain_kinematics,
    kinematics_to_table,
    tracks_to_table,
)


def test_poisson_centroid_and_strain_error_match_monte_carlo() -> None:
    """Analytical centroid_d_err and strain_err match 1,000-trial Poisson Monte Carlo within <10%."""
    rng = np.random.default_rng(42)
    zz, yy, xx = np.mgrid[0:15, 0:15, 0:15]
    sigma = 1.8
    cal = AxisCalibration(
        mu_origin=10.0,
        mu_step=0.05,
        chi_origin=-2.0,
        chi_step=0.04,
        d_origin=1.35,
        d_step=0.002,
        calibrated=True,
    )

    # Expected photon distributions for scan 0 (d_center=7.0) and scan 1 (d_center=8.2)
    lam0 = 2500.0 * np.exp(-((zz - 7.0) ** 2 + (yy - 7.0) ** 2 + (xx - 7.0) ** 2) / (2.0 * sigma**2))
    lam1 = 2500.0 * np.exp(-((zz - 7.0) ** 2 + (yy - 8.2) ** 2 + (xx - 7.0) ** 2) / (2.0 * sigma**2))
    mask0 = (lam0 > 1.0).astype(np.int32)
    mask1 = (lam1 > 1.0).astype(np.int32)

    mc_d0: list[float] = []
    mc_strain1: list[float] = []
    pred_d0_errs: list[float] = []
    pred_strain1_errs: list[float] = []

    for trial in range(1000):
        v0 = rng.poisson(lam0).astype(np.float64)
        v1 = rng.poisson(lam1).astype(np.float64)
        r0 = extract_instance_table(mask0, v0, calibration=cal)[0]
        r1 = extract_instance_table(mask1, v1, calibration=cal)[0]
        mc_d0.append(float(r0["centroid_d"]))
        mc_strain1.append((float(r1["centroid_d"]) - float(r0["centroid_d"])) / float(r0["centroid_d"]))
        if trial < 50:
            G = build_tracks([[r0], [r1]])
            gk = compute_grain_kinematics(tracks_to_table(G))[0]
            pred_d0_errs.append(float(gk.centroid_d_err[0]))
            pred_strain1_errs.append(float(gk.strain_err[1]))
            assert gk.strain_err[0] == 0.0

    emp_d0_std = float(np.std(mc_d0, ddof=1))
    emp_strain_std = float(np.std(mc_strain1, ddof=1))
    mean_pred_d0 = float(np.mean(pred_d0_errs))
    mean_pred_strain = float(np.mean(pred_strain1_errs))

    assert abs(mean_pred_d0 - emp_d0_std) / emp_d0_std < 0.08
    assert abs(mean_pred_strain - emp_strain_std) / emp_strain_std < 0.08


def test_ancestor_referenced_lineage_strain_and_causal_rates_across_split() -> None:
    """Post-split daughter tracks retain root ancestor d0 in lineage_strain and causal rates."""

    # Scan 0 -> Scan 1: parent grain contracts slightly (1.420 -> 1.414 A)
    # Scan 1 -> Scan 2: grain splits 1 -> 2 into H3-collapsed (1.390 A) and H2-pinned (1.413 A)
    def _spot(label: int, mu: float, chi: float, d: float, intensity: float, d_err: float = 1e-4) -> dict:
        return {
            "label": label,
            "voxel_count": 30,
            "integrated_intensity": intensity,
            "centroid_mu": mu,
            "centroid_chi": chi,
            "centroid_d": d,
            "centroid_mu_err": 1e-3,
            "centroid_chi_err": 1e-3,
            "centroid_d_err": d_err,
            "eig_1": 1.2,
            "eig_2": 1.0,
            "eig_3": 0.8,
        }

    scan0 = [_spot(1, 10.0, 5.0, 1.420, 1000.0)]
    scan1 = [_spot(1, 10.1, 5.0, 1.414, 1000.0)]
    scan2 = [
        _spot(1, 10.2, 4.7, 1.390, 550.0),
        _spot(2, 10.2, 5.3, 1.413, 450.0),
    ]
    scan3 = [
        _spot(1, 10.3, 4.6, 1.385, 540.0),
        _spot(2, 10.3, 5.4, 1.412, 460.0),
    ]

    G = build_tracks(
        [scan0, scan1, scan2, scan3],
        cost_fn=PositionShapeCost(),
        split_merge_config=SplitMergeConfig(max_distance=5.0),
    )
    events = G.graph["lineage_events"]
    assert len(events) == 1
    assert events[0].event_type == "split"

    gks = {gk.track_id: gk for gk in compute_grain_kinematics(tracks_to_table(G))}
    parent_tid = events[0].parent_track_ids[0]
    c1_tid, c2_tid = events[0].child_track_ids

    for dtid in (c1_tid, c2_tid):
        dgk = gks[dtid]
        assert dgk.root_track_id == parent_tid
        # Segment-local strain resets at cleavage (scan 2), whereas lineage_strain preserves root d0=1.420
        assert dgk.strain[0] == 0.0
        expected_root_strain = (dgk.centroid_d[0] - 1.420) / 1.420
        assert abs(dgk.lineage_strain[0] - expected_root_strain) < 1e-9
        assert dgk.lineage_strain_err[0] > 0.0
        # Causal rate at split scan (i=0) uses immediate parent's scan-1 d=1.414
        expected_rate_d0 = dgk.centroid_d[0] - 1.414
        assert abs(dgk.rate_d[0] - expected_rate_d0) < 1e-9

    flat = kinematics_to_table(list(gks.values()))
    assert all("lineage_strain" in r and "strain_err" in r and "root_track_id" in r for r in flat)


def test_assignment_margin_distinguishes_crowded_from_isolated() -> None:
    """Ambiguous nearby neighbours yield small assignment_margin vs inf when isolated."""
    s0_iso = [
        {"label": 1, "centroid_mu": 0.0, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1, "eig_2": 1, "eig_3": 1}
    ]
    s1_iso = [
        {"label": 1, "centroid_mu": 0.2, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1, "eig_2": 1, "eig_3": 1}
    ]
    G_iso = build_tracks([s0_iso, s1_iso])
    assert math.isinf(G_iso.nodes["s1_i0"]["assignment_margin"])

    s0_crowd = [
        {"label": 1, "centroid_mu": 0.0, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1, "eig_2": 1, "eig_3": 1},
        {"label": 2, "centroid_mu": 1.0, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1, "eig_2": 1, "eig_3": 1},
    ]
    s1_crowd = [
        {"label": 1, "centroid_mu": 0.4, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1, "eig_2": 1, "eig_3": 1},
        {"label": 2, "centroid_mu": 1.0, "centroid_chi": 0.0, "centroid_d": 0.0, "eig_1": 1, "eig_2": 1, "eig_3": 1},
    ]
    G_crowd = build_tracks([s0_crowd, s1_crowd])
    margin_crowd = G_crowd.nodes["s1_i0"]["assignment_margin"]
    assert math.isfinite(margin_crowd)
    assert 0.0 < margin_crowd < 1.0
