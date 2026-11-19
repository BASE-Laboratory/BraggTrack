"""Regression tests on the bundled ESRF-ID03 pco_nf dataset (data/sample_operando/)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from braggtrack.autonomous import CausalStreamingTracker
from braggtrack.io import (
    BeamlineAdapter,
    detector_foreground_threshold_chunked,
    load_primary_volume,
    open_lazy_volume,
    otsu_threshold_chunked,
    sample_operando_root,
)
from braggtrack.segmentation import detector_foreground_threshold
from braggtrack.tracking import PrecursorConfig, compute_fracture_precursors, compute_grain_kinematics


def test_bundled_esrf_id03_read_noise_floor_prevents_percolation() -> None:
    """detector_foreground_threshold suppresses 46% sCMOS read-noise percolation on real pco_nf cubes."""
    root = sample_operando_root()
    seq = BeamlineAdapter(Path(root)).build_sequence()
    scan0_path = seq.scans[0].file_path

    vol0 = load_primary_volume(scan0_path)
    with open_lazy_volume(scan0_path) as lazy_vol:
        raw_otsu = otsu_threshold_chunked(lazy_vol)
        chunked_guarded = detector_foreground_threshold_chunked(lazy_vol, bg_sigma=5.0)

    dense_guarded = detector_foreground_threshold(vol0, bg_sigma=5.0)
    assert chunked_guarded == pytest.approx(dense_guarded, abs=1.2)

    # Raw 3D Otsu splits the 180 ADU Gaussian read-noise peak above 3D percolation (p_c = 0.3116)
    assert float(np.mean(vol0 >= raw_otsu)) > 0.35
    assert float(np.mean(vol0[50] >= raw_otsu)) > 0.35
    # Guarded threshold isolates sparse Bragg reflections (< 1% of volume) and suppresses >99% of empty z=50
    assert 0.001 < float(np.mean(vol0 >= chunked_guarded)) < 0.01
    assert float(np.mean(vol0[50] >= chunked_guarded)) < 0.01


def test_bundled_esrf_id03_physical_tracking_invariants_and_slice_audit() -> None:
    """Verify physical tracking invariants (no teleports, no noise-speck links) and provisional slice audit."""
    root = sample_operando_root()
    seq = BeamlineAdapter(Path(root)).build_sequence()

    tracker = CausalStreamingTracker()
    for sc in seq.scans:
        sr = tracker.tell(sc.file_path)
        targets = tracker.ask(2, acquisition="ucb")
        assert 15 <= sr.n_spots <= 25
        assert len(targets) <= 2

    # No component may flood into a 100k+ voxel Voronoi camera-noise sponge
    max_voxels = max(int(r["voxel_count"]) for r in tracker.track_table)
    assert max_voxels < 15_000

    G = tracker.graph
    cont_edges = {(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type", "continuation") == "continuation"}

    # 1. Zero unphysical cross-rocking-layer teleports across all continuation edges
    max_dmu = max(abs(float(G.nodes[v]["centroid_mu"]) - float(G.nodes[u]["centroid_mu"])) for u, v in cont_edges)
    assert max_dmu < 3.0

    # 2. Single-slice sCMOS hot-pixel / cosmic-ray specks (8-15 voxels, eig_3 ~ 0) must remain unlinked
    noise_specks = {
        nid for nid, d in G.nodes(data=True) if int(d["voxel_count"]) <= 15 and float(d.get("eig_3", 0.0)) < 1e-6
    }
    assert len(noise_specks) == 5
    assert all(u not in noise_specks and v not in noise_specks for u, v in cont_edges)

    # 3. Provisional slice-inspection regression guard on isolated reflections (mu~19.7, 21.0, 74.0-74.8):
    #    Guards against Hungarian 1e18 sentinel cascades; NOT an independent human ground-truth benchmark.
    isolated_slice_audit_links = {
        ("s0_i2", "s1_i2"),
        ("s1_i2", "s2_i3"),
        ("s0_i13", "s1_i7"),
        ("s1_i7", "s2_i10"),
        ("s1_i17", "s2_i16"),
        ("s0_i1", "s1_i1"),
        ("s1_i1", "s2_i1"),
        ("s0_i3", "s1_i3"),
        ("s1_i3", "s2_i2"),
        ("s0_i4", "s1_i4"),
        ("s1_i4", "s2_i4"),
    }
    assert isolated_slice_audit_links.issubset(cont_edges)

    # 4. Classical default (cost_beta=0.0) matches hybrid mock-descriptor edges on real scans
    tracker_hybrid = CausalStreamingTracker(backend="mock", cost_beta=0.25)
    for sc in seq.scans:
        tracker_hybrid.tell(sc.file_path)
    hybrid_edges = {
        (u, v)
        for u, v, d in tracker_hybrid.graph.edges(data=True)
        if d.get("edge_type", "continuation") == "continuation"
    }
    assert cont_edges == hybrid_edges


def test_pancake_spot_and_single_slice_eigenvalue_stability() -> None:
    """Steady near-field pancake reflections and 1-slice spots neither warn nor saturate precursor scores."""
    pancake_rows = [
        {
            "track_id": 1,
            "scan_idx": s,
            "label": 1,
            "centroid_mu": 74.8,
            "centroid_chi": 180.0 + 5.5 * s,
            "centroid_d": 99.3,
            "integrated_intensity": 580_000.0,
            "voxel_count": 2200,
            "cov_zz": 0.45,
            "cov_xx": 125.0,
            "cov_yy": 110.0,
            "eig_1": 125.0,
            "eig_2": 110.0,
            "eig_3": 0.45,
            "bbox_min_z": 73,
            "bbox_max_z": 76,
            "bbox_min_x": 160 + 5 * s,
            "bbox_max_x": 200 + 5 * s,
            "bbox_min_y": 78,
            "bbox_max_y": 120,
        }
        for s in range(3)
    ]
    candidates, obs_scores = compute_fracture_precursors(pancake_rows, PrecursorConfig(score_threshold=0.30))
    assert candidates == []
    assert all(sc.anisotropy_term == pytest.approx(0.0, abs=1e-6) for sc in obs_scores)

    # Single-z-slice reflection (eig_3 == 0.0) must not emit divide-by-zero RuntimeWarning
    one_slice = [dict(pancake_rows[0], cov_zz=0.0, eig_3=0.0, vox_eig_3=0.0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kin = compute_grain_kinematics(one_slice)
    assert len(kin) == 1
    assert kin[0].anisotropy[0] == pytest.approx(1.0)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
