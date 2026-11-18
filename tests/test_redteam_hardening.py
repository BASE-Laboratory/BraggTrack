"""Red-Team hardening and Facility Autonomy Levels (FAL Level 3) verification suite."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from braggtrack.autonomous import (
    BulkReferencePrior,
    CausalStreamingTracker,
    OperationalDesignDomain,
    ZoomTriggerPolicy,
    compute_sim_to_real_gap,
)
from braggtrack.cli import run_pipeline, watch_autonomous
from braggtrack.io import AxisCalibration, ScanStimulusPoint
from braggtrack.segmentation import extract_instance_table, segment_classical
from braggtrack.semantic.dino import _mips_to_rgb_uint8
from braggtrack.tracking.cost import PositionShapeCost
from braggtrack.tracking.events import SplitMergeConfig
from braggtrack.tracking.gap_closing import GapClosingConfig
from braggtrack.tracking.kinematics import compute_grain_kinematics
from braggtrack.tracking.lifecycle import build_tracks, tracks_to_table
from braggtrack.viz.html_report import generate_html_report


def _add_spot(
    vol: np.ndarray,
    center_zyx: tuple[float, float, float],
    sigma_zyx: tuple[float, float, float],
    amplitude: float,
) -> None:
    zz, yy, xx = np.meshgrid(
        np.arange(vol.shape[0], dtype=np.float64),
        np.arange(vol.shape[1], dtype=np.float64),
        np.arange(vol.shape[2], dtype=np.float64),
        indexing="ij",
    )
    cz, cy, cx = center_zyx
    sz, sy, sx = sigma_zyx
    vol += amplitude * np.exp(-0.5 * (((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def test_axis_calibration_scale_invariance_tracking_gaps_and_splits() -> None:
    """Calibrated (deg, deg, A) runs must produce identical tracks, gap closures, and splits to voxel space."""
    shape = (24, 36, 36)
    vols: list[np.ndarray] = []
    for t in range(5):
        v = np.zeros(shape, dtype=np.float32)
        # Grain 1: drops out at scan 2 (gap=2), continues at scans 3..4
        if t != 2:
            _add_spot(v, (7.0 + 0.3 * t, 10.0 - 0.5 * t, 9.0 + 0.4 * t), (1.2, 1.2, 1.2), 240.0)
        # Grain 2: cleaves 1 -> 2 along chi at scan 3
        if t < 3:
            _add_spot(v, (16.0, 24.0 - 0.4 * t, 20.0), (1.2, 1.2, 1.6), 280.0)
        else:
            _add_spot(v, (16.0, 24.0 - 0.4 * t, 16.0), (1.2, 1.2, 1.2), 145.0)
            _add_spot(v, (16.0, 24.0 - 0.4 * t, 24.0), (1.2, 1.2, 1.2), 135.0)
        vols.append(v)

    cal = AxisCalibration(
        mu_origin=12.5,
        mu_step=0.02,
        chi_origin=-3.0,
        chi_step=0.035,
        d_origin=2.15,
        d_step=-0.0004,
        calibrated=True,
    )

    tables_vox: list[list[dict]] = []
    tables_cal: list[list[dict]] = []
    for v in vols:
        seg = segment_classical(v, threshold=15.0, seed_response_percentile=0.0)
        tables_vox.append(extract_instance_table(seg.labeled_volume, v, calibration=None))
        tables_cal.append(extract_instance_table(seg.labeled_volume, v, calibration=cal))

    g_vox = build_tracks(
        tables_vox,
        cost_fn=PositionShapeCost(),
        gap_closing_config=GapClosingConfig(max_gap=2),
        split_merge_config=SplitMergeConfig(max_distance=20.0),
    )
    g_cal = build_tracks(
        tables_cal,
        cost_fn=PositionShapeCost(),
        gap_closing_config=GapClosingConfig(max_gap=2),
        split_merge_config=SplitMergeConfig(max_distance=20.0),
    )

    assert g_cal.graph["gap_closed_count"] == 1
    assert g_cal.graph["gap_closed_count"] == g_vox.graph["gap_closed_count"]

    ev_vox = g_vox.graph["lineage_events"]
    ev_cal = g_cal.graph["lineage_events"]
    assert len(ev_cal) == 1 and len(ev_vox) == 1
    assert ev_cal[0].event_type == "split"
    assert ev_cal[0].parent_track_ids == ev_vox[0].parent_track_ids
    assert ev_cal[0].child_track_ids == ev_vox[0].child_track_ids

    kin_vox = {k.track_id: k for k in compute_grain_kinematics(tracks_to_table(g_vox))}
    kin_cal = {k.track_id: k for k in compute_grain_kinematics(tracks_to_table(g_cal))}
    assert set(kin_vox.keys()) == set(kin_cal.keys())
    for tid in kin_vox:
        np.testing.assert_allclose(kin_cal[tid].anisotropy, kin_vox[tid].anisotropy, rtol=1e-5, atol=1e-5)


def test_mechanical_vs_phase_split_classification_and_wallclock_strain_rate() -> None:
    """Split resolver must disambiguate orientation cleavage (mechanical_split) from d-jump (phase_split)."""
    cal = AxisCalibration(
        mu_origin=0.0, mu_step=0.02, chi_origin=0.0, chi_step=0.02, d_origin=2.0, d_step=0.001, calibrated=True
    )
    shape = (24, 40, 40)
    v0 = np.zeros(shape, dtype=np.float32)
    v1 = np.zeros(shape, dtype=np.float32)

    # Grain A at (8, 12, 12) -> splits along chi (x) at constant d (y=12): mechanical_split
    _add_spot(v0, (8.0, 12.0, 12.0), (1.2, 1.2, 1.2), 260.0)
    _add_spot(v1, (8.0, 12.0, 8.0), (1.2, 1.2, 1.2), 135.0)
    _add_spot(v1, (8.0, 12.0, 16.0), (1.2, 1.2, 1.2), 125.0)

    # Grain B at (16, 28, 28) -> splits along d (y=24 and y=32) at constant mu, chi: phase_split (H2 -> H3)
    _add_spot(v0, (16.0, 28.0, 28.0), (1.2, 1.2, 1.2), 260.0)
    _add_spot(v1, (16.0, 24.0, 28.0), (1.2, 1.2, 1.2), 130.0)
    _add_spot(v1, (16.0, 32.0, 28.0), (1.2, 1.2, 1.2), 130.0)

    t0 = extract_instance_table(
        segment_classical(v0, threshold=15.0, seed_response_percentile=0.0).labeled_volume, v0, calibration=cal
    )
    t1 = extract_instance_table(
        segment_classical(v1, threshold=15.0, seed_response_percentile=0.0).labeled_volume, v1, calibration=cal
    )
    G = build_tracks([t0, t1], split_merge_config=SplitMergeConfig(max_distance=15.0))
    events = G.graph["lineage_events"]
    subtypes = {ev.split_subtype for ev in events}
    assert subtypes == {"mechanical_split", "phase_split"}

    # Verify non-uniform wall-clock strain_rate_per_s
    rows = [
        {
            "track_id": 1,
            "scan_idx": 0,
            "centroid_mu": 0.0,
            "centroid_chi": 0.0,
            "centroid_d": 2.000,
            "voxel_count": 20,
            "integrated_intensity": 100.0,
        },
        {
            "track_id": 1,
            "scan_idx": 1,
            "centroid_mu": 0.0,
            "centroid_chi": 0.0,
            "centroid_d": 1.990,
            "voxel_count": 20,
            "integrated_intensity": 100.0,
        },
        {
            "track_id": 1,
            "scan_idx": 2,
            "centroid_mu": 0.0,
            "centroid_chi": 0.0,
            "centroid_d": 1.970,
            "voxel_count": 20,
            "integrated_intensity": 100.0,
        },
    ]
    kin = compute_grain_kinematics(rows, scan_elapsed_s={0: 0.0, 1: 50.0, 2: 250.0})[0]
    # Scan 0->1: strain = -0.005 over 50s -> -1e-4 /s; Scan 1->2: delta_strain = -0.010 over 200s -> -5e-5 /s
    assert kin.strain_rate_per_s[1] == pytest.approx(-1.0e-4, rel=1e-6)
    assert kin.strain_rate_per_s[2] == pytest.approx(-5.0e-5, rel=1e-6)


def test_pedestal_subtraction_rocking_kurtosis_border_flags_and_float32_segmentation() -> None:
    """Verify background pedestal debiasing, rocking kurtosis g_2, border flags, float32 seg, and MIP letterboxing."""
    shape = (20, 28, 28)
    clean = np.zeros(shape, dtype=np.float32)
    # Asymmetric reflection with apex at z=3.0 whose footprint extends to z=0 rocking border
    _add_spot(clean, (3.0, 8.0, 8.0), (1.7, 1.3, 1.3), 180.0)
    # Interior bimodal platykurtic reflection (two close sub-lobes inside one mask)
    _add_spot(clean, (11.0, 18.0, 11.2), (1.2, 1.2, 1.0), 160.0)
    _add_spot(clean, (11.0, 18.0, 13.8), (1.2, 1.2, 1.0), 160.0)

    pedestal = clean + 150.0
    seg_clean = segment_classical(clean, threshold=10.0, seed_response_percentile=0.0)
    seg_ped = segment_classical(pedestal, threshold=160.0, seed_response_percentile=0.0)
    assert seg_ped.response.dtype == np.float32

    rows_clean = extract_instance_table(seg_clean.labeled_volume, clean)
    rows_ped = extract_instance_table(seg_clean.labeled_volume, pedestal, detector_gain=2.0)
    assert len(rows_clean) == 2

    # Pedestal subtraction preserves exact centroids and estimates ~150 ADU background
    for rc, rp in zip(rows_clean, rows_ped, strict=True):
        assert rp["background_level"] == pytest.approx(150.0, abs=1.5)
        assert rp["centroid_chi"] == pytest.approx(rc["centroid_chi"], abs=0.05)
        assert rp["net_intensity"] == pytest.approx(rc["integrated_intensity"], rel=0.05)

    border_spot = min(rows_ped, key=lambda r: r["centroid_mu"])
    bimodal_spot = max(rows_ped, key=lambda r: r["centroid_mu"])
    assert border_spot["touches_rocking_border"] is True
    assert bimodal_spot["touches_rocking_border"] is False
    assert bimodal_spot["kurtosis_chi"] < -0.4  # Platykurtic bimodal profile

    # Heavy-tailed leptokurtic profile (sharp Bragg core + directional diffuse stress-concentrator wings)
    halo_vol = np.zeros((28, 28, 28), dtype=np.float32)
    _add_spot(halo_vol, (14.0, 14.0, 14.0), (0.9, 0.9, 0.8), 500.0)
    _add_spot(halo_vol, (14.0, 14.0, 14.0), (1.0, 1.0, 3.2), 28.0)
    halo_mask = (halo_vol > 25.0).astype(np.int32)
    halo_row = extract_instance_table(halo_mask, halo_vol)[0]
    assert halo_row["kurtosis_chi"] > 1.2  # Leptokurtic heavy tails (LeHoux2026)

    # Aspect-ratio-preserving square letterbox padding for elongated MIPs
    elongated_mip = np.ones((6, 40), dtype=np.float32) * 100.0
    rgb_letterbox = _mips_to_rgb_uint8(elongated_mip, preserve_aspect=True)
    rgb_raw = _mips_to_rgb_uint8(elongated_mip, preserve_aspect=False)
    assert rgb_letterbox.shape[0] == rgb_letterbox.shape[1]
    assert rgb_letterbox.shape[0] >= 45  # Padded >= ceil(40 / 0.875) so DINO 224/256 center-crop clips nothing
    assert rgb_raw.shape == (6, 40, 3)


def test_fal_level3_sim_to_real_gap_calibrated_monte_carlo_and_unit_invariance() -> None:
    """Verify FAL Eq. (4) closed-form Sim-to-Real Gap E[Delta_gap]=0, chi^2 coverage, and unit invariance."""
    rng = np.random.default_rng(42)
    pred_mean = (12.4, -1.8, 2.045)
    pred_std = (0.04, 0.08, 0.0015)

    gaps: list[float] = []
    in_95_flags: list[bool] = []
    in_valid_flags: list[bool] = []
    for _ in range(3000):
        obs = (
            float(rng.normal(pred_mean[0], pred_std[0])),
            float(rng.normal(pred_mean[1], pred_std[1])),
            float(rng.normal(pred_mean[2], pred_std[2])),
        )
        g, _, c95, cval = compute_sim_to_real_gap(pred_mean, pred_std, obs)
        gaps.append(g)
        in_95_flags.append(c95)
        in_valid_flags.append(cval)

    assert float(np.mean(gaps)) == pytest.approx(0.0, abs=0.06)
    assert 0.935 <= float(np.mean(in_95_flags)) <= 0.965
    assert 0.982 <= float(np.mean(in_valid_flags)) <= 0.997

    # Exact affine/unit invariance between voxel space and calibrated physical units
    g_vox, m2_vox, _, _ = compute_sim_to_real_gap((10.0, 20.0, 30.0), (1.5, 2.0, 0.8), (11.2, 18.5, 30.6))
    scales = (0.02, 0.05, 0.0004)
    origins = (14.0, -3.5, 4.72)
    g_phys, m2_phys, _, _ = compute_sim_to_real_gap(
        tuple(o + s * x for o, s, x in zip(origins, scales, (10.0, 20.0, 30.0), strict=True)),  # type: ignore[arg-type]
        tuple(s * sig for s, sig in zip(scales, (1.5, 2.0, 0.8), strict=True)),  # type: ignore[arg-type]
        tuple(o + s * x for o, s, x in zip(origins, scales, (11.2, 18.5, 30.6), strict=True)),  # type: ignore[arg-type]
    )
    assert g_phys == pytest.approx(g_vox, abs=1e-10)
    assert m2_phys == pytest.approx(m2_vox, abs=1e-10)


def test_fal_level3_hypothesis_switching_odd_safety_and_beam_dump_suppression() -> None:
    """Verify FAL L3 Condition (iii) hypothesis dependence, ODD clamping/voltage halt, dose cap, and beam-dump suppression."""
    shape = (24, 44, 44)
    vols: list[np.ndarray] = []
    for t in range(4):
        v = np.full(shape, 0.2, dtype=np.float32)
        # Grain M (near z=6, y=10, x=12): progressive anisotropic orientation broadening, zero d-shift
        sx_m = 1.1 if t == 0 else (2.4 + 0.6 * t)
        _add_spot(v, (6.0, 10.0, 12.0), (1.1, 1.1, sx_m), 260.0)

        # Grain P (near z=17, y=34 - 2.2*t, x=30): spherical peak, fast phase-nucleation d-contraction
        _add_spot(v, (17.0, 34.0 - 2.2 * t, 30.0), (1.15, 1.15, 1.15), 260.0)

        # Grain D (near z=17, y=12, x=30): sharp core + sub-Otsu directional rocking wings -> high positive kurtosis g_2
        if t == 0:
            _add_spot(v, (17.0, 12.0, 30.0), (1.15, 1.15, 1.15), 260.0)
        else:
            _add_spot(v, (17.0, 12.0, 30.0), (0.95, 0.95, 0.85), 270.0)
            _add_spot(v, (17.0, 12.0, 30.0), (1.05, 3.0, 1.05), 22.0)
        vols.append(v)

    def _run_with_hypothesis(hyp: str) -> tuple[float, float, float]:
        pol = ZoomTriggerPolicy(
            hypothesis=hyp,  # type: ignore[arg-type]
            min_precursor_score=0.28,
            min_persistence=2,
            max_triggers_per_scan=1,
            cooldown_scans=5,
        )
        trk = CausalStreamingTracker(
            backend="mock",
            trigger_policy=pol,
            reference_prior=BulkReferencePrior(kurtosis_mean=0.0, kurtosis_std=0.20),
        )
        triggered = [tgt for t in range(3) for tgt in trk.step(vols[t]).zoom_targets]
        assert len(triggered) >= 1
        return triggered[-1].current_centroid

    c_mech = _run_with_hypothesis("mechanical_fracture")
    c_phase = _run_with_hypothesis("phase_nucleation")
    c_dis = _run_with_hypothesis("intragranular_disorder")

    # Different declared hypotheses steer DFXM zoom to the three physically distinct grains!
    assert abs(c_mech[0] - 6.0) < 1.5  # Grain M at mu ~ 6
    assert abs(c_phase[0] - 17.0) < 1.5 and c_phase[2] > 28.0  # Grain P at (mu=17, d ~ 29.6..31.8)
    assert abs(c_dis[0] - 17.0) < 1.5 and abs(c_dis[2] - 12.0) < 1.5  # Grain D at (mu=17, d=12)

    # Test ODD motor soft-limit clamping, max_zooms_per_lineage=1, voltage ODD halt, and beam dump suppression
    odd = OperationalDesignDomain(
        mu_bounds=(0.0, 5.5),
        voltage_bounds_v=(2.8, 4.40),
        max_zooms_per_lineage=1,
    )
    pol_active = ZoomTriggerPolicy(
        hypothesis="mechanical_fracture",
        execution_mode="active",
        min_precursor_score=0.28,
        min_persistence=2,
        cooldown_scans=1,
        odd=odd,
    )
    trk_odd = CausalStreamingTracker(backend="mock", trigger_policy=pol_active)
    trk_odd.step(vols[0], stimulus=ScanStimulusPoint(scan_idx=0, scan_name="s0", elapsed_s=0.0, voltage_v=4.05))
    trk_odd.step(vols[1], stimulus=ScanStimulusPoint(scan_idx=1, scan_name="s1", elapsed_s=60.0, voltage_v=4.15))
    s2 = trk_odd.step(vols[2], stimulus=ScanStimulusPoint(scan_idx=2, scan_name="s2", elapsed_s=120.0, voltage_v=4.22))
    assert len(s2.zoom_targets) == 1
    assert s2.zoom_targets[0].actuate is True
    assert s2.zoom_targets[0].predicted_setpoint[0] <= 5.5 + 1e-9
    assert "clamped_to_motor_bounds" in s2.zoom_targets[0].trigger_reasons

    # Scan 3: Grain M already hit max_zooms_per_lineage=1 -> dose cap prevents re-zooming same grain!
    s3 = trk_odd.step(vols[3], stimulus=ScanStimulusPoint(scan_idx=3, scan_name="s3", elapsed_s=180.0, voltage_v=4.28))
    assert len(s3.zoom_targets) == 0
    assert len(s3.audit_records) == 1  # Sim-to-Real Gap automatically audited for scan 2's prediction!

    # Scan 4: Over-voltage excursion (4.65V > 4.40V) halts L3 autonomy -> L1_halt
    s4 = trk_odd.step(vols[2], stimulus=ScanStimulusPoint(scan_idx=4, scan_name="s4", elapsed_s=240.0, voltage_v=4.65))
    assert s4.fal_level_active == "L1_halt"
    assert s4.zoom_targets == []

    # Scan 5: Synchrotron beam dump (flux drops to 1% of baseline) -> instrument_health < 0.25 -> L1_halt
    s5 = trk_odd.step(
        vols[2] * 0.01, stimulus=ScanStimulusPoint(scan_idx=5, scan_name="s5", elapsed_s=300.0, voltage_v=4.20)
    )
    assert s5.instrument_health < 0.10
    assert s5.fal_level_active == "L1_halt"
    assert s5.zoom_targets == []


def test_o1_sliding_window_streaming_and_live_hdf5_quiescence_shadow_mode(tmp_path: Path) -> None:
    """Verify O(1) sliding-window tracking over 36 scans and live shadow-mode resilience to partial HDF5 files."""
    shape = (16, 24, 24)
    trk = CausalStreamingTracker(backend="mock", sliding_window_scans=8)
    d0 = 16.0
    for t in range(36):
        v = np.full(shape, 0.1, dtype=np.float32)
        _add_spot(v, (8.0, d0 - 0.1 * t, 12.0), (1.1, 1.1, 1.1), 220.0)
        res = trk.step(v, scan_name=f"scan_{t:03d}")
        assert res.n_active_tracks == 1

    # Global track_id remains 1 across all 36 scans and root-referenced strain matches exact (d_35 - d_0) / d_0
    assert len(trk.track_table) == 36
    assert {int(r["track_id"]) for r in trk.track_table} == {1}
    assert len(trk._roi_cache) <= 12  # Bounded O(W) ROI cache!
    kin_last = trk.kinematics[0]
    expected_strain = (-0.1 * 35) / d0
    assert kin_last.lineage_strain[-1] == pytest.approx(expected_strain, abs=0.015)

    # Test live watch_autonomous in --mode shadow with an unreadable partial HDF5 file alongside valid scans
    ds_root = tmp_path / "live_stream"
    for t in range(4):
        sdir = ds_root / f"scan_{t + 1:04d}"
        sdir.mkdir(parents=True)
        v = np.full((18, 28, 28), 0.15, dtype=np.float32)
        sx = 1.1 if t == 0 else (2.3 + 0.5 * t)
        _add_spot(v, (9.0, 18.0 - 1.0 * t, 14.0), (1.1, 1.1, sx), 230.0)
        with h5py.File(sdir / "data.h5", "w") as fh:
            fh.create_dataset("/entry/data/data", data=v)

    # Create scan_0005 with a truncated/corrupt in-flight HDF5 file
    partial_dir = ds_root / "scan_0005"
    partial_dir.mkdir(parents=True)
    (partial_dir / "data.h5").write_bytes(b"\x89HDF\r\n\x1a\n_incomplete_lima_flush")

    out_shadow = tmp_path / "shadow_out"
    rc = watch_autonomous.main(
        [
            str(ds_root),
            "--mode",
            "shadow",
            "--backend",
            "mock",
            "--min-precursor-score",
            "0.28",
            "--poll-interval",
            "0.02",
            "--settle-sec",
            "0.01",
            "--max-idle-polls",
            "2",
            "--outdir",
            str(out_shadow),
        ]
    )
    assert rc == 0
    steps = json.loads((out_shadow / "autonomous_steps.json").read_text(encoding="utf-8"))
    assert len(steps) == 4  # Ingested all 4 valid scans and cleanly skipped the partial 5th file!
    trig_lines = [
        json.loads(line)
        for line in (out_shadow / "zoom_triggers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trig_lines) >= 1
    assert all(cmd["actuate"] is False and cmd["execution_mode"] == "shadow" for cmd in trig_lines)
    audit_lines = [
        json.loads(line)
        for line in (out_shadow / "audit_trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(audit_lines) >= 1
    assert audit_lines[0]["schema_version"] == "fal_audit.v1"


def test_html_report_xss_csp_escaping_and_strict_pipeline_config_validation(tmp_path: Path) -> None:
    """Verify DOM XSS mitigation + CSP in HTML report and unknown-key rejection in braggtrack-run --config."""
    art_dir = tmp_path / "artifacts"
    trk_dir = art_dir / "tracking"
    evt_dir = art_dir / "events"
    trk_dir.mkdir(parents=True)
    evt_dir.mkdir(parents=True)

    xss_payload = '<img src=x onerror="alert(1)">'
    (trk_dir / "tracking_summary.json").write_text(
        json.dumps({"n_scans": 2, "scan_names": [f"scan01_{xss_payload}", "scan02"]}),
        encoding="utf-8",
    )
    (evt_dir / "tracks.csv").write_text(
        "track_id,scan_idx,spot_idx,label,centroid_mu,centroid_chi,centroid_d,integrated_intensity,voxel_count,event\n"
        "1,0,0,1,5.0,6.0,2.05,100.0,20,born\n"
        "1,1,0,1,5.1,6.0,2.04,98.0,20,continued\n",
        encoding="utf-8",
    )
    (evt_dir / "events.csv").write_text(
        f"event_type,split_subtype,scan_idx_from,scan_idx_to,parent_track_ids,child_track_ids,intensity_ratio,mean_distance,confidence\n"
        f'split,"{xss_payload}",0,1,1,2;3,0.98,2.1,0.92\n',
        encoding="utf-8",
    )

    html_path = generate_html_report(art_dir, art_dir / "report.html")
    html_text = html_path.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html_text
    assert '"schema_version": "report.v1"' in html_text
    assert "const esc = s =>" in html_text
    # Ensure unescaped payload never appears outside the JSON <script type="application/json"> block
    before_json, after_script = html_text.split('<script id="bt-data" type="application/json">', 1)
    _, inline_js = after_script.split("</script>", 1)
    assert "<img src=x" not in before_json
    assert "<img src=x" not in inline_js

    # Verify strict --config override key check in run_pipeline
    bad_cfg = tmp_path / "bad_config.json"
    bad_cfg.write_text(json.dumps({"precursor_threshld": 0.25}), encoding="utf-8")
    args = run_pipeline.build_parser().parse_args(["--config", str(bad_cfg)])
    with pytest.raises(ValueError, match="precursor_threshld"):
        run_pipeline._merge_config_overrides(args)
