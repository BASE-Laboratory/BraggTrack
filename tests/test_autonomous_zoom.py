"""Unit tests for online causal streaming and autonomous DFXM Scout-and-Zoom."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from braggtrack.autonomous import (
    CausalStreamingTracker,
    ZoomTriggerPolicy,
    evaluate_causal_replay,
)
from braggtrack.cli import watch_autonomous
from braggtrack.io import AxisCalibration, DetectorGeometry, ScanStimulusPoint


def _add_gaussian_spot(
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
    g = amplitude * np.exp(-0.5 * (((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    vol += g


def _build_scout_and_zoom_sequence() -> tuple[list[np.ndarray], list[ScanStimulusPoint]]:
    """Construct a 6-scan operando sequence with:
    - Grain A (target): builds bimodal mosaicity across scans 1..3 while drifting fast
      along d (-2.5 px/scan, mimicking Ni92 H2->H3 contraction) and fractures 1->2 at scan 4.
    - Grain B (bright distractor): higher integrated intensity, spherical, never fractures.
    - Grain C (high-strain distractor): shifts early so |strain| is larger at scan 1..2,
      spherical, never fractures.
    """
    shape = (24, 36, 36)
    vols: list[np.ndarray] = []
    stimuli: list[ScanStimulusPoint] = []
    voltages = [3.95, 4.08, 4.16, 4.21, 4.24, 4.27]
    dq_dvs = [40.0, 95.0, 260.0, 340.0, 180.0, 70.0]

    for t in range(6):
        v = np.full(shape, 0.2, dtype=np.float64)
        d_target = 25.0 - 2.0 * t

        # Target Grain A: spherical at t=0, strong anisotropic mosaicity broadening at t=1,2,3
        # (single LoG peak so watershed keeps it unified), then cleaves into 2 intensity-conserving
        # daughter reflections at t=4,5!
        if t == 0:
            _add_gaussian_spot(v, (8.0, d_target, 12.0), (1.1, 1.1, 1.1), 220.0)
        elif t in (1, 2, 3):
            sx = 2.0 + 0.55 * t  # progressive mosaicity elongation along chi
            amp = 220.0 / (sx / 1.1) ** 0.35  # grows moderately while staying below Distractor B
            _add_gaussian_spot(v, (8.0, d_target, 12.0), (1.1, 1.25, sx), amp)
        else:
            sep = 4.2 + 0.5 * (t - 4)
            _add_gaussian_spot(v, (8.0, d_target, 12.0 - sep), (1.1, 1.2, 1.45), 185.0)
            _add_gaussian_spot(v, (8.0, d_target, 12.0 + sep), (1.1, 1.2, 1.45), 175.0)

        # Distractor Grain B: brightest integrated intensity, completely spherical & stable
        _add_gaussian_spot(v, (16.0, 10.0, 27.0), (1.35, 1.35, 1.35), 340.0)

        # Distractor Grain C: large static d-shift right after t=0, spherical & non-fracturing
        d_c = 12.0 if t == 0 else 8.0
        _add_gaussian_spot(v, (16.0, d_c, 10.0), (1.2, 1.2, 1.2), 260.0)

        vols.append(v)
        stimuli.append(
            ScanStimulusPoint(
                scan_idx=t,
                scan_name=f"scan_{t + 1:04d}",
                elapsed_s=float(t * 180.0),
                timestamp=f"2026-05-01T10:{t * 3:02d}:00Z",
                voltage_v=voltages[t],
                current_ma=0.85,
                capacity_mah_g=float(140.0 + 14.0 * t),
                dq_dv=dq_dvs[t],
            )
        )
    return vols, stimuli


def test_strict_causality_invariance() -> None:
    """Processing scans 0..2 must produce bit-for-bit identical outputs whether future scans 3..5 exist or not."""
    vols, stimuli = _build_scout_and_zoom_sequence()
    # Inject a massive future intensity spike in scan 3..5 that would corrupt any non-causal smoothing window
    vols_spiked = [v.copy() for v in vols]
    vols_spiked[3] = vols_spiked[3] * 12.0 + 50.0
    vols_spiked[4] = vols_spiked[4] * 15.0 + 80.0

    policy = ZoomTriggerPolicy(min_precursor_score=0.30, min_persistence=2)
    tracker_short = CausalStreamingTracker(backend="mock", smooth_window=5, trigger_policy=policy)
    tracker_long = CausalStreamingTracker(backend="mock", smooth_window=5, trigger_policy=policy)

    steps_short = [tracker_short.step(vols[t], stimulus=stimuli[t]) for t in range(3)]
    steps_long_prefix = [tracker_long.step(vols_spiked[t], stimulus=stimuli[t]) for t in range(3)]

    for s_a, s_b in zip(steps_short, steps_long_prefix, strict=True):
        assert s_a.threshold_causal == pytest.approx(s_b.threshold_causal, abs=1e-12)
        assert s_a.n_spots == s_b.n_spots
        assert [t.to_bliss_command() for t in s_a.zoom_targets] == [t.to_bliss_command() for t in s_b.zoom_targets]


def test_velocity_extrapolated_setpoint_hits_fast_drifting_grain() -> None:
    """On a fast-shifting H2->H3 reflection, velocity-extrapolated setpoint centers the t+1 spot whereas static lags."""
    shape = (20, 44, 28)
    vols: list[np.ndarray] = []
    # Drift -6.0 voxels/scan along d (y-axis) so static centroid misses a tight 2.0-sigma window!
    true_centers = [(10.0, 36.0 - 6.0 * t, 14.0) for t in range(4)]
    for t, (cz, cy, cx) in enumerate(true_centers):
        v = np.full(shape, 0.2, dtype=np.float64)
        if t == 0:
            _add_gaussian_spot(v, (cz, cy, cx), (1.0, 1.0, 1.0), 200.0)
        else:
            _add_gaussian_spot(v, (cz, cy, cx), (1.0, 1.1, 2.4 + 0.5 * t), 200.0)
        vols.append(v)

    cal = AxisCalibration(
        mu_origin=10.0, mu_step=0.02, chi_origin=-1.0, chi_step=0.05, d_origin=4.5, d_step=0.002, calibrated=True
    )
    det = DetectorGeometry(distance_mm=150.0, wavelength_angstrom=0.2898)
    # Use a tight window_sigma=2.0 so lagging by 5.5*0.002 = 0.011 A falls outside the static window!
    policy = ZoomTriggerPolicy(min_precursor_score=0.28, min_persistence=2, lead_scans=1.0, window_sigma=2.0)
    tracker = CausalStreamingTracker(backend="mock", calibration=cal, detector=det, trigger_policy=policy)

    step2 = None
    for t in range(3):
        step2 = tracker.step(vols[t], scan_name=f"scan_{t:02d}")
    assert step2 is not None
    assert len(step2.zoom_targets) == 1
    tgt = step2.zoom_targets[0]

    # Step 3 gives actual next-scan centroid in calibrated coordinates
    tracker.step(vols[3], scan_name="scan_03")
    next_obs = [r for r in tracker.track_table if int(r["scan_idx"]) == 3][0]
    next_mu, next_chi, next_d = (
        float(next_obs["centroid_mu"]),
        float(next_obs["centroid_chi"]),
        float(next_obs["centroid_d"]),
    )

    # Extrapolated setpoint captures next scan; un-extrapolated current_centroid misses along d!
    assert tgt.contains_point(next_mu, next_chi, next_d) is True
    cur_d = tgt.current_centroid[2]
    half_d = tgt.scan_half_widths[2]
    assert abs(next_d - cur_d) > half_d

    bliss = tgt.to_bliss_command()
    assert bliss["schema_version"] == "bliss_dfxm_zoom.v1"
    assert "crystallographic_setpoint" in bliss
    assert bliss["crystallographic_setpoint"]["two_theta_deg"] > 0.0


def test_causal_replay_beats_naive_baselines_with_lead_time() -> None:
    """Closed-loop replay captures fracturing lineage ahead of split while naive baselines pick distractors."""
    vols, stimuli = _build_scout_and_zoom_sequence()
    cal = AxisCalibration(
        mu_origin=0.0, mu_step=0.05, chi_origin=0.0, chi_step=0.05, d_origin=2.0, d_step=0.002, calibrated=True
    )
    policy = ZoomTriggerPolicy(min_precursor_score=0.30, min_persistence=2, cooldown_scans=4, max_triggers_per_scan=1)

    res = evaluate_causal_replay(
        vols,
        stimuli=stimuli,
        calibration=cal,
        backend="mock",
        trigger_policy=policy,
        zoom_budget=1,
        seed=123,
    )

    bt = res.policy_comparison["braggtrack_autonomous"]
    bright = res.policy_comparison["brightest_intensity"]
    strain_only = res.policy_comparison["strain_only"]

    assert bt.n_fracture_lineages_total >= 1
    assert bt.fracture_capture_recall == pytest.approx(1.0)
    assert bt.fracture_capture_precision == pytest.approx(1.0)
    assert bt.mean_lead_scans_before_split >= 1.0
    assert bt.mean_lead_voltage_mv is not None and bt.mean_lead_voltage_mv > 10.0
    assert bt.setpoint_hit_rate_next_scan == pytest.approx(1.0)

    # Under budget=1, brightest_intensity picks Grain B and strain_only picks Grain C (both non-fracturing!)
    assert bright.fracture_capture_recall == pytest.approx(0.0)
    assert strain_only.fracture_capture_recall == pytest.approx(0.0)


def test_watch_autonomous_cli_replay_and_watch_modes(tmp_path: Path) -> None:
    """Verify braggtrack-watch CLI in both --mode replay and --mode watch."""
    vols, stimuli = _build_scout_and_zoom_sequence()
    ds_root = tmp_path / "ni92_dataset"
    for t, vol in enumerate(vols):
        sdir = ds_root / f"scan_{t + 1:04d}"
        sdir.mkdir(parents=True)
        with h5py.File(sdir / "data.h5", "w") as fh:
            eg = fh.create_group("entry")
            eg.create_dataset("start_time", data=stimuli[t].timestamp)
            dg = eg.create_group("data")
            dg.create_dataset("data", data=vol.astype(np.float32))

    echem_csv = tmp_path / "echem.csv"
    lines = ["scan_idx,voltage_v,current_ma,capacity_mah_g,dq_dv"]
    for sp in stimuli:
        lines.append(f"{sp.scan_idx},{sp.voltage_v},{sp.current_ma},{sp.capacity_mah_g},{sp.dq_dv}")
    echem_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_replay = tmp_path / "out_replay"
    rc = watch_autonomous.main(
        [
            str(ds_root),
            "--mode",
            "replay",
            "--backend",
            "mock",
            "--operando-log",
            str(echem_csv),
            "--min-precursor-score",
            "0.30",
            "--outdir",
            str(out_replay),
        ]
    )
    assert rc == 0
    summary = json.loads((out_replay / "replay_summary.json").read_text(encoding="utf-8"))
    assert summary["n_scans"] == 6
    assert summary["n_zoom_targets"] >= 1
    jsonl_lines = [
        json.loads(line)
        for line in (out_replay / "zoom_triggers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_lines) == summary["n_zoom_targets"]
    assert jsonl_lines[0]["operando_state"]["voltage_v"] == pytest.approx(4.16, abs=0.1)

    out_watch = tmp_path / "out_watch"
    rc_w = watch_autonomous.main(
        [
            str(ds_root),
            "--mode",
            "watch",
            "--backend",
            "mock",
            "--max-scans",
            "4",
            "--min-precursor-score",
            "0.30",
            "--outdir",
            str(out_watch),
        ]
    )
    assert rc_w == 0
    watch_steps = json.loads((out_watch / "autonomous_steps.json").read_text(encoding="utf-8"))
    assert len(watch_steps) == 4
