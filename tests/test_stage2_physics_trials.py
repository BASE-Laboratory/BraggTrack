"""Comprehensive physics-informed regression suite for Stage 2 trials and archive evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from braggtrack.autonomous import (
    CausalStreamingTracker,
    ZoomTriggerPolicy,
    compute_electrochemical_state_coord,
    forecast_constant_velocity_2pt,
    forecast_grain_trajectory_gp,
    forecast_kalman_cv,
    forecast_persistence,
)
from braggtrack.cli import eval_archive
from braggtrack.io import ScanStimulusPoint, sample_operando_root
from braggtrack.segmentation import (
    detector_foreground_threshold,
    extract_instance_table,
    relabel_sequential,
    remove_small_objects,
    resolve_anisotropic_sigma,
    segment_classical,
)
from braggtrack.tracking import (
    GeometrySemanticCost,
    PositionShapeCost,
    PrecursorConfig,
    SplitMergeConfig,
    WassersteinBuresCost,
    build_tracks,
    compute_fracture_precursors,
    evaluate_precursor_channels_roc,
    is_single_slice_speck,
)


def _add_gaussian_spot(
    vol: np.ndarray,
    cz: float,
    cy: float,
    cx: float,
    *,
    amp: float = 1200.0,
    sz: float = 1.4,
    sy: float = 2.0,
    sx: float = 2.0,
) -> None:
    zz, yy, xx = np.ogrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    g = amp * np.exp(-0.5 * (((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    vol += g.astype(np.float32)


def test_crowded_comoving_layer_drift_and_kalman_recover_identities() -> None:
    """Trial 1: Co-moving crowded reflections (+18 px/scan > 14 px spacing) alias without drift registration."""
    n_scans = 4
    n_grains = 6
    spacing_x = 14.0
    shift_x = 18.0
    base_xs = [18.0 + k * spacing_x for k in range(n_grains)]
    base_ys = [36.0 + (4.0 if k % 2 else -4.0) for k in range(n_grains)]
    amps = [900.0 + 180.0 * k for k in range(n_grains)]

    volumes: list[np.ndarray] = []
    scan_tables: list[list[dict]] = []
    gt_chain_by_grain: list[list[tuple[int, int]]] = [[] for _ in range(n_grains)]

    for s in range(n_scans):
        vol = np.full((24, 80, 168), 180.0, dtype=np.float32)
        for g_idx in range(n_grains):
            _add_gaussian_spot(
                vol,
                cz=12.0,
                cy=base_ys[g_idx] + 1.5 * s,
                cx=base_xs[g_idx] + shift_x * s,
                amp=amps[g_idx],
            )
        # Inject a transient 1-slice sCMOS cosmic-ray speck at a distinct rocking frame per scan
        vol[2 + 5 * s, 65:68, 20 + 25 * s : 23 + 25 * s] = 950.0
        thr = detector_foreground_threshold(vol, bg_sigma=5.0)
        seg = segment_classical(vol, threshold=thr)
        labels = relabel_sequential(remove_small_objects(seg.labeled_volume, min_size=6))
        spots = extract_instance_table(labels, vol)
        volumes.append(vol)
        scan_tables.append(spots)

        for g_idx in range(n_grains):
            tx = base_xs[g_idx] + shift_x * s
            ty = base_ys[g_idx] + 1.5 * s
            best_i = min(
                range(len(spots)),
                key=lambda i: (
                    abs(spots[i]["centroid_mu"] - 12.0) * 5.0
                    + np.hypot(spots[i]["centroid_chi"] - tx, spots[i]["centroid_d"] - ty)
                ),
            )
            gt_chain_by_grain[g_idx].append((s, best_i))

    assert any(is_single_slice_speck(sp) for sp in scan_tables[0])
    expected_edges = {
        (f"s{s0}_i{i0}", f"s{s1}_i{i1}")
        for chain in gt_chain_by_grain
        for (s0, i0), (s1, i1) in zip(chain[:-1], chain[1:], strict=True)
    }
    assert len(expected_edges) == n_grains * (n_scans - 1)

    cost = GeometrySemanticCost(
        PositionShapeCost(max_step_mu=6.0, max_step_chi=35.0, max_step_d=35.0, intensity_weight=5.0),
        cost_alpha=1.0,
        cost_beta=0.0,
    )
    G_none = build_tracks(scan_tables, cost_fn=cost, max_cost=600.0, drift_mode="none")
    edges_none = set(G_none.edges())
    assert len(expected_edges & edges_none) < len(expected_edges) // 2

    for mode in ("consensus", "fft", "fft_kalman"):
        G_m = build_tracks(
            scan_tables,
            cost_fn=cost,
            max_cost=600.0,
            drift_mode=mode,
            volumes=volumes if mode != "consensus" else None,
        )
        edges_m = set(G_m.edges())
        assert expected_edges.issubset(edges_m), f"drift_mode={mode} missed crowded links"
        speck_nodes = {nid for nid, d in G_m.nodes(data=True) if is_single_slice_speck(d)}
        assert speck_nodes
        assert all(u not in speck_nodes and v not in speck_nodes for u, v in edges_m)


def test_streaming_sliding_window_remaps_lineage_events_with_fft_and_bures() -> None:
    """CausalStreamingTracker preserves committed split events and global track IDs across window eviction."""
    tracker = CausalStreamingTracker(
        sliding_window_scans=6,
        drift_mode="fft",
        cost_model="wasserstein_bures",
        split_max_distance=20.0,
    )
    for s in range(8):
        vol = np.full((18, 64, 64), 180.0, dtype=np.float32)
        if s <= 1:
            _add_gaussian_spot(vol, 9.0, 32.0, 24.0 + 2.0 * s, amp=1600.0)
        else:
            _add_gaussian_spot(vol, 9.0, 27.0, 26.0 + 2.0 * s, amp=800.0)
            _add_gaussian_spot(vol, 9.0, 37.0, 26.0 + 2.0 * s, amp=800.0)
        sr = tracker.tell(vol, scan_name=f"scan{s:04d}")
        assert sr.n_spots >= 1

    assert tracker.n_scans_processed == 8
    all_events = list(tracker.graph.graph.get("lineage_events", []))
    assert any(ev.event_type == "split" and ev.scan_idx_to == 2 for ev in all_events)


def test_anisotropic_log_resolves_stacked_pancake_reflections() -> None:
    """Trial 2a: Aspect-aware anisotropic LoG separates closely stacked thin rocking pancakes."""
    sig_z, sig_y, sig_x = resolve_anisotropic_sigma((30, 128, 128))
    assert sig_z < sig_y == sig_x

    vol = np.full((24, 64, 64), 180.0, dtype=np.float32)
    # Two thin pancake reflections separated by dz=3 slices (sz=0.75, sy=sx=3.6)
    _add_gaussian_spot(vol, cz=10.0, cy=32.0, cx=32.0, amp=1500.0, sz=0.75, sy=3.6, sx=3.6)
    _add_gaussian_spot(vol, cz=13.0, cy=32.0, cx=32.0, amp=1500.0, sz=0.75, sy=3.6, sx=3.6)
    thr = detector_foreground_threshold(vol, bg_sigma=5.0)

    seg_iso = segment_classical(vol, threshold=thr, sigma=1.8, min_seed_separation=4, anisotropic_auto=False)
    seg_aniso = segment_classical(vol, threshold=thr, anisotropic_auto=True)
    assert seg_iso.component_count == 1
    assert seg_aniso.component_count == 2


def test_skewness_extraction_and_reversible_overlap_disentanglement() -> None:
    """Trial 2b/c: Asymmetric spots yield signed skewness; 2->1->2 crossing is tagged reversible_overlap."""
    vol = np.full((16, 48, 64), 180.0, dtype=np.float32)
    zz, yy, xx = np.ogrid[:16, :48, :64]
    # Unimodal right-skewed reflection along +x (chi): narrow left flank (sx=1.2) vs broad right wing (sx=3.8)
    sx_asym = np.where(xx < 20.0, 1.2, 3.8)
    spot = 1800.0 * np.exp(-0.5 * (((zz - 8.0) / 1.3) ** 2 + ((yy - 24.0) / 2.0) ** 2 + ((xx - 20.0) / sx_asym) ** 2))
    vol += spot.astype(np.float32)
    thr = detector_foreground_threshold(vol, bg_sigma=5.0)
    seg = segment_classical(vol, threshold=thr)
    rows = extract_instance_table(seg.labeled_volume, vol)
    assert len(rows) == 1
    assert rows[0]["skewness_chi"] > 0.25

    # Reversible overlap (2 -> 1 -> 2) across 4 scans vs persistent mechanical split (1 -> 2 -> 2)
    s0 = [
        {
            "label": 1,
            "centroid_mu": 10.0,
            "centroid_chi": 20.0,
            "centroid_d": 30.0,
            "net_intensity": 500.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
        {
            "label": 2,
            "centroid_mu": 10.0,
            "centroid_chi": 28.0,
            "centroid_d": 30.0,
            "net_intensity": 500.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
        {
            "label": 3,
            "centroid_mu": 40.0,
            "centroid_chi": 80.0,
            "centroid_d": 30.0,
            "net_intensity": 1000.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
    ]
    s1 = [
        {
            "label": 1,
            "centroid_mu": 10.0,
            "centroid_chi": 24.0,
            "centroid_d": 30.0,
            "net_intensity": 1000.0,
            "eig_1": 3.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
        {
            "label": 2,
            "centroid_mu": 40.0,
            "centroid_chi": 80.0,
            "centroid_d": 30.0,
            "net_intensity": 1000.0,
            "eig_1": 3.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
    ]
    s2 = [
        {
            "label": 1,
            "centroid_mu": 10.0,
            "centroid_chi": 20.0,
            "centroid_d": 30.0,
            "net_intensity": 500.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
        {
            "label": 2,
            "centroid_mu": 10.0,
            "centroid_chi": 28.0,
            "centroid_d": 30.0,
            "net_intensity": 500.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
        {
            "label": 3,
            "centroid_mu": 40.0,
            "centroid_chi": 76.0,
            "centroid_d": 30.0,
            "net_intensity": 500.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
        {
            "label": 4,
            "centroid_mu": 40.0,
            "centroid_chi": 84.0,
            "centroid_d": 30.0,
            "net_intensity": 500.0,
            "eig_1": 2.0,
            "eig_2": 1.0,
            "eig_3": 0.5,
        },
    ]
    s3 = list(s2)
    G = build_tracks([s0, s1, s2, s3], split_merge_config=SplitMergeConfig(max_distance=15.0))
    splits = [e for e in G.graph["lineage_events"] if e.event_type == "split"]
    assert len(splits) == 2
    by_mu = {round(e.centroid_mu): e for e in splits}
    assert by_mu[10].split_subtype == "reversible_overlap"
    assert by_mu[10].preceded_by_merge is True
    assert by_mu[40].split_subtype == "mechanical_split"
    assert by_mu[40].preceded_by_merge is False
    assert by_mu[40].daughter_persistence_scans == 2


def test_wasserstein_bures_disambiguates_orthogonal_tilted_streaks() -> None:
    """Trial 3: WassersteinBuresCost separates +45 deg vs -45 deg streaks with identical eigenvalues."""

    def _tilted_spot(chi: float, d: float, sign: float) -> dict:
        return {
            "centroid_mu": 10.0,
            "centroid_chi": chi,
            "centroid_d": d,
            "net_intensity": 1000.0,
            "cov_zz": 0.5,
            "cov_xx": 5.0,
            "cov_yy": 5.0,
            "cov_zx": 0.0,
            "cov_zy": 0.0,
            "cov_yx": 4.5 * sign,
            "eig_1": 9.5,
            "eig_2": 0.5,
            "eig_3": 0.5,
        }

    # Co-located crossing: spot 0 (+45 deg) and spot 1 (-45 deg) swap slightly past each other
    spots_t = [_tilted_spot(20.0, 20.0, +1.0), _tilted_spot(22.0, 20.0, -1.0)]
    spots_t1 = [_tilted_spot(22.2, 20.0, +1.0), _tilted_spot(19.8, 20.0, -1.0)]

    pos_cost = PositionShapeCost(position_weight=1.0, shape_weight=2.0)
    wb_cost = WassersteinBuresCost(position_weight=1.0, bures_weight=2.0, intensity_weight=0.0)

    # Eigenvalue-only cost assigns zero shape penalty to orthogonal +45 vs -45 deg tilt and swaps IDs!
    G_pos = build_tracks([spots_t, spots_t1], cost_fn=pos_cost)
    assert ("s0_i0", "s1_i1") in G_pos.edges()

    # Exact 3D Gaussian Bures transport penalizes orthogonal covariance rotation and preserves identities!
    G_wb = build_tracks([spots_t, spots_t1], cost_fn=wb_cost)
    assert ("s0_i0", "s1_i0") in G_wb.edges()
    assert ("s0_i1", "s1_i1") in G_wb.edges()


def test_causal_electrochemical_state_gp_halts_ocv_overshoot() -> None:
    """Trial 4: Causal GP(Q, V) halts extrapolation during OCV rest where GP(t) overshoots."""
    # Scans 0..4: galvanostatic charge (+10 mAh/g, +0.05 V per scan, d shifts +4.0 px/scan)
    # Scans 4..6: OCV rest (I = 0 mA, dQ = 0, dV = 0, d holds steady at 116.0)
    stimuli: dict[int, ScanStimulusPoint] = {}
    for s in range(7):
        active = min(s, 4)
        is_charging = s < 4
        stimuli[s] = ScanStimulusPoint(
            scan_idx=s,
            scan_name=f"scan{s:04d}",
            elapsed_s=float(s * 60.0),
            voltage_v=3.70 + 0.05 * active,
            current_ma=1.0 if is_charging else 0.0,
            capacity_mah_g=10.0 * active,
        )

    times = np.array([s * 60.0 for s in range(5)], dtype=np.float64)
    mu = np.full(5, 20.0)
    chi = np.full(5, 50.0)
    d_vals = np.array([100.0 + 4.0 * s for s in range(5)], dtype=np.float64)
    errs = np.full(5, 0.25)

    # At scan 4, current drops to 0.0 mA (entering OCV rest before scan 5 at t=300s)
    res_coord = compute_electrochemical_state_coord(
        stimuli,
        [0, 1, 2, 3, 4],
        lead_scans=1.0,
    )
    assert res_coord is not None
    u_hist, u_target = res_coord
    assert u_target == pytest.approx(float(u_hist[-1]), abs=1e-6)

    fc_time = forecast_grain_trajectory_gp(times, mu, chi, d_vals, errs, errs, errs, 300.0, domain="time")
    fc_elec = forecast_grain_trajectory_gp(
        times,
        mu,
        chi,
        d_vals,
        errs,
        errs,
        errs,
        300.0,
        domain="electro_state",
        stimuli=stimuli,
        scan_indices=[0, 1, 2, 3, 4],
        lead_scans=1.0,
    )
    fc_cv = forecast_constant_velocity_2pt(times, mu, chi, d_vals, errs, errs, errs, 300.0)
    fc_kf = forecast_kalman_cv(times, mu, chi, d_vals, errs, errs, errs, 300.0)
    fc_pers = forecast_persistence(mu, chi, d_vals, errs, errs, errs)

    gt_d_ocv = 116.0
    assert abs(fc_elec.mean[2] - gt_d_ocv) < 0.8
    assert abs(fc_time.mean[2] - gt_d_ocv) > 2.5
    assert abs(fc_cv.mean[2] - gt_d_ocv) > 3.5
    assert abs(fc_kf.mean[2] - gt_d_ocv) > 2.5
    assert fc_pers.mean[2] == pytest.approx(gt_d_ocv)
    assert ZoomTriggerPolicy(forecast_domain="electro_state").forecast_domain == "electro_state"


def test_baseline_referenced_precursors_and_roc_benchmark() -> None:
    """Trial 5: Baseline-referenced channels ignore static 6:1 pancakes and achieve high AUROC on cleavage."""
    from braggtrack.tracking.events import LineageEvent

    rows: list[dict] = []
    # Track 1: Pristine 6:1 elongated grain (constant shape across 6 scans -> zero false alarm)
    for s in range(6):
        rows.append(
            {
                "track_id": 1,
                "scan_idx": s,
                "label": 1,
                "centroid_mu": 20.0,
                "centroid_chi": 40.0,
                "centroid_d": 60.0,
                "eig_1": 36.0,
                "eig_2": 4.0,
                "eig_3": 1.0,
                "skewness_mu": 0.1,
                "skewness_chi": 0.1,
                "skewness_d": 0.0,
                "kurtosis_mu": 0.0,
                "kurtosis_chi": 0.0,
                "kurtosis_d": 0.0,
            }
        )
    # Track 2: Round grain in scans 0..1 that elongates, skews, and flattens top (platykurtic < 0) in scans 2..4 before splitting
    for s in range(5):
        pre = max(0, s - 1)
        rows.append(
            {
                "track_id": 2,
                "scan_idx": s,
                "label": 2,
                "centroid_mu": 50.0,
                "centroid_chi": 90.0,
                "centroid_d": 60.0,
                "eig_1": 1.5 + 3.0 * pre,
                "eig_2": 1.4 + 0.6 * pre,
                "eig_3": 1.2,
                "skewness_mu": 0.0,
                "skewness_chi": 0.35 * pre,
                "skewness_d": 0.0,
                "kurtosis_mu": 0.0,
                "kurtosis_chi": -0.45 * pre,
                "kurtosis_d": 0.0,
            }
        )

    split_ev = LineageEvent(
        event_id=1,
        event_type="split",
        scan_idx_from=4,
        scan_idx_to=5,
        parent_track_ids=(2,),
        child_track_ids=(3, 4),
        parent_nodes=("s4_i1",),
        child_nodes=("s5_i1", "s5_i2"),
        centroid_mu=50.0,
        centroid_chi=90.0,
        centroid_d=60.0,
        intensity_ratio=1.0,
        mean_distance=4.0,
        split_subtype="mechanical_split",
        preceded_by_merge=False,
        daughter_persistence_scans=3,
    )
    cfg = PrecursorConfig(baseline_reference=True, baseline_scans=2, score_threshold=0.20, min_persistence=2)
    candidates, obs_scores = compute_fracture_precursors(rows, cfg, lineage_events=[split_ev])

    t1_scores = [sc for sc in obs_scores if sc.track_id == 1]
    assert all(sc.delta_log_anisotropy == pytest.approx(0.0, abs=1e-6) for sc in t1_scores)
    assert [c.track_id for c in candidates] == [2]

    roc = evaluate_precursor_channels_roc(
        obs_scores,
        [split_ev],
        lookback_scans=3,
        min_daughter_persistence=2,
        exclude_reversible_overlap=True,
    )
    assert roc["baseline_referenced_score"].auroc is not None
    assert roc["baseline_referenced_score"].auroc > 0.90
    assert roc["platykurtic_drop"].auroc is not None
    assert roc["platykurtic_drop"].auroc > 0.80


def test_blind_annotator_gt_scorer_and_eval_archive_cli(tmp_path: Path) -> None:
    """End-to-end verification of blind annotator payload, GT link scorer, and 2-pass archive CLI."""
    import importlib.util

    from braggtrack.tracking.gt_eval import score_predictions_against_gt

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_blind_annotator.py"
    spec = importlib.util.spec_from_file_location("build_blind_annotator", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    root = Path(sample_operando_root())
    payload = mod.build_blind_annotator_payload(root, num_scans=3, max_preview_dim=64, seed=7)
    assert len(payload["scans"]) == 3
    html_doc = mod.render_blind_annotator_html(payload)
    assert "ground_truth_links.json" in html_doc

    gt_obj = {
        "schema_version": "braggtrack_gt_links.v1",
        "scans": payload["scans"],
        "links": [
            {"scan_from": 0, "scan_to": 1, "link_type": "1-to-1", "source_spots": [1], "target_spots": [1]},
            {"scan_from": 0, "scan_to": 1, "link_type": "uncertain", "source_spots": [2], "target_spots": [2]},
        ],
    }
    res = score_predictions_against_gt({(0, 1, 1, 1), (0, 1, 2, 2)}, [], gt_obj)
    assert res["overall_1to1"]["tp"] == 1
    assert res["overall_1to1"]["fp"] == 0  # uncertain edge excluded from FP penalty!
    assert res["overall_1to1"]["f1"] == pytest.approx(1.0)

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_obj), encoding="utf-8")
    op_csv = tmp_path / "operando.csv"
    op_csv.write_text(
        "elapsed_s,voltage_v,current_ma,capacity_mah_g\n0,3.70,1.0,0.0\n60,3.78,1.0,12.0\n120,3.85,1.0,24.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "eval_out"
    rc1 = eval_archive.main(
        ["--root", str(root), "--outdir", str(out_dir), "--operando-log", str(op_csv), "--gt-links", str(gt_path)]
    )
    assert rc1 == 0
    rc2 = eval_archive.main(
        [
            "--root",
            str(root),
            "--outdir",
            str(out_dir),
            "--operando-log",
            str(op_csv),
            "--gt-links",
            str(gt_path),
            "--cache-only-pass2",
        ]
    )
    assert rc2 == 0
    rep = json.loads((out_dir / "archive_evaluation_report.json").read_text(encoding="utf-8"))
    assert rep["n_scans_evaluated"] == 3
    assert rep["has_operando_stimulus"] is True
    assert rep["has_blind_gt_links"] is True
    assert len(rep["tracking_grid"]) == 8


def test_estimate_slab_shifts_fft_supports_lazy_volume() -> None:
    """LazyVolume inputs must stream bounded rocking slabs identically to dense ndarrays."""
    from braggtrack.io import LazyVolume
    from braggtrack.tracking import estimate_slab_shifts_fft

    rng = np.random.default_rng(19)
    v0 = 180.0 + rng.normal(0.0, 2.0, size=(20, 64, 64))
    v1 = 180.0 + rng.normal(0.0, 2.0, size=(20, 64, 64))
    zz, yy, xx = np.ogrid[:20, :64, :64]
    for cy, cx in ((20.0, 18.0), (24.0, 32.0)):
        v0 += 650.0 * np.exp(-0.5 * (((zz - 10.0) / 1.5) ** 2 + ((yy - cy) / 2.0) ** 2 + ((xx - cx) / 2.0) ** 2))
        v1 += 650.0 * np.exp(
            -0.5 * (((zz - 10.0) / 1.5) ** 2 + ((yy - (cy + 9.0)) / 2.0) ** 2 + ((xx - (cx - 6.0)) / 2.0) ** 2)
        )
    spots_t = [
        {
            "label": 1,
            "centroid_mu": 10.0,
            "centroid_chi": 18.0,
            "centroid_d": 20.0,
            "voxel_count": 40,
            "eig_1": 4.0,
            "eig_2": 4.0,
            "eig_3": 2.2,
            "bbox_min_z": 8,
            "bbox_max_z": 12,
            "bbox_min_y": 16,
            "bbox_max_y": 24,
            "bbox_min_x": 14,
            "bbox_max_x": 22,
        },
        {
            "label": 2,
            "centroid_mu": 10.0,
            "centroid_chi": 32.0,
            "centroid_d": 24.0,
            "voxel_count": 40,
            "eig_1": 4.0,
            "eig_2": 4.0,
            "eig_3": 2.2,
            "bbox_min_z": 8,
            "bbox_max_z": 12,
            "bbox_min_y": 20,
            "bbox_max_y": 28,
            "bbox_min_x": 28,
            "bbox_max_x": 36,
        },
    ]
    shifts_dense, recs_dense = estimate_slab_shifts_fft(v0, v1, spots_t)
    shifts_lazy, recs_lazy = estimate_slab_shifts_fft(
        LazyVolume.from_array(v0),
        LazyVolume.from_array(v1),
        spots_t,
    )
    assert recs_lazy == recs_dense
    assert np.allclose(shifts_lazy, shifts_dense)
    assert shifts_lazy[0][1] == pytest.approx(-6.0, abs=0.4)
    assert shifts_lazy[0][2] == pytest.approx(9.0, abs=0.4)


def test_cli_stage2_flags_in_run_pipeline_and_watch_replay(tmp_path: Path) -> None:
    """braggtrack-run and braggtrack-watch must accept and propagate Stage 2 physics flags."""
    from braggtrack.cli import run_pipeline, watch_autonomous

    root = Path(sample_operando_root())
    pipe_out = tmp_path / "pipe_stage2"
    rc_pipe = run_pipeline.main(
        [
            str(root),
            "--outdir",
            str(pipe_out),
            "--cost-model",
            "wasserstein_bures",
            "--drift-mode",
            "fft_kalman",
            "--precursor-baseline-reference",
        ]
    )
    assert rc_pipe == 0
    trk_sum = json.loads((pipe_out / "tracking" / "tracking_summary.json").read_text(encoding="utf-8"))
    evt_sum = json.loads((pipe_out / "events" / "events_summary.json").read_text(encoding="utf-8"))
    assert trk_sum["cost_model"] == "wasserstein_bures"
    assert trk_sum["drift_mode"] == "fft_kalman"
    assert evt_sum["cost_model"] == "wasserstein_bures"
    assert evt_sum["drift_mode"] == "fft_kalman"

    op_csv = tmp_path / "operando.csv"
    op_csv.write_text(
        "elapsed_s,voltage_v,current_ma,capacity_mah_g\n0,3.70,1.0,0.0\n60,3.78,1.0,12.0\n120,3.85,1.0,24.0\n",
        encoding="utf-8",
    )
    watch_out = tmp_path / "watch_stage2"
    rc_watch = watch_autonomous.main(
        [
            str(root),
            "--mode",
            "replay",
            "--outdir",
            str(watch_out),
            "--operando-log",
            str(op_csv),
            "--cost-model",
            "wasserstein_bures",
            "--drift-mode",
            "consensus",
            "--forecast-domain",
            "state_space",
            "--precursor-baseline-reference",
        ]
    )
    assert rc_watch == 0
    assert (watch_out / "replay_summary.json").exists()
