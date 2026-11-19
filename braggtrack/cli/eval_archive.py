"""Two-pass O(1)-RAM 1,000-scan operando archive evaluation harness (``braggtrack-eval-archive``).

Pass 1 streams 3D HDF5 rocking-curve scans one pair at a time, extracting morphological
spot tables (3x3 covariance, skewness, kurtosis, sub-peaks) and windowed 2D FFT rocking-slab
translations into a compact resumable ``archive_cache.jsonl`` file.

Pass 2 loads ``archive_cache.jsonl`` in seconds with zero HDF5 re-reads and benchmarks:
1. Tracking ablation grid: ``{none, consensus, fft, fft_kalman} x {position_shape, wasserstein_bures}``
   (plus optional ``DINOv2`` morphological hybrid and blind human ``--gt-links`` F1 scoring).
2. Multi-horizon causal forecasting (``h = 1, 2, 5`` scans): ``persistence``, ``constant_velocity_2pt``,
   ``kalman_cv``, ``clock_time_gp``, and ``electro_state_gp`` (overall + stratified by ``|dQ/dV|``).
3. Fracture precursor channel ROC/PR benchmark: ``AUROC``, ``AUPRC``, false-alarm rate per 100
   non-fracturing observations, mean scan lead, and synchronized electrochemical ``mV`` lead.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.autonomous.gp import (
    GPForecast,
    forecast_constant_velocity_2pt,
    forecast_grain_trajectory_gp,
    forecast_kalman_cv,
    forecast_persistence,
)
from braggtrack.io import (
    AxisCalibration,
    BeamlineAdapter,
    ExperimentSequence,
    ScanStimulusPoint,
    ScanVolumeMeta,
    discover_operando_scans,
    extract_axis_calibration,
    extract_scan_metadata,
    load_operando_log,
    open_lazy_volume,
    synchronize_to_scans,
)
from braggtrack.io.discovery import ScanFile
from braggtrack.io.lazy import detector_foreground_threshold_chunked
from braggtrack.segmentation import (
    extract_instance_table,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
)
from braggtrack.semantic import DEFAULT_DINO_MODEL, make_multiview_encoder, make_patch_encoder
from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips
from braggtrack.tracking.cost import CostFunction, GeometrySemanticCost, PositionShapeCost, WassersteinBuresCost
from braggtrack.tracking.drift import DriftMode, estimate_slab_shifts_fft, is_single_slice_speck
from braggtrack.tracking.events import SplitMergeConfig
from braggtrack.tracking.kinematics import compute_grain_kinematics
from braggtrack.tracking.lifecycle import build_tracks, tracks_to_table
from braggtrack.tracking.precursors import (
    PrecursorConfig,
    _count_internal_subpeaks,
    _patch_dispersion_score,
    compute_fracture_precursors,
    evaluate_precursor_channels_roc,
)


def _discover_scan_files(root: Path) -> list[ScanFile]:
    scans = discover_operando_scans(root)
    if not scans:
        flat_h5 = sorted({*root.glob("*.h5"), *root.glob("*.nxs"), *root.glob("*.hdf5")})
        scans = [ScanFile(scan_name=p.stem, path=p) for p in flat_h5]
    return scans


def _serialize_spot_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            out[k] = [round(float(x), 6) for x in v.ravel()]
        elif isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 6) if math.isfinite(float(v)) else 0.0
        elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _deserialize_spot_row(row: dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    if "embedding" in d and isinstance(d["embedding"], list):
        d["embedding"] = np.asarray(d["embedding"], dtype=np.float32)
    return d


def stream_archive_cache(
    root: Path,
    cache_path: Path,
    *,
    start_scan: int = 0,
    end_scan: int | None = None,
    stride: int = 1,
    bg_sigma: float = 5.0,
    smooth_window: int = 5,
    seed_peak_fraction: float = 0.08,
    seed_response_percentile: float = 95.0,
    min_spot_voxels: int = 6,
    anisotropic: bool = False,
    max_step_chi: float = 40.0,
    max_step_d: float = 40.0,
    include_dino: bool = False,
    dino_backend: str = "auto",
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Stream HDF5 scans in O(1) RAM and persist morphological tables + FFT slab shifts to JSONL."""
    all_files = _discover_scan_files(root)
    stop = len(all_files) if end_scan is None else min(len(all_files), int(end_scan))
    selected = list(enumerate(all_files))[start_scan : stop : max(1, int(stride))]
    if not selected:
        raise ValueError(f"No scans selected in {root} (start={start_scan}, end={end_scan}, stride={stride})")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_by_seq: dict[int, dict[str, Any]] = {}
    if resume and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            existing_by_seq[int(rec["seq_idx"])] = rec

    cal = extract_axis_calibration(selected[0][1].path)
    mv_encoder = make_multiview_encoder(dino_backend, model_name=DEFAULT_DINO_MODEL) if include_dino else None  # type: ignore[arg-type]
    patch_enc = make_patch_encoder(dino_backend, model_name=DEFAULT_DINO_MODEL) if include_dino else None  # type: ignore[arg-type]

    raw_thresholds: list[float] = [float(r["threshold_raw"]) for _, r in sorted(existing_by_seq.items())]
    records: list[dict[str, Any]] = []
    prev_vol: np.ndarray | None = None
    prev_spots: list[dict[str, Any]] | None = None
    prev_file: Path | None = None

    mode = "a" if (resume and existing_by_seq) else "w"
    with cache_path.open(mode, encoding="utf-8") as fh:
        for seq_idx, (orig_idx, sc_file) in enumerate(selected):
            if seq_idx in existing_by_seq:
                rec = existing_by_seq[seq_idx]
                records.append(rec)
                prev_vol = None
                prev_spots = [_deserialize_spot_row(s) for s in rec["spots"]]
                prev_file = sc_file.path
                continue

            if seq_idx > 0 and prev_vol is None and prev_file is not None and prev_file.exists():
                with open_lazy_volume(prev_file) as lv_prev:
                    prev_vol = np.asarray(lv_prev._dataset[()], dtype=np.float32)

            with open_lazy_volume(sc_file.path) as lv:
                raw_thr = float(detector_foreground_threshold_chunked(lv, bg_sigma=bg_sigma))
                vol = np.asarray(lv._dataset[()], dtype=np.float32)
            meta = extract_scan_metadata(sc_file.path)
            raw_thresholds.append(raw_thr)
            win_lo = max(0, len(raw_thresholds) - max(1, int(smooth_window)))
            causal_thr = float(np.median(raw_thresholds[win_lo:]))

            seg = segment_classical(
                vol,
                threshold=causal_thr,
                seed_peak_fraction=seed_peak_fraction,
                seed_response_percentile=seed_response_percentile,
                anisotropic_auto=anisotropic,
            )
            labels = relabel_sequential(remove_small_objects(seg.labeled_volume, min_size=min_spot_voxels))
            spots = extract_instance_table(labels, vol, calibration=cal)

            triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for row in spots:
                lbl_i = int(row["label"])
                masked_cube, mask = crop_spot_cube(vol, labels, lbl_i, row, margin=2)
                n_sub, _ = _count_internal_subpeaks(masked_cube, mask)
                row["n_subpeaks"] = int(n_sub)
                if patch_enc is not None:
                    row["patch_dispersion"] = float(_patch_dispersion_score(masked_cube, mask, patch_enc, 64))
                if mv_encoder is not None:
                    triplets.append(orthogonal_mips(masked_cube))

            if mv_encoder is not None and spots:
                embs = mv_encoder.embed_batch(triplets, batch_size=32)
                for row, vec in zip(spots, embs, strict=True):
                    row["embedding"] = vec

            pair_shifts: list[list[float]] = []
            if seq_idx > 0 and prev_vol is not None and prev_spots:
                shifts, _ = estimate_slab_shifts_fft(
                    prev_vol,
                    vol,
                    prev_spots,
                    max_step_chi=max_step_chi,
                    max_step_d=max_step_d,
                    calibration=cal,
                )
                pair_shifts = [[round(float(a), 5), round(float(b), 5), round(float(c), 5)] for a, b, c in shifts]

            rec = {
                "seq_idx": int(seq_idx),
                "orig_scan_idx": int(orig_idx),
                "scan_name": sc_file.scan_name,
                "file_path": str(sc_file.path),
                "timestamp": meta.get("start_time"),
                "calibration": cal.to_dict(),
                "threshold_raw": round(raw_thr, 5),
                "threshold_causal": round(causal_thr, 5),
                "pair_shifts_from_prev": pair_shifts,
                "spots": [_serialize_spot_row(r) for r in spots],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            records.append(rec)
            prev_vol = vol
            prev_spots = spots
            prev_file = sc_file.path

    return records


def _sync_archive_stimuli(
    records: list[dict[str, Any]],
    operando_log_path: Path | None,
) -> dict[int, ScanStimulusPoint]:
    if operando_log_path is None or not operando_log_path.exists():
        return {}
    op_log = load_operando_log(operando_log_path)
    cal = AxisCalibration.from_dict(records[0].get("calibration", {})) if records else AxisCalibration()
    scan_metas = [
        ScanVolumeMeta(
            scan_name=str(r["scan_name"]),
            file_path=Path(str(r.get("file_path", r["scan_name"]))),
            sequence_index=int(r["seq_idx"]),
            start_time=BeamlineAdapter._parse_datetime(r.get("timestamp")),
            calibration=cal,
        )
        for r in records
    ]
    seq = ExperimentSequence(scans=tuple(scan_metas))
    pts = synchronize_to_scans(op_log, seq)
    return {int(p.scan_idx): p for p in pts}


def _evaluate_forecast_horizons(
    kin_list: list[Any],
    *,
    cal: AxisCalibration,
    stim_map: dict[int, ScanStimulusPoint],
    scan_elapsed_s: dict[int, float],
    horizons: tuple[int, ...] = (1, 2, 5),
) -> dict[str, Any]:
    step_floors = (abs(float(cal.mu_step)), abs(float(cal.chi_step)), abs(float(cal.d_step)))
    models = ["persistence", "constant_velocity_2pt", "kalman_cv", "clock_time_gp"]
    if stim_map:
        models.append("electro_state_gp")

    dqdv_abs_by_scan: dict[int, float] = {}
    for s_idx, st in stim_map.items():
        if st.dq_dv is not None and math.isfinite(float(st.dq_dv)):
            dqdv_abs_by_scan[s_idx] = abs(float(st.dq_dv))
    if not dqdv_abs_by_scan and len(stim_map) >= 2:
        sorted_s = sorted(stim_map)
        for s0, s1 in zip(sorted_s[:-1], sorted_s[1:], strict=False):
            p0, p1 = stim_map[s0], stim_map[s1]
            if (
                p0.capacity_mah_g is not None
                and p1.capacity_mah_g is not None
                and p0.voltage_v is not None
                and p1.voltage_v is not None
            ):
                dv = abs(float(p1.voltage_v) - float(p0.voltage_v))
                dq = abs(float(p1.capacity_mah_g) - float(p0.capacity_mah_g))
                if dv > 1e-5:
                    dqdv_abs_by_scan[s1] = dq / dv

    dqdv_q75 = float(np.percentile(list(dqdv_abs_by_scan.values()), 75.0)) if len(dqdv_abs_by_scan) >= 4 else None

    out_by_h: dict[str, Any] = {}
    for h in horizons:
        acc: dict[str, dict[str, list[float]]] = {
            m: {"err3d": [], "err3d_n3": [], "err_d": [], "cov1": [], "cov2": [], "high_dqdv": [], "low_dqdv": []}
            for m in models
        }
        for kin in kin_list:
            s_idx = list(kin.scan_indices)
            n_obs = len(s_idx)
            if n_obs < 2 + h:
                continue
            pos_by_scan = {int(si): pos for pos, si in enumerate(s_idx)}
            for end_pos in range(1, n_obs - h):
                cur_s = int(s_idx[end_pos])
                tgt_s = cur_s + h
                tgt_pos = pos_by_scan.get(tgt_s)
                if tgt_pos is None:
                    continue
                sub_s = s_idx[: end_pos + 1]
                if scan_elapsed_s and all(int(x) in scan_elapsed_s for x in sub_s) and tgt_s in scan_elapsed_s:
                    t_hist = np.asarray([float(scan_elapsed_s[int(x)]) for x in sub_s], dtype=np.float64)
                    t_target = float(scan_elapsed_s[tgt_s])
                else:
                    t_hist = np.asarray(sub_s, dtype=np.float64)
                    t_target = float(tgt_s)

                mu_h = np.asarray(kin.centroid_mu[: end_pos + 1], dtype=np.float64)
                chi_h = np.asarray(kin.centroid_chi[: end_pos + 1], dtype=np.float64)
                d_h = np.asarray(kin.centroid_d[: end_pos + 1], dtype=np.float64)
                em_h = np.asarray(kin.centroid_mu_err[: end_pos + 1], dtype=np.float64)
                ec_h = np.asarray(kin.centroid_chi_err[: end_pos + 1], dtype=np.float64)
                ed_h = np.asarray(kin.centroid_d_err[: end_pos + 1], dtype=np.float64)

                gt_mu = float(kin.centroid_mu[tgt_pos])
                gt_chi = float(kin.centroid_chi[tgt_pos])
                gt_d = float(kin.centroid_d[tgt_pos])
                is_high_dqdv = (
                    dqdv_q75 is not None and tgt_s in dqdv_abs_by_scan and dqdv_abs_by_scan[tgt_s] >= dqdv_q75
                )

                preds: dict[str, GPForecast] = {
                    "persistence": forecast_persistence(mu_h, chi_h, d_h, em_h, ec_h, ed_h, step_floors=step_floors),
                    "constant_velocity_2pt": forecast_constant_velocity_2pt(
                        t_hist, mu_h, chi_h, d_h, em_h, ec_h, ed_h, t_target, step_floors=step_floors
                    ),
                    "kalman_cv": forecast_kalman_cv(
                        t_hist, mu_h, chi_h, d_h, em_h, ec_h, ed_h, t_target, step_floors=step_floors
                    ),
                    "clock_time_gp": forecast_grain_trajectory_gp(
                        t_hist, mu_h, chi_h, d_h, em_h, ec_h, ed_h, t_target, step_floors=step_floors, domain="time"
                    ),
                }
                if "electro_state_gp" in acc:
                    causal_stim = {k: v for k, v in stim_map.items() if k <= cur_s}
                    preds["electro_state_gp"] = forecast_grain_trajectory_gp(
                        t_hist,
                        mu_h,
                        chi_h,
                        d_h,
                        em_h,
                        ec_h,
                        ed_h,
                        t_target,
                        step_floors=step_floors,
                        domain="electro_state",
                        stimuli=causal_stim,
                        scan_indices=sub_s,
                        lead_scans=float(h),
                    )

                for m_name, fc in preds.items():
                    dm = (fc.mean[0] - gt_mu) / max(step_floors[0], 1e-12)
                    dc = (fc.mean[1] - gt_chi) / max(step_floors[1], 1e-12)
                    dd = (fc.mean[2] - gt_d) / max(step_floors[2], 1e-12)
                    err3d = math.sqrt(dm * dm + dc * dc + dd * dd)
                    sm = max(fc.std[0] / max(step_floors[0], 1e-12), 1e-6)
                    sc = max(fc.std[1] / max(step_floors[1], 1e-12), 1e-6)
                    sd = max(fc.std[2] / max(step_floors[2], 1e-12), 1e-6)
                    z_rms = math.sqrt(((dm / sm) ** 2 + (dc / sc) ** 2 + (dd / sd) ** 2) / 3.0)

                    bucket = acc[m_name]
                    bucket["err3d"].append(err3d)
                    if len(sub_s) >= 3:
                        bucket["err3d_n3"].append(err3d)
                    bucket["err_d"].append(abs(dd))
                    bucket["cov1"].append(1.0 if z_rms <= 1.0 else 0.0)
                    bucket["cov2"].append(1.0 if z_rms <= 2.0 else 0.0)
                    if dqdv_q75 is not None and tgt_s in dqdv_abs_by_scan:
                        (bucket["high_dqdv"] if is_high_dqdv else bucket["low_dqdv"]).append(err3d)

        h_summary: dict[str, Any] = {}
        for m_name, b in acc.items():
            errs = np.asarray(b["err3d"], dtype=np.float64)
            errs_n3 = np.asarray(b["err3d_n3"], dtype=np.float64)
            h_summary[m_name] = {
                "n_evals": int(len(errs)),
                "mae_3d_steps": round(float(np.mean(errs)), 4) if len(errs) else None,
                "rmse_3d_steps": round(float(np.sqrt(np.mean(errs * errs))), 4) if len(errs) else None,
                "mae_3d_steps_n_ge_3": round(float(np.mean(errs_n3)), 4) if len(errs_n3) else None,
                "mae_d_steps": round(float(np.mean(b["err_d"])), 4) if len(b["err_d"]) else None,
                "coverage_1sigma": round(float(np.mean(b["cov1"])), 4) if len(b["cov1"]) else None,
                "coverage_2sigma": round(float(np.mean(b["cov2"])), 4) if len(b["cov2"]) else None,
                "high_dqdv_mae_3d": round(float(np.mean(b["high_dqdv"])), 4) if len(b["high_dqdv"]) else None,
                "low_dqdv_mae_3d": round(float(np.mean(b["low_dqdv"])), 4) if len(b["low_dqdv"]) else None,
            }
        out_by_h[f"h={h}"] = h_summary
    return out_by_h


def evaluate_cached_archive(
    records: list[dict[str, Any]],
    *,
    operando_log_path: Path | None = None,
    gt_links_path: Path | None = None,
    max_cost: float = 900.0,
    max_step_mu: float = 12.0,
    max_step_chi: float = 40.0,
    max_step_d: float = 40.0,
    horizons: tuple[int, ...] = (1, 2, 5),
) -> dict[str, Any]:
    """Run Pass 2 tracking, multi-horizon forecasting, and precursor ROC ablations from cached tables."""
    t0 = time.perf_counter()
    if not records:
        raise ValueError("Empty archive cache records.")

    cal = AxisCalibration.from_dict(records[0].get("calibration", {}))
    scan_tables = [[_deserialize_spot_row(s) for s in r["spots"]] for r in records]
    scan_timestamps = {int(r["seq_idx"]): r.get("timestamp") for r in records}
    pair_shifts: dict[int, list[tuple[float, float, float]]] = {}
    for idx in range(1, len(records)):
        raw_ps = records[idx].get("pair_shifts_from_prev", [])
        if raw_ps:
            pair_shifts[idx - 1] = [(float(p[0]), float(p[1]), float(p[2])) for p in raw_ps]

    stim_map = _sync_archive_stimuli(records, operando_log_path)
    scan_elapsed_s = {s: float(st.elapsed_s) for s, st in stim_map.items() if math.isfinite(float(st.elapsed_s))}

    gt_payload: dict[str, Any] | None = None
    if gt_links_path is not None and gt_links_path.exists():
        gt_payload = json.loads(gt_links_path.read_text(encoding="utf-8"))

    has_dino = any("embedding" in sp for tbl in scan_tables for sp in tbl)
    cost_arms: list[tuple[str, CostFunction]] = [
        (
            "position_shape",
            GeometrySemanticCost(
                PositionShapeCost(
                    max_step_mu=max_step_mu,
                    max_step_chi=max_step_chi,
                    max_step_d=max_step_d,
                    intensity_weight=15.0,
                ),
                cost_alpha=1.0,
                cost_beta=0.0,
            ),
        ),
        (
            "wasserstein_bures",
            GeometrySemanticCost(
                WassersteinBuresCost(max_step_mu=max_step_mu, max_step_chi=max_step_chi, max_step_d=max_step_d),
                cost_alpha=1.0,
                cost_beta=0.0,
            ),
        ),
    ]
    if has_dino:
        cost_arms.append(
            (
                "wasserstein_bures+dino",
                GeometrySemanticCost(
                    WassersteinBuresCost(max_step_mu=max_step_mu, max_step_chi=max_step_chi, max_step_d=max_step_d),
                    cost_alpha=1.0,
                    cost_beta=40.0,
                ),
            )
        )

    drift_modes: tuple[DriftMode, ...] = ("none", "consensus", "fft", "fft_kalman")
    multi_slice_counts = [sum(1 for sp in tbl if not is_single_slice_speck(sp)) for tbl in scan_tables]
    med_spots = max(1.0, float(np.median(multi_slice_counts)) if multi_slice_counts else 1.0)

    tracking_grid: list[dict[str, Any]] = []
    primary_graph = None
    primary_rows: list[dict[str, Any]] = []

    for d_mode in drift_modes:
        for c_name, c_fn in cost_arms:
            G = build_tracks(
                scan_tables,
                cost_fn=c_fn,
                max_cost=max_cost,
                split_merge_config=SplitMergeConfig(max_distance=25.0),
                scan_timestamps=scan_timestamps,
                drift_mode=d_mode,
                precomputed_pair_shifts=pair_shifts if d_mode in ("fft", "fft_kalman") else None,
                scan_elapsed_s=scan_elapsed_s if scan_elapsed_s else None,
            )
            t_rows = tracks_to_table(G)
            by_tid: dict[int, int] = {}
            for r in t_rows:
                by_tid[int(r["track_id"])] = by_tid.get(int(r["track_id"]), 0) + 1
            lengths = list(by_tid.values())
            multi_tracks = [ln for ln in lengths if ln >= 2]
            events = list(G.graph.get("lineage_events", []))
            n_mech_splits = sum(
                1 for e in events if e.event_type == "split" and e.split_subtype != "reversible_overlap"
            )
            n_rev_overlaps = sum(
                1 for e in events if e.event_type == "split" and e.split_subtype == "reversible_overlap"
            )

            entry: dict[str, Any] = {
                "drift_mode": d_mode,
                "cost_model": c_name,
                "n_tracks_total": len(lengths),
                "n_tracks_len_ge_2": len(multi_tracks),
                "n_tracks_len_ge_3": sum(1 for ln in lengths if ln >= 3),
                "mean_multi_track_len": round(float(np.mean(multi_tracks)), 3) if multi_tracks else 0.0,
                "fragmentation_ratio": round(float(len(multi_tracks)) / med_spots, 3),
                "n_persistent_splits": int(n_mech_splits),
                "n_reversible_overlaps": int(n_rev_overlaps),
            }
            if gt_payload is not None:
                from braggtrack.tracking.gt_eval import score_graph_against_gt

                gt_res = score_graph_against_gt(G, gt_payload)
                entry["gt_overall_f1"] = gt_res["overall_1to1"]["f1"]
                entry["gt_isolated_f1"] = gt_res["isolated_1to1"]["f1"]
                entry["gt_crowded_f1"] = gt_res["crowded_1to1"]["f1"]

            tracking_grid.append(entry)
            if d_mode == "fft_kalman" and c_name == "wasserstein_bures":
                primary_graph = G
                primary_rows = t_rows

    assert primary_graph is not None
    kin_list = compute_grain_kinematics(primary_rows, scan_elapsed_s=scan_elapsed_s if scan_elapsed_s else None)
    forecast_summary = _evaluate_forecast_horizons(
        kin_list,
        cal=cal,
        stim_map=stim_map,
        scan_elapsed_s=scan_elapsed_s,
        horizons=horizons,
    )

    primary_events = list(primary_graph.graph.get("lineage_events", []))
    candidates_base, obs_scores = compute_fracture_precursors(
        primary_rows,
        PrecursorConfig(baseline_reference=True, baseline_scans=2),
        lineage_events=primary_events,
    )
    roc_metrics = evaluate_precursor_channels_roc(
        obs_scores,
        primary_events,
        scan_stimuli=stim_map if stim_map else None,
        lookback_scans=8,
        min_daughter_persistence=1,
        exclude_reversible_overlap=True,
    )

    return {
        "n_scans_evaluated": len(records),
        "median_multi_slice_spots_per_scan": round(med_spots, 2),
        "has_operando_stimulus": bool(stim_map),
        "has_blind_gt_links": gt_payload is not None,
        "pass2_elapsed_sec": round(time.perf_counter() - t0, 4),
        "tracking_grid": tracking_grid,
        "forecasting_horizons": forecast_summary,
        "precursor_channels_roc": {k: v.to_dict() for k, v in roc_metrics.items()},
        "n_ranked_precursor_candidates": len(candidates_base),
    }


def _fmt_opt(val: float | None, digits: int = 3) -> str:
    return f"{val:.{digits}f}" if val is not None else "n/a"


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    """Format the archive evaluation dictionary into a structured Markdown report."""
    lines = [
        "# BraggTrack Operando Archive Evaluation Report",
        "",
        f"- **Scans evaluated:** `{report['n_scans_evaluated']}`",
        f"- **Median multi-slice reflections / scan:** `{report['median_multi_slice_spots_per_scan']}`",
        f"- **Synchronized electrochemistry:** `{report['has_operando_stimulus']}`",
        f"- **Blind human ground truth attached:** `{report['has_blind_gt_links']}`",
        f"- **Pass 2 compute time:** `{report['pass2_elapsed_sec']} s`",
        "",
        "## 1. Tracking Ablation Grid (Drift Compensation x Association Cost)",
        "",
    ]
    has_gt = bool(report.get("has_blind_gt_links"))
    if has_gt:
        lines.append(
            "| Drift Mode | Cost Model | Tracks (>=2) | Tracks (>=3) | Mean Len | Splits | Overlaps | Overall F1 | Crowded F1 |"
        )
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    else:
        lines.append(
            "| Drift Mode | Cost Model | Tracks (>=2) | Tracks (>=3) | Mean Len | Frag Ratio | Splits | Overlaps |"
        )
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for row in report["tracking_grid"]:
        if has_gt:
            lines.append(
                f"| `{row['drift_mode']}` | `{row['cost_model']}` | {row['n_tracks_len_ge_2']} | "
                f"{row['n_tracks_len_ge_3']} | {row['mean_multi_track_len']:.2f} | "
                f"{row['n_persistent_splits']} | {row['n_reversible_overlaps']} | "
                f"{row.get('gt_overall_f1', 0.0):.4f} | {row.get('gt_crowded_f1', 0.0):.4f} |"
            )
        else:
            lines.append(
                f"| `{row['drift_mode']}` | `{row['cost_model']}` | {row['n_tracks_len_ge_2']} | "
                f"{row['n_tracks_len_ge_3']} | {row['mean_multi_track_len']:.2f} | {row['fragmentation_ratio']:.2f} | "
                f"{row['n_persistent_splits']} | {row['n_reversible_overlaps']} |"
            )

    lines.extend(["", "## 2. Multi-Horizon Causal Forecasting Benchmark", ""])
    lines.append(
        "| Horizon | Forecaster | N | 3D MAE (steps) | 3D MAE (n>=3) | 1-sigma Cov | 2-sigma Cov | High dQ/dV MAE | Low dQ/dV MAE |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for h_key, models in report["forecasting_horizons"].items():
        for m_name, stats in models.items():
            lines.append(
                f"| `{h_key}` | `{m_name}` | {stats['n_evals']} | {_fmt_opt(stats['mae_3d_steps'])} | "
                f"{_fmt_opt(stats['mae_3d_steps_n_ge_3'])} | {_fmt_opt(stats['coverage_1sigma'])} | "
                f"{_fmt_opt(stats['coverage_2sigma'])} | {_fmt_opt(stats['high_dqdv_mae_3d'])} | "
                f"{_fmt_opt(stats['low_dqdv_mae_3d'])} |"
            )

    lines.extend(["", "## 3. Fracture Precursor Channel ROC / PR Benchmark", ""])
    lines.append(
        "| Channel | AUROC | AUPRC | Pos Obs | Neg Obs | Splits | False Alarms / 100 | Mean Lead (scans) | Mean Lead (mV) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ch, m in report["precursor_channels_roc"].items():
        lines.append(
            f"| `{ch}` | {_fmt_opt(m['auroc'], 4)} | {_fmt_opt(m['auprc'], 4)} | {m['n_positive_obs']} | "
            f"{m['n_negative_obs']} | {m['n_fracture_events']} | {m['false_alarms_per_100_obs']:.2f} | "
            f"{_fmt_opt(m['mean_lead_scans'], 2)} | {_fmt_opt(m['mean_lead_mv'], 2)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-pass O(1)-RAM operando archive evaluation harness (tracking, forecasting, precursors)."
    )
    parser.add_argument("--root", type=Path, default=None, help="Directory of HDF5 rocking-curve scans")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for cache and reports")
    parser.add_argument(
        "--cache-only-pass2", action="store_true", help="Skip Pass 1 and run Pass 2 from existing cache"
    )
    parser.add_argument("--no-resume", action="store_true", help="Rebuild archive_cache.jsonl from scratch")
    parser.add_argument("--start-scan", type=int, default=0, help="0-based starting scan index")
    parser.add_argument("--end-scan", type=int, default=None, help="Exclusive ending scan index")
    parser.add_argument("--stride", type=int, default=1, help="Scan subsampling stride")
    parser.add_argument("--bg-sigma", type=float, default=5.0, help="Detector MAD background multiplier")
    parser.add_argument("--min-spot-voxels", type=int, default=6, help="Minimum spot volume in voxels")
    parser.add_argument("--anisotropic", action="store_true", help="Enable aspect-aware anisotropic LoG seeding")
    parser.add_argument("--include-dino", action="store_true", help="Include DINOv2 morphological ablation arm")
    parser.add_argument("--dino-backend", type=str, default="auto", help="DINOv2 backend ('auto', 'torch', 'mock')")
    parser.add_argument("--operando-log", type=Path, default=None, help="Optional potentiostat CSV/TSV/.mpt log")
    parser.add_argument("--gt-links", type=Path, default=None, help="Optional blind ground_truth_links.json")
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.outdir / "archive_cache.jsonl"

    if args.cache_only_pass2:
        if not cache_path.exists():
            raise FileNotFoundError(f"--cache-only-pass2 requested, but {cache_path} does not exist")
        records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        if args.root is None:
            raise ValueError("--root is required unless --cache-only-pass2 is set")
        records = stream_archive_cache(
            args.root,
            cache_path,
            start_scan=args.start_scan,
            end_scan=args.end_scan,
            stride=args.stride,
            bg_sigma=args.bg_sigma,
            min_spot_voxels=args.min_spot_voxels,
            anisotropic=args.anisotropic,
            include_dino=args.include_dino,
            dino_backend=args.dino_backend,
            resume=not args.no_resume,
        )

    report = evaluate_cached_archive(
        records,
        operando_log_path=args.operando_log,
        gt_links_path=args.gt_links,
    )
    json_path = args.outdir / "archive_evaluation_report.json"
    md_path = args.outdir / "archive_evaluation_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_evaluation_markdown(report), encoding="utf-8")
    print(f"Archive evaluation complete ({report['n_scans_evaluated']} scans, Pass 2={report['pass2_elapsed_sec']}s)")
    print(f"  JSON -> {json_path}")
    print(f"  MD   -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
