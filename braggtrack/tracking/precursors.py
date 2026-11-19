"""Chemomechanical fracture precursor scoring with multi-scan persistence filtering.

Detects intragranular mosaicity buildup and incipient sub-grain splitting before
a Bragg reflection cleaves into separate connected components, combining:
1. 3D covariance elongation (eigenvalue anisotropy eig_1 / eig_3),
2. Relative reciprocal-space broadening (covariance trace growth), and
3. Intra-spot ROI sub-peak bimodality + DINO patch-feature dispersion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.feature import peak_local_max
from sklearn.metrics import average_precision_score, roc_auc_score

from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips

if TYPE_CHECKING:
    from braggtrack.io.operando_log import ScanStimulusPoint
    from braggtrack.semantic.dino import PatchFeatureEncoder
    from braggtrack.tracking.events import LineageEvent


@dataclass(frozen=True)
class PrecursorConfig:
    """Configuration for fracture precursor scoring and persistence filtering."""

    score_threshold: float = 0.40
    min_persistence: int = 2
    anisotropy_scale: float = 3.0
    spread_growth_scale: float = 1.0
    weight_anisotropy: float = 0.35
    weight_spread: float = 0.25
    weight_roi: float = 0.40
    roi_margin: int = 2
    roi_upsample_size: int = 64
    baseline_reference: bool = False
    baseline_scans: int = 2
    delta_log_anisotropy_scale: float = 0.70
    skewness_scale: float = 0.75
    kurtosis_scale: float = 1.0
    weight_skewness: float = 0.15
    weight_kurtosis: float = 0.15


@dataclass(frozen=True)
class ObservationPrecursorScore:
    """Per-observation precursor diagnostics for one spot in one scan."""

    track_id: int
    scan_idx: int
    label: int
    anisotropy: float
    anisotropy_term: float
    spread_growth: float
    spread_term: float
    roi_subpeak_count: int
    roi_dispersion_term: float
    precursor_score: float
    delta_log_anisotropy: float = 0.0
    delta_skewness: float = 0.0
    platykurtic_drop: float = 0.0
    leptokurtic_wing: float = 0.0
    subpeak_score: float = 0.0
    patch_dispersion: float = 0.0
    baseline_referenced_score: float = 0.0


@dataclass(frozen=True)
class PrecursorCandidate:
    """Ranked grain track exhibiting persistent fracture precursor signatures."""

    rank: int
    track_id: int
    onset_scan: int
    peak_scan: int
    persistence_scans: int
    peak_score: float
    mean_active_score: float
    max_anisotropy: float
    max_spread_growth: float
    max_subpeak_count: int
    confidence: float
    subsequent_split: bool


@dataclass(frozen=True)
class PrecursorChannelMetrics:
    """ROC/PR and false-alarm benchmark metrics for an individual precursor channel."""

    channel: str
    auroc: float | None
    auprc: float | None
    n_positive_obs: int
    n_negative_obs: int
    n_fracture_events: int
    false_alarms_per_100_obs: float
    recall_at_alarm_threshold: float | None = None
    mean_lead_scans: float | None = None
    mean_lead_mv: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "auroc": round(float(self.auroc), 4) if self.auroc is not None else None,
            "auprc": round(float(self.auprc), 4) if self.auprc is not None else None,
            "n_positive_obs": int(self.n_positive_obs),
            "n_negative_obs": int(self.n_negative_obs),
            "n_fracture_events": int(self.n_fracture_events),
            "false_alarms_per_100_obs": round(float(self.false_alarms_per_100_obs), 3),
            "recall_at_alarm_threshold": (
                round(float(self.recall_at_alarm_threshold), 4) if self.recall_at_alarm_threshold is not None else None
            ),
            "mean_lead_scans": round(float(self.mean_lead_scans), 3) if self.mean_lead_scans is not None else None,
            "mean_lead_mv": round(float(self.mean_lead_mv), 2) if self.mean_lead_mv is not None else None,
        }


def _count_internal_subpeaks(masked_cube: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    """Count distinct internal intensity maxima inside a segmented 3D spot ROI."""
    fg = mask > 0
    if int(fg.sum()) < 4:
        return 1, 0.0

    vals = masked_cube[fg]
    vmax = float(vals.max())
    vmin = float(vals.min())
    if vmax <= 0.0 or vmax <= vmin:
        return 1, 0.0

    # Subtract 85% of foreground boundary floor so additive camera pedestals do not inflate flank ratios,
    # and scale smoothing / peak separation proportionally on large (>20 px) near-field reflections.
    net_cube = np.where(fg, np.maximum(0.0, masked_cube.astype(np.float64) - 0.85 * max(0.0, vmin)), 0.0)
    max_span = float(max(masked_cube.shape))
    sigma_sub = max(0.7, 0.035 * max_span)
    min_dist = max(2, int(round(0.12 * max_span)))

    smoothed = gaussian_filter(net_cube, sigma=sigma_sub)
    smax = float(smoothed[fg].max())
    if smax <= 0.0:
        return 1, 0.0

    coords = peak_local_max(
        smoothed,
        min_distance=min_dist,
        threshold_abs=0.25 * smax,
        labels=fg.astype(np.uint8),
        num_peaks=32,
    )
    n_peaks = int(len(coords))
    if n_peaks <= 1:
        return 1, 0.0

    subpeak_score = float(np.clip(0.65 + 0.20 * (n_peaks - 2), 0.0, 1.0))
    return n_peaks, subpeak_score


def _dispersion_from_fmap(fmap: np.ndarray, mm_up: np.ndarray) -> float | None:
    """Compute foreground patch-token cosine dispersion for one upsampled MIP feature map."""
    hp, wp, dim = fmap.shape
    if hp * wp < 2:
        return None

    bh = max(1, mm_up.shape[0] // hp)
    bw = max(1, mm_up.shape[1] // wp)
    cropped_fg = (mm_up[: hp * bh, : wp * bw] > 0).reshape(hp, bh, wp, bw).max(axis=(1, 3))
    fg_vecs = fmap[cropped_fg] if int(cropped_fg.sum()) >= 2 else fmap.reshape(-1, dim)
    if fg_vecs.shape[0] < 2:
        return None

    mean_vec = fg_vecs.mean(axis=0)
    nrm = float(np.linalg.norm(mean_vec))
    if nrm <= 0.0:
        return None
    mean_vec = mean_vec / nrm
    cos_sims = np.clip(fg_vecs @ mean_vec, -1.0, 1.0)
    cos_disp = float(1.0 - np.mean(cos_sims))
    return float(np.clip(cos_disp / 0.6, 0.0, 1.0))


def _prepare_upsampled_roi_mips(
    masked_cube: np.ndarray,
    mask: np.ndarray,
    upsample_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return upsampled ``(mip_up, mask_up)`` pairs for non-empty orthogonal views."""
    mip_mu, mip_chi, mip_d = orthogonal_mips(masked_cube)
    mask_mu, mask_chi, mask_d = orthogonal_mips(mask.astype(np.float64))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for mip, mm in ((mip_mu, mask_mu), (mip_chi, mask_chi), (mip_d, mask_d)):
        if mip.size == 0 or float(mm.max()) <= 0.0:
            continue
        h, w = mip.shape
        zh = max(1.0, upsample_size / max(1, h))
        zw = max(1.0, upsample_size / max(1, w))
        mip_up = zoom(mip.astype(np.float64), (zh, zw), order=1) if (zh > 1.0 or zw > 1.0) else mip.astype(np.float64)
        mm_up = zoom(mm.astype(np.float64), (zh, zw), order=0) if (zh > 1.0 or zw > 1.0) else mm.astype(np.float64)
        pairs.append((mip_up, mm_up))
    return pairs


def _patch_dispersion_score(
    masked_cube: np.ndarray,
    mask: np.ndarray,
    patch_encoder: PatchFeatureEncoder,
    upsample_size: int,
) -> float:
    """Compute mean foreground patch-token cosine dispersion across orthogonal MIPs."""
    pairs = _prepare_upsampled_roi_mips(masked_cube, mask, upsample_size)
    if not pairs:
        return 0.0
    mips = [p[0] for p in pairs]
    batch_fn = getattr(patch_encoder, "extract_patch_features_batch", None)
    fmaps = batch_fn(mips) if callable(batch_fn) else [patch_encoder.extract_patch_features(m) for m in mips]
    dispersions: list[float] = []
    for fmap, (_, mm_up) in zip(fmaps, pairs, strict=True):
        val = _dispersion_from_fmap(fmap, mm_up)
        if val is not None:
            dispersions.append(val)
    return float(np.mean(dispersions)) if dispersions else 0.0


def compute_fracture_precursors(
    track_table: list[dict[str, Any]],
    config: PrecursorConfig | None = None,
    *,
    volumes: dict[int, np.ndarray] | list[np.ndarray] | None = None,
    label_volumes: dict[int, np.ndarray] | list[np.ndarray] | None = None,
    patch_encoder: PatchFeatureEncoder | None = None,
    lineage_events: list[LineageEvent] | None = None,
    roi_cache: dict[tuple[int, int], Any] | None = None,
    batch_size: int = 32,
) -> tuple[list[PrecursorCandidate], list[ObservationPrecursorScore]]:
    """Score per-grain fracture precursors and return persistent ranked candidates.

    Parameters
    ----------
    track_table
        Row-per-observation trajectory table from :func:`tracks_to_table`.
    config
        Thresholds, weights, and persistence configuration.
    volumes, label_volumes
        Optional mapping (or scan-indexed list) of 3D intensity and integer label
        volumes enabling intra-spot sub-peak and DINO patch-dispersion analysis.
    patch_encoder
        Optional :class:`PatchFeatureEncoder` for MIP patch dispersion.
    lineage_events
        Optional list of :class:`LineageEvent` records used to annotate whether a
        precursor candidate subsequently fractured (``subsequent_split``).
    roi_cache
        Optional mutable ``(scan_idx, label) -> (n_subpeaks, roi_term[, subpeak_score, patch_disp])``
        cache for incremental online streaming so historical scans are not re-cropped.
    batch_size
        Mini-batch size when running ``patch_encoder.extract_patch_features_batch``.
    """
    cfg = config or PrecursorConfig()
    if not track_table:
        return [], []

    vol_map: dict[int, np.ndarray] = {}
    if isinstance(volumes, dict):
        vol_map = volumes
    elif isinstance(volumes, list):
        vol_map = dict(enumerate(volumes))

    lab_map: dict[int, np.ndarray] = {}
    if isinstance(label_volumes, dict):
        lab_map = label_volumes
    elif isinstance(label_volumes, list):
        lab_map = dict(enumerate(label_volumes))

    has_roi = bool(vol_map and lab_map)
    effective_cache: dict[tuple[int, int], Any] = roi_cache if roi_cache is not None else {}

    # Pre-compute uncached ROI sub-peaks and batch all DINO patch-feature extractions together.
    if has_roi:
        pending_keys: list[tuple[int, int]] = []
        pending_subpeaks: list[tuple[int, float]] = []
        pending_view_slices: list[tuple[int, int]] = []
        all_mips: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []

        for row in track_table:
            s_idx = int(row["scan_idx"])
            lbl_i = int(row.get("label", 0))
            key = (s_idx, lbl_i)
            if lbl_i <= 0 or "bbox_min_z" not in row or key in effective_cache:
                continue
            if s_idx not in vol_map or s_idx not in lab_map:
                continue
            effective_cache[key] = (1, 0.0, 0.0, 0.0)
            masked_cube, mask = crop_spot_cube(
                vol_map[s_idx],
                lab_map[s_idx],
                lbl_i,
                row,
                margin=cfg.roi_margin,
            )
            n_sub, sub_term = _count_internal_subpeaks(masked_cube, mask)
            if patch_encoder is None:
                effective_cache[key] = (n_sub, sub_term, sub_term, 0.0)
            else:
                pairs = _prepare_upsampled_roi_mips(masked_cube, mask, cfg.roi_upsample_size)
                v_start = len(all_mips)
                for mip_up, mm_up in pairs:
                    all_mips.append(mip_up)
                    all_masks.append(mm_up)
                v_end = len(all_mips)
                pending_keys.append(key)
                pending_subpeaks.append((n_sub, sub_term))
                pending_view_slices.append((v_start, v_end))

        if patch_encoder is not None and pending_keys:
            batch_fn = getattr(patch_encoder, "extract_patch_features_batch", None)
            if callable(batch_fn):
                all_fmaps = batch_fn(all_mips, batch_size=batch_size) if all_mips else []
            else:
                all_fmaps = [patch_encoder.extract_patch_features(m) for m in all_mips]

            for key, (n_sub, sub_term), (v_start, v_end) in zip(
                pending_keys, pending_subpeaks, pending_view_slices, strict=True
            ):
                dispersions: list[float] = []
                for v_idx in range(v_start, v_end):
                    val = _dispersion_from_fmap(all_fmaps[v_idx], all_masks[v_idx])
                    if val is not None:
                        dispersions.append(val)
                p_disp = float(np.mean(dispersions)) if dispersions else 0.0
                roi_val = float(np.clip(0.65 * sub_term + 0.35 * p_disp, 0.0, 1.0))
                effective_cache[key] = (n_sub, roi_val, sub_term, p_disp)

    split_parent_tids: set[int] = set()
    if lineage_events:
        for ev in lineage_events:
            if ev.event_type == "split" and getattr(ev, "split_subtype", None) != "reversible_overlap":
                split_parent_tids.update(ev.parent_track_ids)

    by_track: dict[int, list[dict[str, Any]]] = {}
    for row in track_table:
        by_track.setdefault(int(row["track_id"]), []).append(row)

    all_obs_scores: list[ObservationPrecursorScore] = []
    unranked: list[PrecursorCandidate] = []

    def _obs_aniso_and_trace(
        obs_row: dict[str, Any],
        *,
        mixed_u: bool,
        q_floors: tuple[float, float, float],
        c0_vals: tuple[float, float, float],
    ) -> tuple[float, float]:
        eig1 = max(0.0, float(obs_row.get("eig_1", 0.0)))
        eig2 = max(0.0, float(obs_row.get("eig_2", 0.0)))
        eig3 = max(0.0, float(obs_row.get("eig_3", 0.0)))
        if mixed_u:
            r_mu = max(q_floors[0], float(obs_row.get("cov_zz", 0.0))) / c0_vals[0]
            r_chi = max(q_floors[1], float(obs_row.get("cov_xx", 0.0))) / c0_vals[1]
            r_d = max(q_floors[2], float(obs_row.get("cov_yy", 0.0))) / c0_vals[2]
            r_max = max(r_mu, r_chi, r_d)
            r_min = max(min(r_mu, r_chi, r_d), 1e-9)
            return r_max / r_min, (r_mu + r_chi + r_d) / 3.0
        an = eig1 / eig3 if eig3 > 1e-9 else (1.0 if eig1 <= 1e-9 else 10.0)
        return an, eig1 + eig2 + eig3

    for tid in sorted(by_track):
        obs_list = sorted(by_track[tid], key=lambda r: int(r["scan_idx"]))
        eig1_0 = max(0.0, float(obs_list[0].get("eig_1", 0.0)))
        eig2_0 = max(0.0, float(obs_list[0].get("eig_2", 0.0)))
        eig3_0 = max(0.0, float(obs_list[0].get("eig_3", 0.0)))
        e0 = eig1_0 + eig2_0 + eig3_0
        aniso_0 = eig1_0 / eig3_0 if eig3_0 > 1e-12 else 1.0
        q_floor_mu = max(1e-12, (float(obs_list[0].get("step_mu", 1.0)) ** 2) / 12.0)
        q_floor_chi = max(1e-12, (float(obs_list[0].get("step_chi", 1.0)) ** 2) / 12.0)
        q_floor_d = max(1e-12, (float(obs_list[0].get("step_d", 1.0)) ** 2) / 12.0)
        q_floors = (q_floor_mu, q_floor_chi, q_floor_d)
        c0_mu = max(q_floor_mu, float(obs_list[0].get("cov_zz", 1.0)))
        c0_chi = max(q_floor_chi, float(obs_list[0].get("cov_xx", 1.0)))
        c0_d = max(q_floor_d, float(obs_list[0].get("cov_yy", 1.0)))
        c0_vals = (c0_mu, c0_chi, c0_d)
        has_cov = all(k in obs_list[0] for k in ("cov_zz", "cov_xx", "cov_yy"))
        has_bbox = all(
            k in obs_list[0]
            for k in ("bbox_min_z", "bbox_max_z", "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y")
        )
        if has_cov and has_bbox:
            dz0 = max(float(obs_list[0]["bbox_max_z"]) - float(obs_list[0]["bbox_min_z"]), 1.0)
            dx0 = max(float(obs_list[0]["bbox_max_x"]) - float(obs_list[0]["bbox_min_x"]), 1.0)
            dy0 = max(float(obs_list[0]["bbox_max_y"]) - float(obs_list[0]["bbox_min_y"]), 1.0)
            scales2 = [c0_mu / (dz0 * dz0), c0_chi / (dx0 * dx0), c0_d / (dy0 * dy0)]
            mixed_units = (max(scales2) / max(min(scales2), 1e-18)) > 25.0 or aniso_0 > 15.0
        else:
            mixed_units = has_cov and aniso_0 > 15.0

        # Pristine per-grain baseline across the first b scans (strictly causal: uses <= scan_idx)
        b_len = max(1, min(len(obs_list), int(cfg.baseline_scans)))
        base_aniso_logs: list[float] = []
        base_traces: list[float] = []
        base_skews: list[tuple[float, float, float]] = []
        base_kurts: list[tuple[float, float, float]] = []
        for b_obs in obs_list[:b_len]:
            b_an, b_tr = _obs_aniso_and_trace(b_obs, mixed_u=mixed_units, q_floors=q_floors, c0_vals=c0_vals)
            base_aniso_logs.append(float(np.log(max(b_an, 1.0))))
            base_traces.append(float(b_tr))
            base_skews.append(
                (
                    float(b_obs.get("skewness_mu", 0.0)),
                    float(b_obs.get("skewness_chi", 0.0)),
                    float(b_obs.get("skewness_d", 0.0)),
                )
            )
            base_kurts.append(
                (
                    float(b_obs.get("kurtosis_mu", 3.0)),
                    float(b_obs.get("kurtosis_chi", 3.0)),
                    float(b_obs.get("kurtosis_d", 3.0)),
                )
            )
        base_log_aniso_med = float(np.median(base_aniso_logs))
        base_trace_med = float(np.median(base_traces))
        base_skew_med = np.median(np.asarray(base_skews, dtype=np.float64), axis=0)
        base_kurt_med = np.median(np.asarray(base_kurts, dtype=np.float64), axis=0)

        track_scores: list[ObservationPrecursorScore] = []

        for idx_obs, obs in enumerate(obs_list):
            scan_idx = int(obs["scan_idx"])
            lbl = int(obs.get("label", 0))

            # Strictly causal baseline window up to min(idx_obs + 1, b_len)
            if idx_obs + 1 < b_len:
                cur_b = idx_obs + 1
                cur_base_log_aniso = float(np.median(base_aniso_logs[:cur_b]))
                cur_base_trace = float(np.median(base_traces[:cur_b]))
                cur_base_skew = np.median(np.asarray(base_skews[:cur_b], dtype=np.float64), axis=0)
                cur_base_kurt = np.median(np.asarray(base_kurts[:cur_b], dtype=np.float64), axis=0)
            else:
                cur_base_log_aniso = base_log_aniso_med
                cur_base_trace = base_trace_med
                cur_base_skew = base_skew_med
                cur_base_kurt = base_kurt_med

            aniso, trace_now = _obs_aniso_and_trace(obs, mixed_u=mixed_units, q_floors=q_floors, c0_vals=c0_vals)
            if mixed_units:
                spread_growth = trace_now - 1.0
                base_spread_growth = (trace_now - cur_base_trace) / max(cur_base_trace, 1e-9)
            else:
                spread_growth = (trace_now - e0) / e0 if e0 > 1e-9 else 0.0
                base_spread_growth = (trace_now - cur_base_trace) / cur_base_trace if cur_base_trace > 1e-9 else 0.0

            aniso_term = float(np.clip((aniso - 1.0) / max(cfg.anisotropy_scale, 1e-6), 0.0, 1.0))
            spread_term = float(np.clip(max(0.0, spread_growth) / max(cfg.spread_growth_scale, 1e-6), 0.0, 1.0))

            # Higher-order baseline-referenced morphological channels (Trial 5)
            delta_log_aniso = max(0.0, float(np.log(max(aniso, 1.0))) - cur_base_log_aniso)
            skew_vec = np.array(
                [
                    float(obs.get("skewness_mu", cur_base_skew[0])),
                    float(obs.get("skewness_chi", cur_base_skew[1])),
                    float(obs.get("skewness_d", cur_base_skew[2])),
                ],
                dtype=np.float64,
            )
            delta_skew = float(np.linalg.norm(skew_vec - cur_base_skew))
            kurt_vec = np.array(
                [
                    float(obs.get("kurtosis_mu", cur_base_kurt[0])),
                    float(obs.get("kurtosis_chi", cur_base_kurt[1])),
                    float(obs.get("kurtosis_d", cur_base_kurt[2])),
                ],
                dtype=np.float64,
            )
            platy_drop = float(np.max(np.maximum(0.0, cur_base_kurt - kurt_vec)))
            lepto_wing = float(np.max(np.maximum(0.0, kurt_vec - cur_base_kurt)))

            n_subpeaks = 1
            roi_term = 0.0
            subpeak_sc = float(
                np.clip(0.65 + 0.20 * (int(obs.get("n_subpeaks", 1)) - 2), 0.0, 1.0)
                if int(obs.get("n_subpeaks", 1)) >= 2
                else 0.0
            )
            patch_disp = float(obs.get("patch_dispersion", 0.0))

            use_roi_here = lbl > 0 and (
                (scan_idx, lbl) in effective_cache or (has_roi and scan_idx in vol_map and scan_idx in lab_map)
            )
            if use_roi_here and "bbox_min_z" in obs:
                cached_entry = effective_cache.get((scan_idx, lbl), (1, 0.0, 0.0, 0.0))
                if len(cached_entry) >= 4:
                    n_subpeaks = int(cached_entry[0])
                    roi_term = float(cached_entry[1])
                    subpeak_sc = float(cached_entry[2])
                    patch_disp = float(cached_entry[3])
                else:
                    n_subpeaks = int(cached_entry[0])
                    roi_term = float(cached_entry[1])
                    subpeak_sc = roi_term
                w_sum = cfg.weight_anisotropy + cfg.weight_spread + cfg.weight_roi
                raw_score = (
                    cfg.weight_anisotropy * aniso_term + cfg.weight_spread * spread_term + cfg.weight_roi * roi_term
                ) / max(w_sum, 1e-6)
                has_roi_signal = True
            elif int(obs.get("n_subpeaks", 0)) > 0:
                n_subpeaks = int(obs.get("n_subpeaks", 1))
                roi_term = (
                    float(np.clip(0.65 * subpeak_sc + 0.35 * patch_disp, 0.0, 1.0)) if patch_disp > 0 else subpeak_sc
                )
                w_sum = cfg.weight_anisotropy + cfg.weight_spread + cfg.weight_roi
                raw_score = (
                    cfg.weight_anisotropy * aniso_term + cfg.weight_spread * spread_term + cfg.weight_roi * roi_term
                ) / max(w_sum, 1e-6)
                has_roi_signal = True
            else:
                w_geo = cfg.weight_anisotropy + cfg.weight_spread
                raw_score = (cfg.weight_anisotropy * aniso_term + cfg.weight_spread * spread_term) / max(w_geo, 1e-6)
                has_roi_signal = False

            d_aniso_term = float(np.clip(delta_log_aniso / max(cfg.delta_log_anisotropy_scale, 1e-6), 0.0, 1.0))
            b_spread_term = float(np.clip(max(0.0, base_spread_growth) / max(cfg.spread_growth_scale, 1e-6), 0.0, 1.0))
            skew_term = float(np.clip(delta_skew / max(cfg.skewness_scale, 1e-6), 0.0, 1.0))
            kurt_term = float(np.clip(max(platy_drop, 0.5 * lepto_wing) / max(cfg.kurtosis_scale, 1e-6), 0.0, 1.0))
            w_base_sum = (
                cfg.weight_anisotropy
                + cfg.weight_spread
                + cfg.weight_skewness
                + cfg.weight_kurtosis
                + (cfg.weight_roi if has_roi_signal else 0.0)
            )
            base_ref_raw = (
                cfg.weight_anisotropy * d_aniso_term
                + cfg.weight_spread * b_spread_term
                + cfg.weight_skewness * skew_term
                + cfg.weight_kurtosis * kurt_term
                + (cfg.weight_roi * roi_term if has_roi_signal else 0.0)
            ) / max(w_base_sum, 1e-6)
            base_ref_score = float(np.clip(base_ref_raw, 0.0, 1.0))
            active_score = base_ref_score if cfg.baseline_reference else float(np.clip(raw_score, 0.0, 1.0))

            score_obj = ObservationPrecursorScore(
                track_id=tid,
                scan_idx=scan_idx,
                label=lbl,
                anisotropy=round(aniso, 4),
                anisotropy_term=round(aniso_term, 4),
                spread_growth=round(base_spread_growth if cfg.baseline_reference else spread_growth, 4),
                spread_term=round(b_spread_term if cfg.baseline_reference else spread_term, 4),
                roi_subpeak_count=n_subpeaks,
                roi_dispersion_term=round(roi_term, 4),
                precursor_score=round(active_score, 4),
                delta_log_anisotropy=round(delta_log_aniso, 4),
                delta_skewness=round(delta_skew, 4),
                platykurtic_drop=round(platy_drop, 4),
                leptokurtic_wing=round(lepto_wing, 4),
                subpeak_score=round(subpeak_sc, 4),
                patch_dispersion=round(patch_disp, 4),
                baseline_referenced_score=round(base_ref_score, 4),
            )
            track_scores.append(score_obj)
            all_obs_scores.append(score_obj)

        # Evaluate contiguous persistence runs where precursor_score >= score_threshold.
        best_run: list[ObservationPrecursorScore] = []
        current_run: list[ObservationPrecursorScore] = []
        for idx_s, sc in enumerate(track_scores):
            is_consecutive = idx_s == 0 or (sc.scan_idx == track_scores[idx_s - 1].scan_idx + 1)
            if sc.precursor_score >= cfg.score_threshold:
                if current_run and is_consecutive:
                    current_run.append(sc)
                else:
                    current_run = [sc]
                if len(current_run) > len(best_run) or (
                    len(current_run) == len(best_run)
                    and max(x.precursor_score for x in current_run) > max(x.precursor_score for x in best_run)
                ):
                    best_run = list(current_run)
            else:
                current_run = []

        if len(best_run) >= max(1, cfg.min_persistence):
            peak_obs = max(best_run, key=lambda x: x.precursor_score)
            mean_active = float(np.mean([x.precursor_score for x in best_run]))
            pers_bonus = min(0.20, 0.05 * (len(best_run) - cfg.min_persistence + 1))
            conf = float(np.clip(0.55 * peak_obs.precursor_score + 0.35 * mean_active + pers_bonus, 0.05, 1.0))
            unranked.append(
                PrecursorCandidate(
                    rank=0,
                    track_id=tid,
                    onset_scan=best_run[0].scan_idx,
                    peak_scan=peak_obs.scan_idx,
                    persistence_scans=len(best_run),
                    peak_score=round(peak_obs.precursor_score, 4),
                    mean_active_score=round(mean_active, 4),
                    max_anisotropy=round(max(x.anisotropy for x in track_scores), 4),
                    max_spread_growth=round(max(x.spread_growth for x in track_scores), 4),
                    max_subpeak_count=max(x.roi_subpeak_count for x in track_scores),
                    confidence=round(conf, 4),
                    subsequent_split=tid in split_parent_tids,
                )
            )

    # Sort descending by confidence, breaking ties by peak_score then track_id.
    unranked.sort(key=lambda c: (-c.confidence, -c.peak_score, c.track_id))
    ranked = [
        PrecursorCandidate(
            rank=idx + 1,
            track_id=c.track_id,
            onset_scan=c.onset_scan,
            peak_scan=c.peak_scan,
            persistence_scans=c.persistence_scans,
            peak_score=c.peak_score,
            mean_active_score=c.mean_active_score,
            max_anisotropy=c.max_anisotropy,
            max_spread_growth=c.max_spread_growth,
            max_subpeak_count=c.max_subpeak_count,
            confidence=c.confidence,
            subsequent_split=c.subsequent_split,
        )
        for idx, c in enumerate(unranked)
    ]
    return ranked, all_obs_scores


_EVAL_CHANNELS: tuple[str, ...] = (
    "delta_log_anisotropy",
    "spread_growth",
    "delta_skewness",
    "platykurtic_drop",
    "leptokurtic_wing",
    "subpeak_score",
    "patch_dispersion",
    "precursor_score",
    "baseline_referenced_score",
)


def evaluate_precursor_channels_roc(
    obs_scores: Sequence[ObservationPrecursorScore],
    split_events: Sequence[LineageEvent],
    *,
    scan_stimuli: Mapping[int, ScanStimulusPoint] | Sequence[ScanStimulusPoint] | None = None,
    lookback_scans: int = 8,
    min_daughter_persistence: int = 1,
    exclude_reversible_overlap: bool = True,
    alarm_threshold: float = 0.35,
) -> dict[str, PrecursorChannelMetrics]:
    """Evaluate per-channel AUROC, AUPRC, false-alarm rate, and real scan/mV lead time.

    Positives (``y = 1``) are observations on parent tracks within ``lookback_scans``
    leading up to a persistent split event (``scan_idx_from - lookback_scans < s <= scan_idx_from``).
    Negatives (``y = 0``) are observations on tracks that never undergo a split event.
    Reversible 2 -> 1 -> 2 trajectory overlaps are excluded by default so precursor metrics
    measure irreversible cleavage rather than transient diffraction spot crossings.
    """
    stim_map: dict[int, ScanStimulusPoint] = {}
    if isinstance(scan_stimuli, Mapping):
        stim_map = {int(k): v for k, v in scan_stimuli.items()}
    elif scan_stimuli is not None:
        stim_map = {int(st.scan_idx): st for st in scan_stimuli}

    genuine_splits: list[LineageEvent] = []
    for ev in split_events:
        if ev.event_type != "split":
            continue
        if exclude_reversible_overlap and (
            getattr(ev, "split_subtype", None) == "reversible_overlap" or bool(getattr(ev, "preceded_by_merge", False))
        ):
            continue
        if int(getattr(ev, "daughter_persistence_scans", 1)) < int(min_daughter_persistence):
            continue
        genuine_splits.append(ev)

    split_windows_by_tid: dict[int, list[tuple[int, int]]] = {}
    for ev in genuine_splits:
        for ptid in ev.parent_track_ids:
            split_windows_by_tid.setdefault(int(ptid), []).append((int(ev.scan_idx_from), int(ev.scan_idx_to)))

    labeled_obs: list[tuple[ObservationPrecursorScore, int]] = []
    for sc in obs_scores:
        tid = int(sc.track_id)
        s = int(sc.scan_idx)
        if tid in split_windows_by_tid:
            in_pre_split = any(
                (t_from - max(1, int(lookback_scans))) < s <= t_from for t_from, _ in split_windows_by_tid[tid]
            )
            if in_pre_split:
                labeled_obs.append((sc, 1))
        else:
            labeled_obs.append((sc, 0))

    y_arr = np.asarray([lbl for _, lbl in labeled_obs], dtype=np.int32)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    n_events = len(genuine_splits)

    obs_by_tid: dict[int, list[ObservationPrecursorScore]] = {}
    for sc in obs_scores:
        obs_by_tid.setdefault(int(sc.track_id), []).append(sc)
    for lst in obs_by_tid.values():
        lst.sort(key=lambda item: int(item.scan_idx))

    results: dict[str, PrecursorChannelMetrics] = {}
    bounded_channels = {"subpeak_score", "patch_dispersion", "precursor_score", "baseline_referenced_score"}

    for ch in _EVAL_CHANNELS:
        vals = np.asarray([float(getattr(sc, ch, 0.0)) for sc, _ in labeled_obs], dtype=np.float64)
        neg_vals = vals[y_arr == 0]
        pos_vals = vals[y_arr == 1]

        if n_pos > 0 and n_neg > 0 and float(np.ptp(vals)) > 1e-12:
            auroc_val: float | None = float(roc_auc_score(y_arr, vals))
            auprc_val: float | None = float(average_precision_score(y_arr, vals))
        elif n_pos > 0 and n_neg > 0:
            auroc_val = 0.5
            auprc_val = float(n_pos) / float(n_pos + n_neg)
        else:
            auroc_val = None
            auprc_val = None

        if ch in bounded_channels:
            thr_eff = float(alarm_threshold)
        elif len(neg_vals) > 0:
            med_neg = float(np.median(neg_vals))
            mad_neg = 1.4826 * float(np.median(np.abs(neg_vals - med_neg)))
            thr_eff = max(float(alarm_threshold), med_neg + 2.5 * mad_neg)
        else:
            thr_eff = float(alarm_threshold)

        fa_per_100 = float(100.0 * np.mean(neg_vals >= thr_eff)) if len(neg_vals) > 0 else 0.0
        rec_at_thr = float(np.mean(pos_vals >= thr_eff)) if len(pos_vals) > 0 else None

        lead_scans_list: list[float] = []
        lead_mv_list: list[float] = []
        for ev in genuine_splits:
            best_onset_s: int | None = None
            t_from = int(ev.scan_idx_from)
            t_to = int(ev.scan_idx_to)
            for ptid in ev.parent_track_ids:
                for sc in obs_by_tid.get(int(ptid), []):
                    if (
                        (t_from - max(1, int(lookback_scans))) < sc.scan_idx <= t_from
                        and float(getattr(sc, ch, 0.0)) >= thr_eff
                        and (best_onset_s is None or sc.scan_idx < best_onset_s)
                    ):
                        best_onset_s = int(sc.scan_idx)
            if best_onset_s is not None:
                lead_scans_list.append(float(t_to - best_onset_s))
                st_onset = stim_map.get(best_onset_s)
                st_split = stim_map.get(t_to)
                if (
                    st_onset is not None
                    and st_split is not None
                    and st_onset.voltage_v is not None
                    and st_split.voltage_v is not None
                    and np.isfinite(st_onset.voltage_v)
                    and np.isfinite(st_split.voltage_v)
                ):
                    lead_mv_list.append(1000.0 * abs(float(st_split.voltage_v) - float(st_onset.voltage_v)))

        results[ch] = PrecursorChannelMetrics(
            channel=ch,
            auroc=auroc_val,
            auprc=auprc_val,
            n_positive_obs=n_pos,
            n_negative_obs=n_neg,
            n_fracture_events=n_events,
            false_alarms_per_100_obs=fa_per_100,
            recall_at_alarm_threshold=rec_at_thr,
            mean_lead_scans=float(np.mean(lead_scans_list)) if lead_scans_list else None,
            mean_lead_mv=float(np.mean(lead_mv_list)) if lead_mv_list else None,
        )

    return results


def precursors_to_table(candidates: list[PrecursorCandidate]) -> list[dict[str, Any]]:
    """Serialize ranked :class:`PrecursorCandidate` items to row dicts."""
    return [
        {
            "rank": c.rank,
            "track_id": c.track_id,
            "onset_scan": c.onset_scan,
            "peak_scan": c.peak_scan,
            "persistence_scans": c.persistence_scans,
            "peak_score": c.peak_score,
            "mean_active_score": c.mean_active_score,
            "max_anisotropy": c.max_anisotropy,
            "max_spread_growth": c.max_spread_growth,
            "max_subpeak_count": c.max_subpeak_count,
            "confidence": c.confidence,
            "subsequent_split": c.subsequent_split,
        }
        for c in candidates
    ]
