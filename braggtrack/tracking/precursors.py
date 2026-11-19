"""Chemomechanical fracture precursor scoring with multi-scan persistence filtering.

Detects intragranular mosaicity buildup and incipient sub-grain splitting before
a Bragg reflection cleaves into separate connected components, combining:
1. 3D covariance elongation (eigenvalue anisotropy eig_1 / eig_3),
2. Relative reciprocal-space broadening (covariance trace growth), and
3. Intra-spot ROI sub-peak bimodality + DINO patch-feature dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.feature import peak_local_max

from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips

if TYPE_CHECKING:
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
    roi_cache: dict[tuple[int, int], tuple[int, float]] | None = None,
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
        Optional mutable ``(scan_idx, label) -> (n_subpeaks, roi_term)`` cache
        for incremental online streaming so historical scans are not re-cropped.
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
    effective_cache: dict[tuple[int, int], tuple[int, float]] = roi_cache if roi_cache is not None else {}

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
            # Mark placeholder immediately in case duplicate rows exist
            effective_cache[key] = (1, 0.0)
            masked_cube, mask = crop_spot_cube(
                vol_map[s_idx],
                lab_map[s_idx],
                lbl_i,
                row,
                margin=cfg.roi_margin,
            )
            n_sub, sub_term = _count_internal_subpeaks(masked_cube, mask)
            if patch_encoder is None:
                effective_cache[key] = (n_sub, sub_term)
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
                effective_cache[key] = (n_sub, roi_val)

    split_parent_tids: set[int] = set()
    if lineage_events:
        for ev in lineage_events:
            if ev.event_type == "split":
                split_parent_tids.update(ev.parent_track_ids)

    by_track: dict[int, list[dict[str, Any]]] = {}
    for row in track_table:
        by_track.setdefault(int(row["track_id"]), []).append(row)

    all_obs_scores: list[ObservationPrecursorScore] = []
    unranked: list[PrecursorCandidate] = []

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
        c0_mu = max(q_floor_mu, float(obs_list[0].get("cov_zz", 1.0)))
        c0_chi = max(q_floor_chi, float(obs_list[0].get("cov_xx", 1.0)))
        c0_d = max(q_floor_d, float(obs_list[0].get("cov_yy", 1.0)))
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
        track_scores: list[ObservationPrecursorScore] = []

        for obs in obs_list:
            scan_idx = int(obs["scan_idx"])
            lbl = int(obs.get("label", 0))
            eig1 = max(0.0, float(obs.get("eig_1", 0.0)))
            eig2 = max(0.0, float(obs.get("eig_2", 0.0)))
            eig3 = max(0.0, float(obs.get("eig_3", 0.0)))

            if mixed_units:
                r_mu = max(q_floor_mu, float(obs.get("cov_zz", 0.0))) / c0_mu
                r_chi = max(q_floor_chi, float(obs.get("cov_xx", 0.0))) / c0_chi
                r_d = max(q_floor_d, float(obs.get("cov_yy", 0.0))) / c0_d
                r_max = max(r_mu, r_chi, r_d)
                r_min = max(min(r_mu, r_chi, r_d), 1e-9)
                aniso = r_max / r_min
                spread_growth = (r_mu + r_chi + r_d - 3.0) / 3.0
            else:
                aniso = eig1 / eig3 if eig3 > 1e-9 else (1.0 if eig1 <= 1e-9 else 10.0)
                trace = eig1 + eig2 + eig3
                spread_growth = (trace - e0) / e0 if e0 > 1e-9 else 0.0

            aniso_term = float(np.clip((aniso - 1.0) / max(cfg.anisotropy_scale, 1e-6), 0.0, 1.0))
            spread_term = float(np.clip(max(0.0, spread_growth) / max(cfg.spread_growth_scale, 1e-6), 0.0, 1.0))

            n_subpeaks = 1
            roi_term = 0.0
            use_roi_here = lbl > 0 and (
                (scan_idx, lbl) in effective_cache or (has_roi and scan_idx in vol_map and scan_idx in lab_map)
            )
            if use_roi_here and "bbox_min_z" in obs:
                n_subpeaks, roi_term = effective_cache.get((scan_idx, lbl), (1, 0.0))
                w_sum = cfg.weight_anisotropy + cfg.weight_spread + cfg.weight_roi
                raw_score = (
                    cfg.weight_anisotropy * aniso_term + cfg.weight_spread * spread_term + cfg.weight_roi * roi_term
                ) / max(w_sum, 1e-6)
            else:
                w_geo = cfg.weight_anisotropy + cfg.weight_spread
                raw_score = (cfg.weight_anisotropy * aniso_term + cfg.weight_spread * spread_term) / max(w_geo, 1e-6)

            score_obj = ObservationPrecursorScore(
                track_id=tid,
                scan_idx=scan_idx,
                label=lbl,
                anisotropy=round(aniso, 4),
                anisotropy_term=round(aniso_term, 4),
                spread_growth=round(spread_growth, 4),
                spread_term=round(spread_term, 4),
                roi_subpeak_count=n_subpeaks,
                roi_dispersion_term=round(roi_term, 4),
                precursor_score=round(float(np.clip(raw_score, 0.0, 1.0)), 4),
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
