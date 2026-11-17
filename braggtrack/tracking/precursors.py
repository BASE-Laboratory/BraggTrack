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
    if vmax <= 0.0:
        return 1, 0.0

    smoothed = gaussian_filter(np.where(fg, masked_cube, 0.0).astype(np.float64), sigma=0.7)
    smax = float(smoothed[fg].max())
    if smax <= 0.0:
        return 1, 0.0

    coords = peak_local_max(
        smoothed,
        min_distance=2,
        threshold_abs=0.25 * smax,
        labels=fg.astype(np.uint8),
        num_peaks=32,
    )
    n_peaks = int(len(coords))
    if n_peaks <= 1:
        return 1, 0.0

    subpeak_score = float(np.clip(0.65 + 0.20 * (n_peaks - 2), 0.0, 1.0))
    return n_peaks, subpeak_score


def _patch_dispersion_score(
    masked_cube: np.ndarray,
    mask: np.ndarray,
    patch_encoder: PatchFeatureEncoder,
    upsample_size: int,
) -> float:
    """Compute mean foreground patch-token cosine dispersion across orthogonal MIPs."""
    mip_mu, mip_chi, mip_d = orthogonal_mips(masked_cube)
    mask_mu, mask_chi, mask_d = orthogonal_mips(mask.astype(np.float64))

    dispersions: list[float] = []
    for mip, mm in ((mip_mu, mask_mu), (mip_chi, mask_chi), (mip_d, mask_d)):
        if mip.size == 0 or float(mm.max()) <= 0.0:
            continue
        h, w = mip.shape
        zh = max(1.0, upsample_size / max(1, h))
        zw = max(1.0, upsample_size / max(1, w))
        mip_up = zoom(mip.astype(np.float64), (zh, zw), order=1) if (zh > 1.0 or zw > 1.0) else mip.astype(np.float64)
        mm_up = zoom(mm.astype(np.float64), (zh, zw), order=0) if (zh > 1.0 or zw > 1.0) else mm.astype(np.float64)

        fmap = patch_encoder.extract_patch_features(mip_up)
        hp, wp, dim = fmap.shape
        if hp * wp < 2:
            continue

        # Map foreground mask onto (hp, wp) patch grid.
        bh = max(1, mm_up.shape[0] // hp)
        bw = max(1, mm_up.shape[1] // wp)
        cropped_fg = (mm_up[: hp * bh, : wp * bw] > 0).reshape(hp, bh, wp, bw).max(axis=(1, 3))
        fg_vecs = fmap[cropped_fg] if int(cropped_fg.sum()) >= 2 else fmap.reshape(-1, dim)
        if fg_vecs.shape[0] < 2:
            continue

        mean_vec = fg_vecs.mean(axis=0)
        nrm = float(np.linalg.norm(mean_vec))
        if nrm <= 0.0:
            continue
        mean_vec = mean_vec / nrm
        cos_sims = np.clip(fg_vecs @ mean_vec, -1.0, 1.0)
        cos_disp = float(1.0 - np.mean(cos_sims))
        dispersions.append(float(np.clip(cos_disp / 0.6, 0.0, 1.0)))

    return float(np.mean(dispersions)) if dispersions else 0.0


def compute_fracture_precursors(
    track_table: list[dict[str, Any]],
    config: PrecursorConfig | None = None,
    *,
    volumes: dict[int, np.ndarray] | list[np.ndarray] | None = None,
    label_volumes: dict[int, np.ndarray] | list[np.ndarray] | None = None,
    patch_encoder: PatchFeatureEncoder | None = None,
    lineage_events: list[LineageEvent] | None = None,
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
        e0 = (
            float(obs_list[0].get("eig_1", 0.0))
            + float(obs_list[0].get("eig_2", 0.0))
            + float(obs_list[0].get("eig_3", 0.0))
        )
        track_scores: list[ObservationPrecursorScore] = []

        for obs in obs_list:
            scan_idx = int(obs["scan_idx"])
            lbl = int(obs.get("label", 0))
            eig1 = max(0.0, float(obs.get("eig_1", 0.0)))
            eig2 = max(0.0, float(obs.get("eig_2", 0.0)))
            eig3 = max(0.0, float(obs.get("eig_3", 0.0)))

            aniso = eig1 / eig3 if eig3 > 1e-9 else (1.0 if eig1 <= 1e-9 else 10.0)
            aniso_term = float(np.clip((aniso - 1.0) / max(cfg.anisotropy_scale, 1e-6), 0.0, 1.0))

            trace = eig1 + eig2 + eig3
            spread_growth = (trace - e0) / e0 if e0 > 1e-9 else 0.0
            spread_term = float(np.clip(max(0.0, spread_growth) / max(cfg.spread_growth_scale, 1e-6), 0.0, 1.0))

            n_subpeaks = 1
            roi_term = 0.0
            use_roi_here = has_roi and scan_idx in vol_map and scan_idx in lab_map and lbl > 0
            if use_roi_here and "bbox_min_z" in obs:
                masked_cube, mask = crop_spot_cube(
                    vol_map[scan_idx],
                    lab_map[scan_idx],
                    lbl,
                    obs,
                    margin=cfg.roi_margin,
                )
                n_subpeaks, subpeak_term = _count_internal_subpeaks(masked_cube, mask)
                if patch_encoder is not None:
                    p_disp = _patch_dispersion_score(masked_cube, mask, patch_encoder, cfg.roi_upsample_size)
                    roi_term = float(np.clip(0.65 * subpeak_term + 0.35 * p_disp, 0.0, 1.0))
                else:
                    roi_term = subpeak_term

            if use_roi_here and "bbox_min_z" in obs:
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
