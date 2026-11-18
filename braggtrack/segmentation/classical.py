"""Classical 3D segmentation building blocks (LoG + watershed)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass(frozen=True)
class ClassicalSegmentationResult:
    threshold: float
    seed_count: int
    component_count: int
    labeled_volume: np.ndarray
    response: np.ndarray


def _float_work_dtype(arr: np.ndarray) -> np.dtype:
    """Use float32 for <=32-bit detector arrays (50% RAM cut) and float64 when explicitly float64."""
    dt = np.asarray(arr).dtype
    if dt == np.float64:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def gaussian_blur_3d(volume: np.ndarray, passes: int = 1, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur along each axis."""
    work_dt = _float_work_dtype(volume)
    result = np.asarray(volume, dtype=work_dt)
    for _ in range(max(1, passes)):
        result = gaussian_filter(result, sigma=sigma)
    return result


def laplacian_3d(volume: np.ndarray) -> np.ndarray:
    """6-neighbor discrete Laplacian."""
    work_dt = _float_work_dtype(volume)
    return laplace(np.asarray(volume, dtype=work_dt)).astype(work_dt, copy=False)


def log_enhance_3d(
    volume: np.ndarray,
    blur_passes: int = 1,
    sigma: float = 1.0,
) -> np.ndarray:
    """LoG-like enhancement: blur then negative Laplacian."""
    smoothed = gaussian_blur_3d(volume, passes=blur_passes, sigma=sigma)
    lap = laplacian_3d(smoothed)
    return -lap


def h_maxima_seeds(
    volume: np.ndarray,
    min_value: float,
    h: float,
    min_separation: int = 1,
) -> list[tuple[int, int, int]]:
    """Find h-maxima seeds above threshold with non-maximum suppression.

    Each seed must be a local maximum whose value exceeds the highest
    neighbour by at least *h*, and must be >= *min_value*.
    """
    vol = np.asarray(volume)
    if not np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32)
    coords = peak_local_max(
        vol,
        min_distance=max(1, min_separation),
        threshold_abs=float(min_value),
    )
    if len(coords) == 0:
        return []

    seeds: list[tuple[int, int, int]] = []
    for coord in coords:
        z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
        center_val = float(vol[z, y, x])
        z_lo, z_hi = max(0, z - 1), min(vol.shape[0], z + 2)
        y_lo, y_hi = max(0, y - 1), min(vol.shape[1], y + 2)
        x_lo, x_hi = max(0, x - 1), min(vol.shape[2], x + 2)
        neighborhood = vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].astype(np.float64, copy=True)
        neighborhood[z - z_lo, y - y_lo, x - x_lo] = -np.inf
        max_neighbor = float(neighborhood.max())
        if center_val - max_neighbor >= h:
            seeds.append((z, y, x))
            continue
        # Handle symmetric sub-voxel peaks (e.g. center at y=15.5) where 2..8 apex voxels tie:
        # verify prominence >= h against the 5x5x5 shell outside tied immediate apex neighbors.
        tie_tol = 1e-4 * max(1.0, abs(center_val))
        if h > 0.0 and abs(center_val - max_neighbor) <= tie_tol:
            z2_lo, z2_hi = max(0, z - 2), min(vol.shape[0], z + 3)
            y2_lo, y2_hi = max(0, y - 2), min(vol.shape[1], y + 3)
            x2_lo, x2_hi = max(0, x - 2), min(vol.shape[2], x + 3)
            box5 = vol[z2_lo:z2_hi, y2_lo:y2_hi, x2_lo:x2_hi].astype(np.float64, copy=True)
            cz, cy, cx = z - z2_lo, y - y2_lo, x - x2_lo
            core = box5[max(0, cz - 1) : cz + 2, max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2]
            core[np.abs(core - center_val) <= tie_tol] = -np.inf
            rem_max = float(box5.max())
            if math.isfinite(rem_max) and (center_val - rem_max) >= h:
                seeds.append((z, y, x))

    return seeds


def local_maxima_seeds(
    volume: np.ndarray,
    min_value: float,
    min_separation: int = 1,
) -> list[tuple[int, int, int]]:
    """Find local maxima seeds above a minimum value."""
    vol = np.asarray(volume)
    if not np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32)
    coords = peak_local_max(
        vol,
        min_distance=max(1, min_separation),
        threshold_abs=float(min_value),
    )
    return [(int(c[0]), int(c[1]), int(c[2])) for c in coords]


def watershed_from_seeds(
    response: np.ndarray,
    seeds: list[tuple[int, int, int]],
    mask: np.ndarray,
) -> np.ndarray:
    """Seeded watershed over ``-response`` restricted to a boolean foreground ``mask``.

    ``mask`` lives in the *intensity* domain — typically ``volume >= otsu``.
    Passing a mask derived from the response itself under-shoots catastrophically
    because the response and volume distributions are on different scales.
    """
    work_dt = _float_work_dtype(response)
    response_arr = np.asarray(response, dtype=work_dt)
    markers = np.zeros(response_arr.shape, dtype=np.int32)
    for label_id, (z, y, x) in enumerate(seeds, start=1):
        markers[z, y, x] = label_id

    labeled = watershed(-response_arr, markers=markers, mask=np.asarray(mask, dtype=bool))
    return labeled.astype(np.int32)


def _count_labels(labels: np.ndarray) -> int:
    return len(np.unique(labels[labels > 0]))


_PERCENTILE_MIN_FOREGROUND = 100

_ROBUST_PEAK_PERCENTILE = 99.99


def _seed_floor_from_response(
    response_foreground: np.ndarray,
    *,
    seed_peak_fraction: float,
    seed_response_percentile: float,
) -> float:
    """Seed admissibility floor derived from the LoG response inside the foreground.

    The fraction term uses a robust peak reference (99.99th percentile on
    large foregrounds, absolute max on small ones) so a single outlier voxel
    doesn't dominate. The percentile term suppresses noise seeds on large
    foregrounds; it is skipped when the foreground is too small for a
    meaningful percentile.
    """
    if response_foreground.size == 0:
        return float("inf")
    max_val = float(response_foreground.max())
    min_val = float(response_foreground.min())
    tail_frac = max(1.0 - float(seed_response_percentile) / 100.0, 1e-4)
    min_fg_for_pct = (
        max(_PERCENTILE_MIN_FOREGROUND, int(math.ceil(16.0 / tail_frac)))
        if seed_response_percentile > 0.0
        else _PERCENTILE_MIN_FOREGROUND
    )
    if response_foreground.size >= min_fg_for_pct:
        peak_ref = float(np.percentile(response_foreground, _ROBUST_PEAK_PERCENTILE))
        floor_f = seed_peak_fraction * peak_ref
        floor_p = float(np.percentile(response_foreground, seed_response_percentile))
        seed_floor = max(floor_p, floor_f)
    else:
        floor_f = seed_peak_fraction * max_val
        seed_floor = floor_f
    eps = max(1e-9, 1e-6 * max(1.0, max_val - min_val))
    return min(seed_floor, max_val - eps)


def segment_classical(
    volume: np.ndarray,
    threshold: float,
    blur_passes: int = 1,
    sigma: float = 1.0,
    h_value: float = 0.1,
    min_seed_separation: int = 2,
    seed_peak_fraction: float = 0.2,
    seed_response_percentile: float = 99.95,
    *,
    working_dtype: type | np.dtype = np.float32,
    log_scale: bool = False,
) -> ClassicalSegmentationResult:
    """Run classical LoG + h-maxima + seeded watershed pipeline.

    Parameters
    ----------
    volume
        Raw 3-D intensity cube.
    threshold
        **Intensity-domain** foreground threshold (e.g. from Otsu on ``volume``).
        Voxels ``volume >= threshold`` define the watershed mask.
    seed_peak_fraction, seed_response_percentile
        Admissibility floor on the LoG response, computed *within the
        foreground mask*. The fraction term uses a robust peak reference
        (99.99th percentile) rather than the absolute max, so isolated
        bright voxels don't inflate the floor.
    working_dtype
        Floating-point dtype used for 3D blur, Laplacian, and watershed buffers
        (default ``np.float32``, reducing peak 3D buffer RAM by 50% vs ``float64``).
    log_scale
        When ``True``, compute LoG enhancement and seed peaks on ``log1p(max(volume, 0))``
        so weak satellite reflections are not suppressed by intense primary Bragg peaks.
    """
    vol_arr = np.asarray(volume, dtype=working_dtype)
    foreground = vol_arr >= np.asarray(threshold, dtype=working_dtype)
    enh_input = np.log1p(np.maximum(vol_arr, 0.0)) if log_scale else vol_arr
    response = log_enhance_3d(enh_input, blur_passes=blur_passes, sigma=sigma)
    seed_floor = _seed_floor_from_response(
        response[foreground],
        seed_peak_fraction=seed_peak_fraction,
        seed_response_percentile=seed_response_percentile,
    )
    seeds = h_maxima_seeds(
        response,
        min_value=seed_floor,
        h=h_value,
        min_separation=min_seed_separation,
    )
    labels = watershed_from_seeds(response, seeds=seeds, mask=foreground)
    return ClassicalSegmentationResult(
        threshold=float(threshold),
        seed_count=len(seeds),
        component_count=_count_labels(labels),
        labeled_volume=labels,
        response=response,
    )
