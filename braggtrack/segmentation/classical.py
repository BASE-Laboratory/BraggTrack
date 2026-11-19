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


def _normalize_sigma_3d(sigma: float | tuple[float, float, float] | list[float]) -> float | tuple[float, float, float]:
    if isinstance(sigma, (tuple, list, np.ndarray)):
        if len(sigma) != 3:
            raise ValueError(f"3D sigma tuple must have length 3 (sigma_z, sigma_y, sigma_x), got {sigma!r}")
        return (max(0.0, float(sigma[0])), max(0.0, float(sigma[1])), max(0.0, float(sigma[2])))
    return max(0.0, float(sigma))


def resolve_anisotropic_sigma(shape: tuple[int, int, int]) -> tuple[float, float, float]:
    """Select default ``(sigma_z, sigma_y, sigma_x)`` accounting for thin rocking-frame stacks.

    Near-field ESRF-ID03 ``pco_nf`` cubes typically have ``nz <= 120`` coarse rocking steps
    against ``ny, nx >= 200`` fine sCMOS detector pixels, yielding pancake spots spanning
    only ``2-4`` frames in ``z`` vs ``15-60`` pixels in ``(y, x)``. Using ``sigma_z = 0.6``
    prevents blurring adjacent rocking layers together while ``sigma_y = sigma_x = 1.35``
    suppresses intra-pancake pixel ripple over-segmentation.
    """
    nz, ny, nx = int(shape[0]), int(shape[1]), int(shape[2])
    det_min = min(ny, nx)
    if det_min >= 32 and nz <= int(0.8 * det_min):
        return (0.6, 1.35, 1.35)
    return (1.0, 1.0, 1.0)


def gaussian_blur_3d(
    volume: np.ndarray,
    passes: int = 1,
    sigma: float | tuple[float, float, float] = 1.0,
) -> np.ndarray:
    """Apply Gaussian blur along each axis (isotropic scalar or ``(sigma_z, sigma_y, sigma_x)``)."""
    work_dt = _float_work_dtype(volume)
    result = np.asarray(volume, dtype=work_dt)
    sig = _normalize_sigma_3d(sigma)
    for _ in range(max(1, passes)):
        result = gaussian_filter(result, sigma=sig)
    return result


def laplacian_3d(
    volume: np.ndarray,
    *,
    weights: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """6-neighbor discrete Laplacian (optionally scale-weighted per axis)."""
    work_dt = _float_work_dtype(volume)
    arr = np.asarray(volume, dtype=work_dt)
    if weights is None:
        return laplace(arr).astype(work_dt, copy=False)

    wz, wy, wx = float(weights[0]), float(weights[1]), float(weights[2])
    out = np.zeros_like(arr, dtype=work_dt)
    for axis, w_ax in ((0, wz), (1, wy), (2, wx)):
        if w_ax == 0.0 or arr.shape[axis] < 2:
            continue
        padded = np.pad(arr, [(1, 1) if i == axis else (0, 0) for i in range(3)], mode="edge")
        sl_lo = [slice(None)] * 3
        sl_mid = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -2)
        sl_mid[axis] = slice(1, -1)
        sl_hi[axis] = slice(2, None)
        d2 = padded[tuple(sl_lo)] - 2.0 * padded[tuple(sl_mid)] + padded[tuple(sl_hi)]
        out = out + np.asarray(w_ax, dtype=work_dt) * d2.astype(work_dt, copy=False)
    return out


def log_enhance_3d(
    volume: np.ndarray,
    blur_passes: int = 1,
    sigma: float | tuple[float, float, float] = 1.0,
) -> np.ndarray:
    """LoG-like enhancement: blur then (scale-normalized when anisotropic) negative Laplacian."""
    sig = _normalize_sigma_3d(sigma)
    smoothed = gaussian_blur_3d(volume, passes=blur_passes, sigma=sig)
    if isinstance(sig, tuple):
        s2 = (max(sig[0], 0.25) ** 2, max(sig[1], 0.25) ** 2, max(sig[2], 0.25) ** 2)
        mean_s2 = max((s2[0] + s2[1] + s2[2]) / 3.0, 1e-6)
        weights = (s2[0] / mean_s2, s2[1] / mean_s2, s2[2] / mean_s2)
        lap = laplacian_3d(smoothed, weights=weights)
    else:
        lap = laplacian_3d(smoothed)
    return -lap


def _anisotropic_peak_coords(
    vol: np.ndarray,
    min_value: float,
    sep: tuple[int, int, int],
) -> np.ndarray:
    from scipy.ndimage import maximum_filter

    sz, sy, sx = max(1, int(sep[0])), max(1, int(sep[1])), max(1, int(sep[2]))
    footprint_size = (2 * sz + 1, 2 * sy + 1, 2 * sx + 1)
    local_max = maximum_filter(vol, size=footprint_size, mode="nearest")
    mask = (vol >= float(min_value)) & (vol == local_max)
    # Suppress border-adjacent plateaus identically to peak_local_max default behavior
    raw_coords = np.argwhere(mask)
    if len(raw_coords) <= 1:
        return raw_coords
    # Greedy descending intensity NMS within ellipsoid/box (sz, sy, sx) to break exact plateau ties
    vals = vol[raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2]]
    order = np.argsort(-vals, kind="mergesort")
    kept: list[np.ndarray] = []
    for idx in order:
        c = raw_coords[idx]
        if any(
            abs(int(c[0] - k[0])) <= sz and abs(int(c[1] - k[1])) <= sy and abs(int(c[2] - k[2])) <= sx for k in kept
        ):
            continue
        kept.append(c)
    return np.asarray(kept, dtype=np.int64) if kept else np.empty((0, 3), dtype=np.int64)


def h_maxima_seeds(
    volume: np.ndarray,
    min_value: float,
    h: float,
    min_separation: int | tuple[int, int, int] = 1,
) -> list[tuple[int, int, int]]:
    """Find h-maxima seeds above threshold with non-maximum suppression.

    Each seed must be a local maximum whose value exceeds the highest
    neighbour by at least *h*, and must be >= *min_value*.
    """
    vol = np.asarray(volume)
    if not np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32)
    if isinstance(min_separation, (tuple, list)):
        coords = _anisotropic_peak_coords(
            vol, float(min_value), (int(min_separation[0]), int(min_separation[1]), int(min_separation[2]))
        )
    else:
        coords = peak_local_max(
            vol,
            min_distance=max(1, int(min_separation)),
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
    min_separation: int | tuple[int, int, int] = 1,
) -> list[tuple[int, int, int]]:
    """Find local maxima seeds above a minimum value."""
    vol = np.asarray(volume)
    if not np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32)
    if isinstance(min_separation, (tuple, list)):
        coords = _anisotropic_peak_coords(
            vol, float(min_value), (int(min_separation[0]), int(min_separation[1]), int(min_separation[2]))
        )
    else:
        coords = peak_local_max(
            vol,
            min_distance=max(1, int(min_separation)),
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
    sigma: float | tuple[float, float, float] = 1.0,
    h_value: float = 0.1,
    min_seed_separation: int | tuple[int, int, int] = 2,
    seed_peak_fraction: float = 0.2,
    seed_response_percentile: float = 99.95,
    *,
    working_dtype: type | np.dtype = np.float32,
    log_scale: bool = False,
    anisotropic_auto: bool = False,
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
    anisotropic_auto
        When ``True`` and ``sigma == 1.0`` (scalar default), select per-axis
        ``(sigma_z, sigma_y, sigma_x)`` via :func:`resolve_anisotropic_sigma` and
        anisotropic ``min_seed_separation = (1, 3, 3)`` for thin rocking-frame cubes.
    """
    vol_arr = np.asarray(volume, dtype=working_dtype)
    foreground = vol_arr >= np.asarray(threshold, dtype=working_dtype)
    enh_input = np.log1p(np.maximum(vol_arr, 0.0)) if log_scale else vol_arr
    eff_sigma: float | tuple[float, float, float] = sigma
    eff_sep: int | tuple[int, int, int] = min_seed_separation
    if anisotropic_auto and not isinstance(sigma, (tuple, list, np.ndarray)) and float(sigma) == 1.0:
        eff_sigma = resolve_anisotropic_sigma(vol_arr.shape)
        if isinstance(min_seed_separation, int) and min_seed_separation == 2 and eff_sigma[0] < eff_sigma[1]:
            eff_sep = (1, 3, 3)
    response = log_enhance_3d(enh_input, blur_passes=blur_passes, sigma=eff_sigma)
    seed_floor = _seed_floor_from_response(
        response[foreground],
        seed_peak_fraction=seed_peak_fraction,
        seed_response_percentile=seed_response_percentile,
    )
    seeds = h_maxima_seeds(
        response,
        min_value=seed_floor,
        h=h_value,
        min_separation=eff_sep,
    )
    labels = watershed_from_seeds(response, seeds=seeds, mask=foreground)
    return ClassicalSegmentationResult(
        threshold=float(threshold),
        seed_count=len(seeds),
        component_count=_count_labels(labels),
        labeled_volume=labels,
        response=response,
    )
