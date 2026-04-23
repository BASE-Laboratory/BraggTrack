"""Classical 3D segmentation building blocks for Week 2."""

from __future__ import annotations

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


def gaussian_blur_3d(volume: np.ndarray, passes: int = 1, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur along each axis."""
    result = np.asarray(volume, dtype=np.float64)
    for _ in range(max(1, passes)):
        result = gaussian_filter(result, sigma=sigma)
    return result


def laplacian_3d(volume: np.ndarray) -> np.ndarray:
    """6-neighbor discrete Laplacian."""
    return laplace(np.asarray(volume, dtype=np.float64)).astype(np.float64)


def log_enhance_3d(
    volume: np.ndarray, blur_passes: int = 1, sigma: float = 1.0,
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
    vol = np.asarray(volume, dtype=np.float64)
    coords = peak_local_max(
        vol,
        min_distance=max(1, min_separation),
        threshold_abs=min_value,
    )
    if len(coords) == 0:
        return []

    seeds: list[tuple[int, int, int]] = []
    for coord in coords:
        z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
        center_val = vol[z, y, x]
        z_lo, z_hi = max(0, z - 1), min(vol.shape[0], z + 2)
        y_lo, y_hi = max(0, y - 1), min(vol.shape[1], y + 2)
        x_lo, x_hi = max(0, x - 1), min(vol.shape[2], x + 2)
        neighborhood = vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].copy()
        neighborhood[z - z_lo, y - y_lo, x - x_lo] = -np.inf
        max_neighbor = neighborhood.max()
        if center_val - max_neighbor >= h:
            seeds.append((z, y, x))

    return seeds


def local_maxima_seeds(
    volume: np.ndarray,
    min_value: float,
    min_separation: int = 1,
) -> list[tuple[int, int, int]]:
    """Find local maxima seeds above a minimum value."""
    vol = np.asarray(volume, dtype=np.float64)
    coords = peak_local_max(
        vol,
        min_distance=max(1, min_separation),
        threshold_abs=min_value,
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
    response = np.asarray(response, dtype=np.float64)
    markers = np.zeros(response.shape, dtype=np.int32)
    for label_id, (z, y, x) in enumerate(seeds, start=1):
        markers[z, y, x] = label_id

    labeled = watershed(-response, markers=markers, mask=np.asarray(mask, dtype=bool))
    return labeled.astype(np.int32)


def _count_labels(labels: np.ndarray) -> int:
    return len(np.unique(labels[labels > 0]))


_PERCENTILE_MIN_FOREGROUND = 100


def _seed_floor_from_response(
    response_foreground: np.ndarray,
    *,
    seed_peak_fraction: float,
    seed_response_percentile: float,
) -> float:
    """Seed admissibility floor derived from the LoG response inside the foreground.

    The fraction-of-max term keeps toy volumes with few hot voxels from
    rejecting all but the single brightest peak. The percentile term
    suppresses noise seeds on large foregrounds where many voxels sit
    above the fraction floor; it is skipped on tiny foregrounds because
    a 99.5th percentile over <100 voxels is just the max.
    """
    if response_foreground.size == 0:
        return float("inf")
    max_val = float(response_foreground.max())
    min_val = float(response_foreground.min())
    floor_f = seed_peak_fraction * max_val
    if response_foreground.size >= _PERCENTILE_MIN_FOREGROUND:
        floor_p = float(np.percentile(response_foreground, seed_response_percentile))
        seed_floor = max(floor_p, floor_f)
    else:
        seed_floor = floor_f
    eps = max(1e-9, 1e-6 * max(1.0, max_val - min_val))
    return min(seed_floor, max_val - eps)


def segment_classical(
    volume: np.ndarray,
    threshold: float,
    blur_passes: int = 1,
    sigma: float = 1.0,
    h_value: float = 0.1,
    min_seed_separation: int = 1,
    seed_peak_fraction: float = 0.2,
    seed_response_percentile: float = 99.5,
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
        foreground mask*. A seed must clear both
        ``seed_peak_fraction * max(response[mask])`` and
        ``percentile(response[mask], seed_response_percentile)``.
    """
    volume = np.asarray(volume, dtype=np.float64)
    response = log_enhance_3d(volume, blur_passes=blur_passes, sigma=sigma)
    foreground = volume >= threshold
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
        threshold=threshold,
        seed_count=len(seeds),
        component_count=_count_labels(labels),
        labeled_volume=labels,
        response=response,
    )
