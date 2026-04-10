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
    threshold: float,
) -> np.ndarray:
    """Seeded watershed over voxels above threshold."""
    response = np.asarray(response, dtype=np.float64)
    markers = np.zeros(response.shape, dtype=np.int32)
    for label_id, (z, y, x) in enumerate(seeds, start=1):
        markers[z, y, x] = label_id

    mask = response >= threshold
    labeled = watershed(-response, markers=markers, mask=mask)
    return labeled.astype(np.int32)


def _count_labels(labels: np.ndarray) -> int:
    return len(np.unique(labels[labels > 0]))


def segment_classical(
    volume: np.ndarray,
    threshold: float,
    blur_passes: int = 1,
    sigma: float = 1.0,
    h_value: float = 0.1,
    min_seed_separation: int = 1,
) -> ClassicalSegmentationResult:
    """Run classical LoG + h-maxima + seeded watershed pipeline."""
    volume = np.asarray(volume, dtype=np.float64)
    response = log_enhance_3d(volume, blur_passes=blur_passes, sigma=sigma)
    seeds = h_maxima_seeds(
        response,
        min_value=threshold,
        h=h_value,
        min_separation=min_seed_separation,
    )
    labels = watershed_from_seeds(response, seeds=seeds, threshold=threshold)
    return ClassicalSegmentationResult(
        threshold=threshold,
        seed_count=len(seeds),
        component_count=_count_labels(labels),
        labeled_volume=labels,
        response=response,
    )
