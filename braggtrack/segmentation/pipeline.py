"""Simple segmentation pipeline with Otsu baseline for Week 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from .otsu import otsu_threshold


@dataclass(frozen=True)
class SegmentationResult:
    threshold: float
    voxel_count: int
    component_count: int
    mask: np.ndarray


def connected_components_3d(mask: np.ndarray) -> int:
    """Count connected components in a 3D boolean mask with 6-connectivity."""
    mask = np.asarray(mask, dtype=bool)
    _, n_components = label(mask)
    return n_components


def segment_volume(
    volume: np.ndarray,
    method: str = "otsu",
    threshold: float | None = None,
) -> SegmentationResult:
    """Segment a 3D volume into foreground/background using thresholding."""
    volume = np.asarray(volume, dtype=np.float64)
    if volume.size == 0:
        raise ValueError("Volume must be a non-empty 3D array-like.")

    flat = volume.ravel()

    if method == "otsu":
        th = otsu_threshold(flat) if threshold is None else threshold
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    mask = volume > th
    voxels = int(mask.sum())
    components = connected_components_3d(mask)

    return SegmentationResult(threshold=th, voxel_count=voxels, component_count=components, mask=mask)
