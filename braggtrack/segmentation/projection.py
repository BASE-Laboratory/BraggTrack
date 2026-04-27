"""Projections of 3-D label volumes for QC visualisation.

The naive ``labels.max(axis=k)`` picks the *largest label id* along each ray,
which is meaningless — it produces a Voronoi-looking tiling even when
intensity max-IP shows only a few spots. These helpers project along
whichever voxel is brightest on the same ray, and (optionally) hide rays
whose maximum intensity is below an MIP-domain floor.
"""

from __future__ import annotations

import numpy as np

from .otsu import otsu_threshold


def label_projection_by_intensity(
    intensity: np.ndarray,
    labels: np.ndarray,
    axis: int = 0,
    mip_floor: float | None = None,
) -> np.ndarray:
    """Project ``labels`` by selecting the voxel with max ``intensity`` per ray.

    Parameters
    ----------
    intensity
        3-D intensity cube (usually the raw volume).
    labels
        3-D label volume aligned with ``intensity``.
    axis
        Axis to collapse.
    mip_floor
        If given, pixels whose max-IP intensity is below this floor get
        label 0. Use :func:`otsu_floor_from_mip` for a sensible default
        when you want to hide diffuse background.
    """
    intensity = np.asarray(intensity)
    labels = np.asarray(labels)
    if intensity.shape != labels.shape:
        raise ValueError(f"shape mismatch: intensity={intensity.shape} labels={labels.shape}")
    idx = np.argmax(intensity, axis=axis, keepdims=True)
    projected = np.take_along_axis(labels, idx, axis=axis).squeeze(axis=axis)
    if mip_floor is not None:
        mip = intensity.max(axis=axis)
        projected = np.where(mip >= mip_floor, projected, 0)
    return projected.astype(labels.dtype, copy=False)


def otsu_floor_from_mip(intensity: np.ndarray, axis: int = 0, *, scale: float = 1.0) -> float:
    """Otsu threshold on the 2-D max-IP of ``intensity``, optionally scaled.

    A scale in the 0.88–0.95 range is handy when the MIP histogram is
    narrow and you want to admit slightly more structure than raw Otsu.
    """
    mip = np.asarray(intensity).max(axis=axis)
    return float(otsu_threshold(mip.ravel())) * float(scale)
