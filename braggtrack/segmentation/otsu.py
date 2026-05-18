"""Pure-Python Otsu thresholding and multi-frame threshold smoothing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from skimage.filters import threshold_otsu


def otsu_threshold(values) -> float:
    """Compute Otsu threshold for a 1D sequence of intensity values.

    Accepts any array-like (numpy array, list, etc.).
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Otsu threshold requires at least one value.")
    if arr.min() == arr.max():
        return float(arr[0])
    return float(threshold_otsu(arr))


def smooth_thresholds(
    per_frame: Sequence[float],
    window: int = 5,
    *,
    mad_scale: float = 3.0,
) -> np.ndarray:
    """Rolling-median smoothing of per-frame Otsu thresholds.

    Steps
    -----
    1. Pad the sequence symmetrically so edge frames get full windows.
    2. Take the rolling median — robust to isolated beam drops / flashes.
    3. Flag frames whose raw threshold deviates from the local median
       by more than ``mad_scale × MAD`` (median absolute deviation).

    Returns an array of smoothed thresholds, one per input frame.
    Flagged outlier frames inherit the local median directly.
    """
    raw = np.asarray(per_frame, dtype=np.float64)
    n = len(raw)
    if n == 0:
        return raw.copy()
    w = max(1, min(window, n))
    half = w // 2
    padded = np.pad(raw, half, mode="reflect")
    smoothed = np.empty(n, dtype=np.float64)
    for i in range(n):
        smoothed[i] = float(np.median(padded[i : i + w]))
    return smoothed


def flag_outlier_frames(
    per_frame: Sequence[float],
    window: int = 5,
    *,
    mad_scale: float = 3.0,
) -> np.ndarray:
    """Boolean mask of frames whose per-frame Otsu is an outlier.

    True means the frame's raw threshold deviates from its rolling median
    by more than ``mad_scale × MAD``. Useful for QC dashboards.
    """
    raw = np.asarray(per_frame, dtype=np.float64)
    smoothed = smooth_thresholds(raw, window=window, mad_scale=mad_scale)
    residual = np.abs(raw - smoothed)
    mad = float(np.median(residual)) if len(raw) > 1 else 0.0
    if mad == 0:
        return residual > 0
    return residual > mad_scale * mad
