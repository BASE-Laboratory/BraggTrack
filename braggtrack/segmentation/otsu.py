"""Pure-Python Otsu thresholding for baseline segmentation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _clamp_255(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    scaled = (value - lo) / (hi - lo)
    if scaled < 0:
        scaled = 0
    if scaled > 1:
        scaled = 1
    return int(round(scaled * 255))


def otsu_threshold(values: Iterable[float]) -> float:
    """Compute Otsu threshold for a 1D sequence of intensity values."""

    data = [float(v) for v in values]
    if not data:
        raise ValueError("Otsu threshold requires at least one value.")

    lo = min(data)
    hi = max(data)
    if lo == hi:
        return lo

    hist = [0] * 256
    for v in data:
        hist[_clamp_255(v, lo, hi)] += 1

    total = len(data)
    sum_total = sum(i * hist[i] for i in range(256))

    sum_bg = 0.0
    weight_bg = 0
    best_var = -1.0
    best_idx = 0

    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_idx = i

    # Use bin center to avoid degenerate thresholds at exact low mode value.
    return lo + ((best_idx + 0.5) / 255.0) * (hi - lo)


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
    residual = np.abs(raw - smoothed)
    mad = float(np.median(residual)) if n > 1 else 0.0
    if mad > 0:
        outlier = residual > mad_scale * mad
        smoothed[outlier] = smoothed[outlier]  # already local median
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
        # MAD is zero when most frames are identical. Fall back to
        # flagging any frame with nonzero deviation from the local median.
        return residual > 0
    return residual > mad_scale * mad
