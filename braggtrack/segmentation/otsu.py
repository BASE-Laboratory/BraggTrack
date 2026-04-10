"""Pure-Python Otsu thresholding for baseline segmentation."""

from __future__ import annotations

from typing import Iterable


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
