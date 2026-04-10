"""Simple segmentation pipeline with Otsu baseline for Week 2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .otsu import otsu_threshold


@dataclass(frozen=True)
class SegmentationResult:
    threshold: float
    voxel_count: int
    component_count: int
    mask: list[list[list[bool]]]


def _flatten(volume: list[list[list[float]]]) -> list[float]:
    return [v for plane in volume for row in plane for v in row]


def _neighbors(z: int, y: int, x: int, shape: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    z_max, y_max, x_max = shape
    for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx < x_max:
            yield nz, ny, nx


def connected_components_3d(mask: list[list[list[bool]]]) -> int:
    """Count connected components in a 3D boolean mask with 6-connectivity."""

    z_max = len(mask)
    y_max = len(mask[0]) if z_max else 0
    x_max = len(mask[0][0]) if y_max else 0

    visited = set()
    count = 0

    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):
                if not mask[z][y][x] or (z, y, x) in visited:
                    continue
                count += 1
                queue = deque([(z, y, x)])
                visited.add((z, y, x))
                while queue:
                    cz, cy, cx = queue.popleft()
                    for nz, ny, nx in _neighbors(cz, cy, cx, (z_max, y_max, x_max)):
                        if mask[nz][ny][nx] and (nz, ny, nx) not in visited:
                            visited.add((nz, ny, nx))
                            queue.append((nz, ny, nx))

    return count


def segment_volume(
    volume: list[list[list[float]]],
    method: str = "otsu",
    threshold: float | None = None,
) -> SegmentationResult:
    """Segment a 3D volume into foreground/background using thresholding."""

    if not volume or not volume[0] or not volume[0][0]:
        raise ValueError("Volume must be a non-empty 3D array-like list.")

    flat = _flatten(volume)

    if method == "otsu":
        th = otsu_threshold(flat) if threshold is None else threshold
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    mask = [[[voxel > th for voxel in row] for row in plane] for plane in volume]
    voxels = sum(1 for v in flat if v > th)
    components = connected_components_3d(mask)

    return SegmentationResult(threshold=th, voxel_count=voxels, component_count=components, mask=mask)
