"""Classical 3D segmentation building blocks for Week 2.1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalSegmentationResult:
    threshold: float
    seed_count: int
    component_count: int
    labeled_volume: list[list[list[int]]]


def _shape(volume: list[list[list[float]]]) -> tuple[int, int, int]:
    return len(volume), len(volume[0]), len(volume[0][0])


def _zeros_like(volume: list[list[list[float]]], value: float = 0.0) -> list[list[list[float]]]:
    z, y, x = _shape(volume)
    return [[[value for _ in range(x)] for _ in range(y)] for _ in range(z)]


def gaussian_blur_3d(volume: list[list[list[float]]], passes: int = 1) -> list[list[list[float]]]:
    """Apply a lightweight separable [1,2,1]/4 blur along each axis."""

    z_max, y_max, x_max = _shape(volume)
    current = [[[float(v) for v in row] for row in plane] for plane in volume]

    def blur_axis_x(src: list[list[list[float]]]) -> list[list[list[float]]]:
        out = _zeros_like(src)
        for z in range(z_max):
            for y in range(y_max):
                for x in range(x_max):
                    left = src[z][y][x - 1] if x > 0 else src[z][y][x]
                    mid = src[z][y][x]
                    right = src[z][y][x + 1] if x + 1 < x_max else src[z][y][x]
                    out[z][y][x] = (left + 2 * mid + right) / 4.0
        return out

    def blur_axis_y(src: list[list[list[float]]]) -> list[list[list[float]]]:
        out = _zeros_like(src)
        for z in range(z_max):
            for y in range(y_max):
                for x in range(x_max):
                    up = src[z][y - 1][x] if y > 0 else src[z][y][x]
                    mid = src[z][y][x]
                    down = src[z][y + 1][x] if y + 1 < y_max else src[z][y][x]
                    out[z][y][x] = (up + 2 * mid + down) / 4.0
        return out

    def blur_axis_z(src: list[list[list[float]]]) -> list[list[list[float]]]:
        out = _zeros_like(src)
        for z in range(z_max):
            for y in range(y_max):
                for x in range(x_max):
                    prev = src[z - 1][y][x] if z > 0 else src[z][y][x]
                    mid = src[z][y][x]
                    nxt = src[z + 1][y][x] if z + 1 < z_max else src[z][y][x]
                    out[z][y][x] = (prev + 2 * mid + nxt) / 4.0
        return out

    for _ in range(max(1, passes)):
        current = blur_axis_x(current)
        current = blur_axis_y(current)
        current = blur_axis_z(current)

    return current


def laplacian_3d(volume: list[list[list[float]]]) -> list[list[list[float]]]:
    """6-neighbor discrete Laplacian."""

    z_max, y_max, x_max = _shape(volume)
    out = _zeros_like(volume)

    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):
                center = volume[z][y][x]
                neighbors = [
                    volume[z - 1][y][x] if z > 0 else center,
                    volume[z + 1][y][x] if z + 1 < z_max else center,
                    volume[z][y - 1][x] if y > 0 else center,
                    volume[z][y + 1][x] if y + 1 < y_max else center,
                    volume[z][y][x - 1] if x > 0 else center,
                    volume[z][y][x + 1] if x + 1 < x_max else center,
                ]
                out[z][y][x] = sum(neighbors) - 6.0 * center

    return out


def log_enhance_3d(volume: list[list[list[float]]], blur_passes: int = 1) -> list[list[list[float]]]:
    """LoG-like enhancement: blur then negative Laplacian."""

    smoothed = gaussian_blur_3d(volume, passes=blur_passes)
    lap = laplacian_3d(smoothed)
    return [[[-v for v in row] for row in plane] for plane in lap]


def local_maxima_seeds(
    volume: list[list[list[float]]],
    min_value: float,
    min_separation: int = 1,
) -> list[tuple[int, int, int]]:
    """Find local maxima seeds above a minimum value."""

    z_max, y_max, x_max = _shape(volume)
    seeds: list[tuple[int, int, int, float]] = []

    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):
                center = volume[z][y][x]
                if center < min_value:
                    continue

                is_max = True
                for nz in range(max(0, z - 1), min(z_max, z + 2)):
                    for ny in range(max(0, y - 1), min(y_max, y + 2)):
                        for nx in range(max(0, x - 1), min(x_max, x + 2)):
                            if (nz, ny, nx) == (z, y, x):
                                continue
                            if volume[nz][ny][nx] > center:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if not is_max:
                        break

                if is_max:
                    seeds.append((z, y, x, center))

    seeds.sort(key=lambda item: item[3], reverse=True)
    picked: list[tuple[int, int, int]] = []

    for z, y, x, _ in seeds:
        if all(abs(z - pz) > min_separation or abs(y - py) > min_separation or abs(x - px) > min_separation for pz, py, px in picked):
            picked.append((z, y, x))

    return picked


def watershed_from_seeds(
    response: list[list[list[float]]],
    seeds: list[tuple[int, int, int]],
    threshold: float,
) -> list[list[list[int]]]:
    """Simple multi-source region growing over voxels above threshold."""

    z_max, y_max, x_max = _shape(response)
    labels = [[[0 for _ in range(x_max)] for _ in range(y_max)] for _ in range(z_max)]
    queue: deque[tuple[int, int, int, int]] = deque()

    for label_id, (z, y, x) in enumerate(seeds, start=1):
        if response[z][y][x] >= threshold:
            labels[z][y][x] = label_id
            queue.append((z, y, x, label_id))

    while queue:
        z, y, x, label_id = queue.popleft()
        for nz, ny, nx in ((z - 1, y, x), (z + 1, y, x), (z, y - 1, x), (z, y + 1, x), (z, y, x - 1), (z, y, x + 1)):
            if not (0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx < x_max):
                continue
            if labels[nz][ny][nx] != 0:
                continue
            if response[nz][ny][nx] < threshold:
                continue
            labels[nz][ny][nx] = label_id
            queue.append((nz, ny, nx, label_id))

    return labels


def _count_labels(labels: list[list[list[int]]]) -> int:
    found = {v for plane in labels for row in plane for v in row if v > 0}
    return len(found)


def segment_classical(
    volume: list[list[list[float]]],
    threshold: float,
    blur_passes: int = 1,
    min_seed_separation: int = 1,
) -> ClassicalSegmentationResult:
    """Run classical LoG + local maxima + seeded watershed pipeline."""

    response = log_enhance_3d(volume, blur_passes=blur_passes)
    seeds = local_maxima_seeds(response, min_value=threshold, min_separation=min_seed_separation)
    labels = watershed_from_seeds(response, seeds=seeds, threshold=threshold)
    return ClassicalSegmentationResult(
        threshold=threshold,
        seed_count=len(seeds),
        component_count=_count_labels(labels),
        labeled_volume=labels,
    )
