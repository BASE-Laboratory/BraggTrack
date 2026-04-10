"""Post-processing helpers for labeled 3D masks."""

from __future__ import annotations

from collections import deque


def remove_small_objects(labels: list[list[list[int]]], min_size: int) -> list[list[list[int]]]:
    counts: dict[int, int] = {}
    for plane in labels:
        for row in plane:
            for v in row:
                if v > 0:
                    counts[v] = counts.get(v, 0) + 1

    keep = {label for label, size in counts.items() if size >= min_size}
    return [[[v if v in keep else 0 for v in row] for row in plane] for plane in labels]


def fill_holes_binary(mask: list[list[list[bool]]]) -> list[list[list[bool]]]:
    """Fill enclosed holes by flood-filling background from boundaries."""

    z_max = len(mask)
    y_max = len(mask[0]) if z_max else 0
    x_max = len(mask[0][0]) if y_max else 0

    outside = [[[False for _ in range(x_max)] for _ in range(y_max)] for _ in range(z_max)]
    q: deque[tuple[int, int, int]] = deque()

    def push_if_bg(z: int, y: int, x: int) -> None:
        if 0 <= z < z_max and 0 <= y < y_max and 0 <= x < x_max:
            if (not mask[z][y][x]) and (not outside[z][y][x]):
                outside[z][y][x] = True
                q.append((z, y, x))

    for z in range(z_max):
        for y in range(y_max):
            push_if_bg(z, y, 0)
            push_if_bg(z, y, x_max - 1)
        for x in range(x_max):
            push_if_bg(z, 0, x)
            push_if_bg(z, y_max - 1, x)
    for y in range(y_max):
        for x in range(x_max):
            push_if_bg(0, y, x)
            push_if_bg(z_max - 1, y, x)

    while q:
        z, y, x = q.popleft()
        for nz, ny, nx in ((z - 1, y, x), (z + 1, y, x), (z, y - 1, x), (z, y + 1, x), (z, y, x - 1), (z, y, x + 1)):
            if 0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx < x_max:
                if (not mask[nz][ny][nx]) and (not outside[nz][ny][nx]):
                    outside[nz][ny][nx] = True
                    q.append((nz, ny, nx))

    filled = [[[mask[z][y][x] or (not outside[z][y][x]) for x in range(x_max)] for y in range(y_max)] for z in range(z_max)]
    return filled


def relabel_sequential(labels: list[list[list[int]]]) -> list[list[list[int]]]:
    mapping: dict[int, int] = {}
    next_id = 1
    out = []
    for plane in labels:
        out_plane = []
        for row in plane:
            out_row = []
            for v in row:
                if v <= 0:
                    out_row.append(0)
                else:
                    if v not in mapping:
                        mapping[v] = next_id
                        next_id += 1
                    out_row.append(mapping[v])
            out_plane.append(out_row)
        out.append(out_plane)
    return out
