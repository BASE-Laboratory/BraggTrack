"""Run a Week 2 segmentation smoke test on a synthetic 3D volume."""

from __future__ import annotations

import json

from braggtrack.segmentation import segment_volume


def make_volume(size: int = 12) -> list[list[list[float]]]:
    volume = [[[1.0 for _ in range(size)] for _ in range(size)] for _ in range(size)]

    centers = [(3, 3, 3), (8, 8, 7)]
    for cz, cy, cx in centers:
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    dist2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
                    if dist2 <= 4:
                        volume[z][y][x] = 20.0
    return volume


def main() -> int:
    volume = make_volume()
    result = segment_volume(volume, method="otsu")

    payload = {
        "method": "otsu",
        "threshold": result.threshold,
        "voxel_count": result.voxel_count,
        "component_count": result.component_count,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
