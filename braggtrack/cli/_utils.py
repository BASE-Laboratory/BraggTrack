"""Shared CLI helpers for volume loading, CSV I/O, and synthetic fallback."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def synth_volume_from_file(path: Path, size: int = 24) -> np.ndarray:
    """Generate a deterministic synthetic volume with Gaussian blobs."""
    digest = hashlib.sha256(path.read_bytes()[:4096]).digest()
    seed_vals = [b for b in digest[:12]]
    volume = np.ones((size, size, size), dtype=np.float64)
    centers = [
        (4 + seed_vals[0] % 8, 4 + seed_vals[1] % 8, 4 + seed_vals[2] % 8),
        (10 + seed_vals[3] % 8, 10 + seed_vals[4] % 8, 10 + seed_vals[5] % 8),
        (6 + seed_vals[6] % 10, 6 + seed_vals[7] % 10, 6 + seed_vals[8] % 10),
    ]
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    for cz, cy, cx in centers:
        amp = 10.0 + (seed_vals[(cz + cy + cx) % len(seed_vals)] % 20)
        sigma_blob = 1.5
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume += amp * np.exp(-d2 / (2.0 * sigma_blob**2))
    return volume


def load_feature_csv(path: Path) -> list[dict[str, Any]]:
    """Load a features.csv into a list of dicts with numeric types."""
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            typed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v)
                except ValueError:
                    try:
                        typed[k] = float(v)
                    except ValueError:
                        typed[k] = v
            rows.append(typed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to CSV with auto-detected fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
