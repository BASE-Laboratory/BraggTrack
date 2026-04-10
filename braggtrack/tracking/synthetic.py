"""Synthetic spot-table generator for testing tracking with known ground truth.

Produces multi-frame scenarios with deliberate crossing trajectories,
birth/death events, and near-overlap cases so that Week 3 metrics
(ID-switch rate, fragmentation) are meaningful even without real data.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _make_spot(
    label: int,
    mu: float,
    chi: float,
    d: float,
    eig: tuple[float, float, float] = (0.5, 0.5, 0.5),
    intensity: float = 100.0,
) -> dict[str, Any]:
    return {
        "label": label,
        "voxel_count": 27,
        "integrated_intensity": intensity,
        "centroid_mu": mu,
        "centroid_chi": chi,
        "centroid_d": d,
        "bbox_min_z": 0,
        "bbox_max_z": 0,
        "bbox_min_y": 0,
        "bbox_max_y": 0,
        "bbox_min_x": 0,
        "bbox_max_x": 0,
        "cov_zz": eig[0],
        "cov_yy": eig[1],
        "cov_xx": eig[2],
        "cov_zy": 0.0,
        "cov_zx": 0.0,
        "cov_yx": 0.0,
        "eig_1": eig[0],
        "eig_2": eig[1],
        "eig_3": eig[2],
    }


def generate_crossing_scenario(
    n_scans: int = 3,
    seed: int = 42,
) -> tuple[list[list[dict]], list[dict[str, int]]]:
    """Generate a scenario with 6-8 spots, crossing paths, and birth/death.

    Returns
    -------
    scan_tables : list[list[dict]]
        Per-scan feature tables.
    ground_truth : list[dict]
        ``{"scan_idx", "spot_idx", "true_id"}`` for every observation.
    """
    rng = random.Random(seed)

    # Define ground-truth trajectories as (true_id, start_scan, end_scan,
    # position_at_each_scan).  Some trajectories deliberately cross paths.
    trajectories: list[dict[str, Any]] = [
        # Persistent spots that cross in mu/chi around scan 1
        {"true_id": 1, "start": 0, "end": n_scans - 1,
         "positions": [(2.0 + t * 3.0, 10.0 - t * 2.0, 5.0 + t * 0.1) for t in range(n_scans)],
         "eig": (0.5, 0.4, 0.3)},
        {"true_id": 2, "start": 0, "end": n_scans - 1,
         "positions": [(8.0 - t * 3.0, 4.0 + t * 2.0, 5.1 + t * 0.1) for t in range(n_scans)],
         "eig": (0.6, 0.5, 0.2)},
        # Persistent spot, no crossing
        {"true_id": 3, "start": 0, "end": n_scans - 1,
         "positions": [(15.0 + t * 0.5, 15.0 + t * 0.3, 8.0 - t * 0.2) for t in range(n_scans)],
         "eig": (0.7, 0.6, 0.5)},
        # Born in scan 1
        {"true_id": 4, "start": 1, "end": n_scans - 1,
         "positions": [(20.0, 20.0, 12.0 + t * 0.1) for t in range(n_scans)],
         "eig": (0.4, 0.4, 0.4)},
        # Dies after scan 0
        {"true_id": 5, "start": 0, "end": 0,
         "positions": [(12.0, 3.0, 3.0)] * n_scans,
         "eig": (0.5, 0.5, 0.5)},
    ]

    # Add a 6th spot if we have 3+ scans — born in last scan
    if n_scans >= 3:
        trajectories.append(
            {"true_id": 6, "start": n_scans - 1, "end": n_scans - 1,
             "positions": [(1.0, 1.0, 1.0)] * n_scans,
             "eig": (0.3, 0.3, 0.3)},
        )

    scan_tables: list[list[dict]] = []
    ground_truth: list[dict[str, int]] = []

    for scan_idx in range(n_scans):
        spots: list[dict] = []
        for traj in trajectories:
            if traj["start"] <= scan_idx <= traj["end"]:
                mu, chi, d = traj["positions"][scan_idx]
                # Add small jitter for realism
                mu += rng.gauss(0, 0.05)
                chi += rng.gauss(0, 0.05)
                d += rng.gauss(0, 0.01)
                spot_idx = len(spots)
                spots.append(_make_spot(
                    label=spot_idx + 1,
                    mu=mu,
                    chi=chi,
                    d=d,
                    eig=traj["eig"],
                    intensity=80.0 + rng.gauss(0, 5),
                ))
                ground_truth.append({
                    "scan_idx": scan_idx,
                    "spot_idx": spot_idx,
                    "true_id": traj["true_id"],
                })
        scan_tables.append(spots)

    return scan_tables, ground_truth
