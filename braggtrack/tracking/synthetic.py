"""Synthetic spot-table generator for testing tracking with known ground truth.

Produces multi-frame scenarios with deliberate crossing trajectories,
birth/death events, and near-overlap cases so that tracking metrics
(ID-switch rate, fragmentation) are meaningful even without real data.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


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
        {
            "true_id": 1,
            "start": 0,
            "end": n_scans - 1,
            "positions": [(2.0 + t * 3.0, 10.0 - t * 2.0, 5.0 + t * 0.1) for t in range(n_scans)],
            "eig": (0.5, 0.4, 0.3),
        },
        {
            "true_id": 2,
            "start": 0,
            "end": n_scans - 1,
            "positions": [(8.0 - t * 3.0, 4.0 + t * 2.0, 5.1 + t * 0.1) for t in range(n_scans)],
            "eig": (0.6, 0.5, 0.2),
        },
        # Persistent spot, no crossing
        {
            "true_id": 3,
            "start": 0,
            "end": n_scans - 1,
            "positions": [(15.0 + t * 0.5, 15.0 + t * 0.3, 8.0 - t * 0.2) for t in range(n_scans)],
            "eig": (0.7, 0.6, 0.5),
        },
        # Born in scan 1
        {
            "true_id": 4,
            "start": 1,
            "end": n_scans - 1,
            "positions": [(20.0, 20.0, 12.0 + t * 0.1) for t in range(n_scans)],
            "eig": (0.4, 0.4, 0.4),
        },
        # Dies after scan 0
        {"true_id": 5, "start": 0, "end": 0, "positions": [(12.0, 3.0, 3.0)] * n_scans, "eig": (0.5, 0.5, 0.5)},
    ]

    # Add a 6th spot if we have 3+ scans — born in last scan
    if n_scans >= 3:
        trajectories.append(
            {
                "true_id": 6,
                "start": n_scans - 1,
                "end": n_scans - 1,
                "positions": [(1.0, 1.0, 1.0)] * n_scans,
                "eig": (0.3, 0.3, 0.3),
            },
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
                spots.append(
                    _make_spot(
                        label=spot_idx + 1,
                        mu=mu,
                        chi=chi,
                        d=d,
                        eig=traj["eig"],
                        intensity=80.0 + rng.gauss(0, 5),
                    )
                )
                ground_truth.append(
                    {
                        "scan_idx": scan_idx,
                        "spot_idx": spot_idx,
                        "true_id": traj["true_id"],
                    }
                )
        scan_tables.append(spots)

    return scan_tables, ground_truth


def _add_gaussian_blob(
    volume: np.ndarray,
    center: tuple[float, float, float],
    sigma: tuple[float, float, float],
    amplitude: float,
) -> np.ndarray:
    """Add a 3D Gaussian lobe to *volume* and return its isolated contribution."""
    zz, yy, xx = np.indices(volume.shape, dtype=np.float64)
    cz, cy, cx = center
    sz, sy, sx = sigma
    exponent = -0.5 * (((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)
    lobe = amplitude * np.exp(exponent)
    volume += lobe
    return lobe


def generate_fracture_scenario(
    n_scans: int = 5,
    shape: tuple[int, int, int] = (24, 48, 48),
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[dict[str, Any]]], dict[str, Any]]:
    """Generate a 4D synthetic scenario with precursor broadening, split, merge, and control grain.

    Scenario layout across *n_scans* (requires ``n_scans >= 4``):
      - **Grain A (fracturing):** At ``(z=12, y=12, x=14)``. During scans ``0..1`` it
        develops bimodal sub-lobes inside a single connected reflection (persistent
        fracture precursor), then cleaves into two distinct daughter reflections at
        scan ``2`` (``1 -> 2`` split) with diverging ``d`` shifts.
      - **Grain B (coalescing):** Two separate compact reflections near
        ``(z=12, y=34, x=11..17)`` during scans ``0..1`` that merge into a single
        reflection at scan ``2`` (``2 -> 1`` merge) conserving integrated intensity.
      - **Grain C (control):** Stable isotropic grain at ``(z=12, y=24, x=36)``
        across all scans (must produce zero splits/merges and zero precursors).
    """
    from braggtrack.segmentation.features import extract_instance_table

    if n_scans < 4:
        raise ValueError("generate_fracture_scenario requires n_scans >= 4")

    rng = np.random.default_rng(seed)
    volumes: list[np.ndarray] = []
    label_volumes: list[np.ndarray] = []
    scan_tables: list[list[dict[str, Any]]] = []

    split_scan = 2
    merge_scan = 2

    for t in range(n_scans):
        vol = rng.uniform(0.0, 0.2, size=shape).astype(np.float64)
        labels = np.zeros(shape, dtype=np.int32)
        next_label = 1

        # --- Grain A: Precursor (scans 0..1) -> Split (scan >= 2) ---
        if t < split_scan:
            # Progressive internal sub-lobe separation inside one connected mask.
            half_sep = 1.9 + 0.5 * t  # 1.9 at t=0, 2.4 at t=1 (distinct internal maxima, overlapping support)
            lobe1 = _add_gaussian_blob(vol, (12.0, 12.0, 14.0 - half_sep), (1.3, 1.3, 1.4 + 0.3 * t), 55.0)
            lobe2 = _add_gaussian_blob(vol, (12.0, 12.0, 14.0 + half_sep), (1.3, 1.3, 1.4 + 0.3 * t), 50.0)
            combined = lobe1 + lobe2
            labels[combined >= 4.0] = next_label
            next_label += 1
        else:
            dt = t - split_scan
            # Cleaved into two separated daughter fragments with diverging d (y-axis).
            d1 = _add_gaussian_blob(vol, (12.0, 12.0 - 0.4 * dt, 10.0 - 0.3 * dt), (1.3, 1.3, 1.3), 54.0)
            labels[d1 >= 4.0] = next_label
            next_label += 1

            d2 = _add_gaussian_blob(vol, (12.0, 12.0 + 0.5 * dt, 18.0 + 0.3 * dt), (1.3, 1.3, 1.3), 49.0)
            labels[(d2 >= 4.0) & (labels == 0)] = next_label
            next_label += 1

        # --- Grain B: Two compact sub-domains (scans 0..1) -> Merge (scan >= 2) ---
        if t < merge_scan:
            offset = 3.5 - 0.5 * t  # 3.5 at t=0, 3.0 at t=1 (well-separated masks)
            m1 = _add_gaussian_blob(vol, (12.0, 34.0, 14.0 - offset), (1.2, 1.2, 1.2), 45.0)
            labels[(m1 >= 5.0) & (labels == 0)] = next_label
            next_label += 1

            m2 = _add_gaussian_blob(vol, (12.0, 34.0, 14.0 + offset), (1.2, 1.2, 1.2), 45.0)
            labels[(m2 >= 5.0) & (labels == 0)] = next_label
            next_label += 1
        else:
            merged = _add_gaussian_blob(vol, (12.0, 34.0, 14.0), (1.35, 1.35, 1.35), 68.0)
            labels[(merged >= 5.0) & (labels == 0)] = next_label
            next_label += 1

        # --- Grain C: Stable isotropic control grain ---
        ctrl = _add_gaussian_blob(vol, (12.0, 24.0, 36.0), (1.4, 1.4, 1.4), 70.0)
        labels[(ctrl >= 5.0) & (labels == 0)] = next_label

        volumes.append(vol)
        label_volumes.append(labels)
        scan_tables.append(extract_instance_table(labels, vol))

    expected = {
        "n_scans": n_scans,
        "split_scan_from": split_scan - 1,
        "split_scan_to": split_scan,
        "merge_scan_from": merge_scan - 1,
        "merge_scan_to": merge_scan,
        "fracture_initial_mu_chi_d": (12.0, 14.0, 12.0),
        "merge_target_mu_chi_d": (12.0, 14.0, 34.0),
        "control_mu_chi_d": (12.0, 36.0, 24.0),
    }
    return volumes, label_volumes, scan_tables, expected
