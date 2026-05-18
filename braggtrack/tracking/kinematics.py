"""Per-grain kinematic time-series from tracked diffraction spots.

Computes physically meaningful evolution quantities for each tracked grain:
strain (Δd/d₀), misorientation (angular drift in μ and χ), growth/dissolution
(relative intensity and volume changes), and shape evolution (anisotropy,
covariance trace).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GrainKinematics:
    """Time-resolved physical evolution of a single tracked grain."""

    track_id: int
    scan_indices: list[int]

    # Centroids per observation
    centroid_mu: np.ndarray
    centroid_chi: np.ndarray
    centroid_d: np.ndarray

    # Strain: (d - d₀) / d₀
    strain: np.ndarray

    # Misorientation: cumulative angular drift from first observation
    misorientation_mu: np.ndarray
    misorientation_chi: np.ndarray
    misorientation_total: np.ndarray

    # Growth / dissolution
    integrated_intensity: np.ndarray
    voxel_count: np.ndarray
    relative_intensity: np.ndarray
    relative_volume: np.ndarray

    # Shape evolution
    eigenvalues: np.ndarray  # (n_obs, 3)
    anisotropy: np.ndarray  # eig_1 / eig_3
    covariance_trace: np.ndarray  # eig_1 + eig_2 + eig_3


@dataclass
class KinematicsSummary:
    """Aggregate statistics across all tracked grains."""

    n_tracks: int
    n_full_tracks: int
    n_scans: int

    # Per-grain summaries (one entry per track, same order as grain_kinematics)
    track_ids: list[int]
    max_strain: list[float]
    total_misorientation: list[float]
    intensity_change_frac: list[float]
    volume_change_frac: list[float]
    max_anisotropy: list[float]

    grain_kinematics: list[GrainKinematics] = field(repr=False)


def compute_grain_kinematics(
    track_table: list[dict[str, Any]],
) -> list[GrainKinematics]:
    """Compute kinematic time-series for each tracked grain.

    Parameters
    ----------
    track_table
        Output of :func:`~braggtrack.tracking.lifecycle.tracks_to_table`:
        list of dicts with ``track_id``, ``scan_idx``, ``centroid_mu``,
        ``centroid_chi``, ``centroid_d``, ``eig_1``, ``eig_2``, ``eig_3``,
        ``integrated_intensity``, ``voxel_count``.

    Returns
    -------
    list[GrainKinematics]
        One entry per track, sorted by track_id.
    """
    tracks: dict[int, list[dict]] = {}
    for row in track_table:
        tracks.setdefault(int(row["track_id"]), []).append(row)

    results: list[GrainKinematics] = []
    for tid in sorted(tracks):
        obs = sorted(tracks[tid], key=lambda r: r["scan_idx"])
        n = len(obs)

        scan_indices = [int(r["scan_idx"]) for r in obs]
        mu = np.array([float(r["centroid_mu"]) for r in obs])
        chi = np.array([float(r["centroid_chi"]) for r in obs])
        d = np.array([float(r["centroid_d"]) for r in obs])

        # Strain: (d - d₀) / d₀
        d0 = d[0]
        strain = (d - d0) / d0 if d0 != 0 else np.zeros(n)

        # Misorientation: drift from initial position
        dmu = mu - mu[0]
        dchi = chi - chi[0]
        misorientation_total = np.sqrt(dmu**2 + dchi**2)

        intensity = np.array([float(r["integrated_intensity"]) for r in obs])
        voxels = np.array([float(r["voxel_count"]) for r in obs])

        i0 = intensity[0]
        v0 = voxels[0]
        rel_intensity = (intensity - i0) / i0 if i0 != 0 else np.zeros(n)
        rel_volume = (voxels - v0) / v0 if v0 != 0 else np.zeros(n)

        eigs = np.array([[float(r["eig_1"]), float(r["eig_2"]), float(r["eig_3"])] for r in obs])
        e3 = eigs[:, 2]
        anisotropy = np.where(e3 > 0, eigs[:, 0] / e3, 1.0)
        cov_trace = eigs.sum(axis=1)

        results.append(
            GrainKinematics(
                track_id=tid,
                scan_indices=scan_indices,
                centroid_mu=mu,
                centroid_chi=chi,
                centroid_d=d,
                strain=strain,
                misorientation_mu=dmu,
                misorientation_chi=dchi,
                misorientation_total=misorientation_total,
                integrated_intensity=intensity,
                voxel_count=voxels,
                relative_intensity=rel_intensity,
                relative_volume=rel_volume,
                eigenvalues=eigs,
                anisotropy=anisotropy,
                covariance_trace=cov_trace,
            )
        )
    return results


def summarize_kinematics(
    grain_kinematics: list[GrainKinematics],
    n_scans: int,
) -> KinematicsSummary:
    """Aggregate per-grain kinematics into a summary table.

    Parameters
    ----------
    grain_kinematics
        Output of :func:`compute_grain_kinematics`.
    n_scans
        Total number of scans in the sequence.
    """
    track_ids: list[int] = []
    max_strain: list[float] = []
    total_misorientation: list[float] = []
    intensity_change: list[float] = []
    volume_change: list[float] = []
    max_anisotropy: list[float] = []

    for gk in grain_kinematics:
        track_ids.append(gk.track_id)
        max_strain.append(float(np.max(np.abs(gk.strain))))
        total_misorientation.append(float(gk.misorientation_total[-1]) if len(gk.misorientation_total) > 0 else 0.0)
        intensity_change.append(float(gk.relative_intensity[-1]) if len(gk.relative_intensity) > 0 else 0.0)
        volume_change.append(float(gk.relative_volume[-1]) if len(gk.relative_volume) > 0 else 0.0)
        max_anisotropy.append(float(np.max(gk.anisotropy)))

    n_full = sum(1 for gk in grain_kinematics if len(gk.scan_indices) >= n_scans)

    return KinematicsSummary(
        n_tracks=len(grain_kinematics),
        n_full_tracks=n_full,
        n_scans=n_scans,
        track_ids=track_ids,
        max_strain=max_strain,
        total_misorientation=total_misorientation,
        intensity_change_frac=intensity_change,
        volume_change_frac=volume_change,
        max_anisotropy=max_anisotropy,
        grain_kinematics=grain_kinematics,
    )


def kinematics_to_table(
    grain_kinematics: list[GrainKinematics],
) -> list[dict[str, Any]]:
    """Flatten per-grain kinematics into a row-per-observation table.

    Each row contains the track_id, scan_idx, and all computed kinematic
    quantities for that observation. Suitable for CSV export or DataFrame
    construction.
    """
    rows: list[dict[str, Any]] = []
    for gk in grain_kinematics:
        for i, scan_idx in enumerate(gk.scan_indices):
            rows.append(
                {
                    "track_id": gk.track_id,
                    "scan_idx": scan_idx,
                    "centroid_mu": float(gk.centroid_mu[i]),
                    "centroid_chi": float(gk.centroid_chi[i]),
                    "centroid_d": float(gk.centroid_d[i]),
                    "strain": float(gk.strain[i]),
                    "misorientation_mu": float(gk.misorientation_mu[i]),
                    "misorientation_chi": float(gk.misorientation_chi[i]),
                    "misorientation_total": float(gk.misorientation_total[i]),
                    "integrated_intensity": float(gk.integrated_intensity[i]),
                    "voxel_count": float(gk.voxel_count[i]),
                    "relative_intensity": float(gk.relative_intensity[i]),
                    "relative_volume": float(gk.relative_volume[i]),
                    "eig_1": float(gk.eigenvalues[i, 0]),
                    "eig_2": float(gk.eigenvalues[i, 1]),
                    "eig_3": float(gk.eigenvalues[i, 2]),
                    "anisotropy": float(gk.anisotropy[i]),
                    "covariance_trace": float(gk.covariance_trace[i]),
                }
            )
    return rows
