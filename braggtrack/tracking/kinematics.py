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

    # Raw shift in d-axis (always meaningful even when uncalibrated)
    delta_d: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Analytical 1-sigma uncertainties (propagated from centroid_*_err)
    centroid_mu_err: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    centroid_chi_err: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    centroid_d_err: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    strain_err: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Ancestor-referenced lineage strain across 1->2+ splits and 2+->1 merges
    root_track_id: int = 0
    lineage_strain: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    lineage_strain_err: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Strictly causal backward finite-difference rates (per scan step, plus per-second when elapsed_s known)
    rate_mu: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rate_chi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rate_d: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    strain_rate: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    strain_rate_per_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


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

    calibrated_axes: bool = False
    max_lineage_strain: list[float] = field(default_factory=list)
    grain_kinematics: list[GrainKinematics] = field(default_factory=list, repr=False)


def _parse_parent_ids(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for item in raw:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, (int, np.integer)):
        return [int(raw)]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        parsed: list[int] = []
        for p in parts:
            try:
                parsed.append(int(p))
            except ValueError:
                continue
        return parsed
    return []


def _propagate_strain_err(
    d: np.ndarray,
    d_err: np.ndarray,
    d_ref: float,
    d_ref_err: float,
    *,
    zero_first: bool,
) -> np.ndarray:
    """Propagate 1-sigma uncertainty on ``(d - d_ref) / d_ref``."""
    n = len(d)
    if n == 0 or d_ref == 0.0:
        return np.zeros(n, dtype=np.float64)
    ratio = d / d_ref
    err = np.sqrt(d_err**2 + (ratio * d_ref_err) ** 2) / abs(d_ref)
    if zero_first and n > 0:
        err[0] = 0.0
    return err


def _obs_elapsed_s(row: dict[str, Any], scan_elapsed_s: dict[int, float] | None) -> float | None:
    s_idx = int(row["scan_idx"])
    if scan_elapsed_s is not None and s_idx in scan_elapsed_s:
        val = scan_elapsed_s[s_idx]
        if val is not None and np.isfinite(val):
            return float(val)
    raw = row.get("elapsed_s")
    if raw is not None and raw != "":
        try:
            f = float(raw)
            if np.isfinite(f):
                return f
        except (TypeError, ValueError):
            return None
    return None


def compute_grain_kinematics(
    track_table: list[dict[str, Any]],
    *,
    scan_elapsed_s: dict[int, float] | None = None,
) -> list[GrainKinematics]:
    """Compute kinematic time-series for each tracked grain.

    Parameters
    ----------
    track_table
        Output of :func:`~braggtrack.tracking.lifecycle.tracks_to_table`:
        list of dicts with ``track_id``, ``scan_idx``, ``centroid_mu``,
        ``centroid_chi``, ``centroid_d``, ``eig_1``, ``eig_2``, ``eig_3``,
        ``integrated_intensity``, ``voxel_count``, plus optional
        ``centroid_*_err`` and ``parent_track_ids``.
    scan_elapsed_s
        Optional mapping ``scan_idx -> elapsed_seconds`` for non-uniform scan cadences.

    Returns
    -------
    list[GrainKinematics]
        One entry per track, sorted by track_id.
    """
    tracks: dict[int, list[dict[str, Any]]] = {}
    parent_map: dict[int, list[int]] = {}
    for row in track_table:
        tid = int(row["track_id"])
        tracks.setdefault(tid, []).append(row)
        pids = _parse_parent_ids(row.get("parent_track_ids"))
        if pids and tid not in parent_map:
            parent_map[tid] = pids

    sorted_tracks: dict[int, list[dict[str, Any]]] = {
        tid: sorted(rows, key=lambda r: int(r["scan_idx"])) for tid, rows in tracks.items()
    }

    def _resolve_root_tid(start_tid: int) -> int:
        curr = start_tid
        visited: set[int] = set()
        while curr not in visited:
            visited.add(curr)
            parents = [p for p in parent_map.get(curr, []) if p in sorted_tracks and p != curr]
            if not parents:
                break
            # Prefer earliest-onset parent (tie-break by highest initial intensity, then lowest ID)
            parents.sort(
                key=lambda p: (
                    int(sorted_tracks[p][0]["scan_idx"]),
                    -float(sorted_tracks[p][0].get("integrated_intensity", 0.0)),
                    p,
                )
            )
            curr = parents[0]
        return curr

    results: list[GrainKinematics] = []
    for tid in sorted(sorted_tracks):
        obs = sorted_tracks[tid]
        n = len(obs)

        scan_indices = [int(r["scan_idx"]) for r in obs]
        mu = np.array([float(r["centroid_mu"]) for r in obs], dtype=np.float64)
        chi = np.array([float(r["centroid_chi"]) for r in obs], dtype=np.float64)
        d = np.array([float(r["centroid_d"]) for r in obs], dtype=np.float64)

        mu_err = np.array([float(r.get("centroid_mu_err", 0.0)) for r in obs], dtype=np.float64)
        chi_err = np.array([float(r.get("centroid_chi_err", 0.0)) for r in obs], dtype=np.float64)
        d_err = np.array([float(r.get("centroid_d_err", 0.0)) for r in obs], dtype=np.float64)

        # Segment-local strain: (d - d₀) / d₀ and raw shift Δd = d - d₀
        d0 = float(d[0])
        d0_err = float(d_err[0])
        delta_d = d - d0
        strain = delta_d / d0 if d0 != 0.0 else np.zeros(n, dtype=np.float64)
        strain_err = _propagate_strain_err(d, d_err, d0, d0_err, zero_first=True)

        # Ancestor-referenced lineage strain across splits/merges
        root_tid = _resolve_root_tid(tid)
        root_obs0 = sorted_tracks[root_tid][0]
        d_root0 = float(root_obs0["centroid_d"])
        d_root0_err = float(root_obs0.get("centroid_d_err", 0.0))
        lineage_strain = (d - d_root0) / d_root0 if d_root0 != 0.0 else np.zeros(n, dtype=np.float64)
        lineage_strain_err = _propagate_strain_err(
            d,
            d_err,
            d_root0,
            d_root0_err,
            zero_first=(root_tid == tid),
        )

        # Strictly causal backward finite-difference rates (per scan step & wall-clock /s)
        rate_mu = np.zeros(n, dtype=np.float64)
        rate_chi = np.zeros(n, dtype=np.float64)
        rate_d = np.zeros(n, dtype=np.float64)
        strain_rate = np.zeros(n, dtype=np.float64)
        strain_rate_per_s = np.zeros(n, dtype=np.float64)

        # Seed initial rate at i=0 from immediate parent's last observation if split/merge child
        imm_parents = [p for p in parent_map.get(tid, []) if p in sorted_tracks]
        if imm_parents and n > 0:
            p_last = sorted_tracks[imm_parents[0]][-1]
            dt0 = max(1, scan_indices[0] - int(p_last["scan_idx"]))
            rate_mu[0] = (mu[0] - float(p_last["centroid_mu"])) / dt0
            rate_chi[0] = (chi[0] - float(p_last["centroid_chi"])) / dt0
            rate_d[0] = (d[0] - float(p_last["centroid_d"])) / dt0
            if d_root0 != 0.0:
                p_lin_strain = (float(p_last["centroid_d"]) - d_root0) / d_root0
                strain_rate[0] = (lineage_strain[0] - p_lin_strain) / dt0
                t_now0 = _obs_elapsed_s(obs[0], scan_elapsed_s)
                t_prev0 = _obs_elapsed_s(p_last, scan_elapsed_s)
                if t_now0 is not None and t_prev0 is not None and (t_now0 - t_prev0) > 1e-6:
                    strain_rate_per_s[0] = (lineage_strain[0] - p_lin_strain) / (t_now0 - t_prev0)

        for i in range(1, n):
            dt = max(1, scan_indices[i] - scan_indices[i - 1])
            rate_mu[i] = (mu[i] - mu[i - 1]) / dt
            rate_chi[i] = (chi[i] - chi[i - 1]) / dt
            rate_d[i] = (d[i] - d[i - 1]) / dt
            strain_rate[i] = (lineage_strain[i] - lineage_strain[i - 1]) / dt
            t_now = _obs_elapsed_s(obs[i], scan_elapsed_s)
            t_prev = _obs_elapsed_s(obs[i - 1], scan_elapsed_s)
            if t_now is not None and t_prev is not None and (t_now - t_prev) > 1e-6:
                strain_rate_per_s[i] = (lineage_strain[i] - lineage_strain[i - 1]) / (t_now - t_prev)

        # Misorientation: drift from initial position
        dmu = mu - mu[0]
        dchi = chi - chi[0]
        misorientation_total = np.sqrt(dmu**2 + dchi**2)

        intensity = np.array([float(r["integrated_intensity"]) for r in obs], dtype=np.float64)
        voxels = np.array([float(r["voxel_count"]) for r in obs], dtype=np.float64)

        i0 = intensity[0]
        v0 = voxels[0]
        rel_intensity = (intensity - i0) / i0 if i0 != 0 else np.zeros(n, dtype=np.float64)
        rel_volume = (voxels - v0) / v0 if v0 != 0 else np.zeros(n, dtype=np.float64)

        eigs = np.array(
            [[float(r.get("eig_1", 1.0)), float(r.get("eig_2", 1.0)), float(r.get("eig_3", 1.0))] for r in obs],
            dtype=np.float64,
        )
        shape_eigs = np.array(
            [
                [
                    float(r.get("vox_eig_1", r.get("eig_1", 1.0))),
                    float(r.get("vox_eig_2", r.get("eig_2", 1.0))),
                    float(r.get("vox_eig_3", r.get("eig_3", 1.0))),
                ]
                for r in obs
            ],
            dtype=np.float64,
        )
        e3 = shape_eigs[:, 2]
        anisotropy = np.ones(n, dtype=np.float64)
        np.divide(shape_eigs[:, 0], e3, out=anisotropy, where=e3 > 1e-6)
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
                delta_d=delta_d,
                centroid_mu_err=mu_err,
                centroid_chi_err=chi_err,
                centroid_d_err=d_err,
                strain_err=strain_err,
                root_track_id=root_tid,
                lineage_strain=lineage_strain,
                lineage_strain_err=lineage_strain_err,
                rate_mu=rate_mu,
                rate_chi=rate_chi,
                rate_d=rate_d,
                strain_rate=strain_rate,
                strain_rate_per_s=strain_rate_per_s,
            )
        )
    return results


def summarize_kinematics(
    grain_kinematics: list[GrainKinematics],
    n_scans: int,
    *,
    calibrated_axes: bool = False,
) -> KinematicsSummary:
    """Aggregate per-grain kinematics into a summary table."""
    track_ids: list[int] = []
    max_strain: list[float] = []
    max_lineage_strain: list[float] = []
    total_misorientation: list[float] = []
    intensity_change: list[float] = []
    volume_change: list[float] = []
    max_anisotropy: list[float] = []

    for gk in grain_kinematics:
        track_ids.append(gk.track_id)
        max_strain.append(float(np.max(np.abs(gk.strain))) if len(gk.strain) > 0 else 0.0)
        lin_s = gk.lineage_strain if len(gk.lineage_strain) > 0 else gk.strain
        max_lineage_strain.append(float(np.max(np.abs(lin_s))) if len(lin_s) > 0 else 0.0)
        total_misorientation.append(float(gk.misorientation_total[-1]) if len(gk.misorientation_total) > 0 else 0.0)
        intensity_change.append(float(gk.relative_intensity[-1]) if len(gk.relative_intensity) > 0 else 0.0)
        volume_change.append(float(gk.relative_volume[-1]) if len(gk.relative_volume) > 0 else 0.0)
        max_anisotropy.append(float(np.max(gk.anisotropy)) if len(gk.anisotropy) > 0 else 1.0)

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
        calibrated_axes=calibrated_axes,
        max_lineage_strain=max_lineage_strain,
        grain_kinematics=grain_kinematics,
    )


def summary_to_dict(summary: KinematicsSummary) -> dict[str, Any]:
    """Serialize :class:`KinematicsSummary` to a JSON-friendly dictionary."""
    return {
        "n_tracks": summary.n_tracks,
        "n_full_tracks": summary.n_full_tracks,
        "n_scans": summary.n_scans,
        "calibrated_axes": summary.calibrated_axes,
        "track_ids": summary.track_ids,
        "max_strain": summary.max_strain,
        "max_lineage_strain": summary.max_lineage_strain,
        "total_misorientation": summary.total_misorientation,
        "intensity_change_frac": summary.intensity_change_frac,
        "volume_change_frac": summary.volume_change_frac,
        "max_anisotropy": summary.max_anisotropy,
    }


def kinematics_to_table(
    grain_kinematics: list[GrainKinematics],
    *,
    include_delta_d: bool = False,
) -> list[dict[str, Any]]:
    """Flatten per-grain kinematics into a row-per-observation table."""
    rows: list[dict[str, Any]] = []
    for gk in grain_kinematics:
        root_tid = gk.root_track_id if gk.root_track_id > 0 else gk.track_id
        for i, scan_idx in enumerate(gk.scan_indices):
            row: dict[str, Any] = {
                "track_id": gk.track_id,
                "root_track_id": root_tid,
                "scan_idx": scan_idx,
                "centroid_mu": float(gk.centroid_mu[i]),
                "centroid_chi": float(gk.centroid_chi[i]),
                "centroid_d": float(gk.centroid_d[i]),
                "centroid_mu_err": float(gk.centroid_mu_err[i]) if len(gk.centroid_mu_err) > i else 0.0,
                "centroid_chi_err": float(gk.centroid_chi_err[i]) if len(gk.centroid_chi_err) > i else 0.0,
                "centroid_d_err": float(gk.centroid_d_err[i]) if len(gk.centroid_d_err) > i else 0.0,
                "strain": float(gk.strain[i]),
                "strain_err": float(gk.strain_err[i]) if len(gk.strain_err) > i else 0.0,
                "lineage_strain": float(gk.lineage_strain[i]) if len(gk.lineage_strain) > i else float(gk.strain[i]),
                "lineage_strain_err": (float(gk.lineage_strain_err[i]) if len(gk.lineage_strain_err) > i else 0.0),
                "strain_rate": float(gk.strain_rate[i]) if len(gk.strain_rate) > i else 0.0,
                "rate_mu": float(gk.rate_mu[i]) if len(gk.rate_mu) > i else 0.0,
                "rate_chi": float(gk.rate_chi[i]) if len(gk.rate_chi) > i else 0.0,
                "rate_d": float(gk.rate_d[i]) if len(gk.rate_d) > i else 0.0,
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
            if include_delta_d and len(gk.delta_d) > i:
                row["delta_d"] = float(gk.delta_d[i])
            rows.append(row)
    return rows
