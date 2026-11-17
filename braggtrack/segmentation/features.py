"""Feature extraction for segmented 3D instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import find_objects

if TYPE_CHECKING:
    from braggtrack.io.models import AxisCalibration


def extract_instance_table(
    labels: np.ndarray,
    intensity: np.ndarray,
    *,
    calibration: AxisCalibration | None = None,
) -> list[dict[str, float | int]]:
    """Compute centroid, bbox, integrated intensity, covariance and eigenvalues.

    Voxel array axes are ``(z, y, x)`` mapping to reciprocal-space coordinates
    as **μ → z**, **d → y**, **χ → x** (STFC / operando convention).

    When *calibration* is provided, centroids, covariance components, and
    eigenvalues are transformed into physical units while ``bbox_*`` bounds
    remain integer voxel indices for 3D array slicing.
    """
    labels = np.asarray(labels)
    intensity = np.asarray(intensity, dtype=np.float64)
    if labels.size == 0 or int(labels.max()) <= 0:
        return []

    slices = find_objects(np.where(labels > 0, labels, 0))
    rows: list[dict[str, float | int]] = []
    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        lbl = idx + 1
        sz, sy, sx = slc
        sub_lbl = labels[sz, sy, sx]
        mask = sub_lbl == lbl
        if not np.any(mask):
            continue
        local_coords = np.argwhere(mask)
        vals = intensity[sz, sy, sx][mask]
        count = len(vals)
        sum_i = float(vals.sum())

        z_coords = local_coords[:, 0].astype(np.float64) + float(sz.start)
        y_coords = local_coords[:, 1].astype(np.float64) + float(sy.start)
        x_coords = local_coords[:, 2].astype(np.float64) + float(sx.start)

        # Intensity-weighted centroid; fall back to geometric centroid
        # if total intensity is non-positive (avoids sign-flip).
        if sum_i > 0:
            weights = vals / sum_i
            cz = float(np.dot(weights, z_coords))
            cy = float(np.dot(weights, y_coords))
            cx = float(np.dot(weights, x_coords))
        else:
            cz = float(z_coords.mean())
            cy = float(y_coords.mean())
            cx = float(x_coords.mean())

        if calibration is not None:
            c_mu, c_chi, c_d = calibration.voxel_to_physical(cz, cy, cx)
            dz = (z_coords - cz) * calibration.mu_step
            dy = (y_coords - cy) * calibration.d_step
            dx = (x_coords - cx) * calibration.chi_step
            step_mu = abs(calibration.mu_step)
            step_d = abs(calibration.d_step)
            step_chi = abs(calibration.chi_step)
        else:
            c_mu, c_chi, c_d = cz, cx, cy
            dz = z_coords - cz
            dy = y_coords - cy
            dx = x_coords - cx
            step_mu = 1.0
            step_d = 1.0
            step_chi = 1.0

        # Weighted covariance (same weight logic)
        if sum_i > 0:
            w = vals / sum_i
        else:
            w = np.ones(count, dtype=np.float64) / count

        czz = float(np.dot(w, dz * dz))
        cyy = float(np.dot(w, dy * dy))
        cxx = float(np.dot(w, dx * dx))
        czy = float(np.dot(w, dz * dy))
        czx = float(np.dot(w, dz * dx))
        cyx = float(np.dot(w, dy * dx))

        # Analytical 1-sigma centroid standard error: under Poisson photon statistics
        # Var(c) = cov / sum_i; when intensities are normalized (< n_eff), fall back to
        # Kish effective sample size n_eff = 1 / sum(w_k^2), floored by voxel quantization.
        w_sq = float(np.dot(w, w))
        n_eff = (1.0 / w_sq) if w_sq > 0.0 else float(max(1, count))
        eff_denom = max(sum_i, n_eff, 1.0)
        c_mu_err = float(np.sqrt(max(czz, (step_mu**2) / 12.0) / eff_denom))
        c_chi_err = float(np.sqrt(max(cxx, (step_chi**2) / 12.0) / eff_denom))
        c_d_err = float(np.sqrt(max(cyy, (step_d**2) / 12.0) / eff_denom))

        cov = np.array(
            [
                [czz, czy, czx],
                [czy, cyy, cyx],
                [czx, cyx, cxx],
            ]
        )
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(0.0, np.sort(eigvals)[::-1])  # non-negative descending

        rows.append(
            {
                "label": int(lbl),
                "voxel_count": count,
                "integrated_intensity": sum_i,
                "centroid_mu": c_mu,
                "centroid_chi": c_chi,
                "centroid_d": c_d,
                "centroid_mu_err": c_mu_err,
                "centroid_chi_err": c_chi_err,
                "centroid_d_err": c_d_err,
                "bbox_min_z": int(z_coords.min()),
                "bbox_max_z": int(z_coords.max()),
                "bbox_min_y": int(y_coords.min()),
                "bbox_max_y": int(y_coords.max()),
                "bbox_min_x": int(x_coords.min()),
                "bbox_max_x": int(x_coords.max()),
                "cov_zz": czz,
                "cov_yy": cyy,
                "cov_xx": cxx,
                "cov_zy": czy,
                "cov_zx": czx,
                "cov_yx": cyx,
                "eig_1": float(eigvals[0]),
                "eig_2": float(eigvals[1]),
                "eig_3": float(eigvals[2]),
            }
        )

    return rows
