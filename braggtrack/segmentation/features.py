"""Feature extraction for segmented 3D instances."""

from __future__ import annotations

import numpy as np


def extract_instance_table(
    labels: np.ndarray,
    intensity: np.ndarray,
) -> list[dict[str, float | int]]:
    """Compute centroid, bbox, integrated intensity, covariance and eigenvalues.

    Voxel array axes are ``(z, y, x)`` mapping to reciprocal-space coordinates
    as **μ → z**, **d → y**, **χ → x** (STFC / operando convention).

    Intensity-weighted centroids guard against negative or zero total
    intensity by falling back to the geometric (unweighted) centroid.
    """
    labels = np.asarray(labels)
    intensity = np.asarray(intensity, dtype=np.float64)

    rows: list[dict[str, float | int]] = []
    for lbl in np.unique(labels):
        if lbl <= 0:
            continue
        mask = labels == lbl
        coords = np.argwhere(mask)  # (N, 3) array of (z, y, x)
        vals = intensity[mask]
        count = len(vals)
        sum_i = float(vals.sum())

        z_coords = coords[:, 0].astype(np.float64)
        y_coords = coords[:, 1].astype(np.float64)
        x_coords = coords[:, 2].astype(np.float64)

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

        # Weighted covariance (same weight logic)
        dz = z_coords - cz
        dy = y_coords - cy
        dx = x_coords - cx

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

        cov = np.array([
            [czz, czy, czx],
            [czy, cyy, cyx],
            [czx, cyx, cxx],
        ])
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # descending

        rows.append(
            {
                "label": int(lbl),
                "voxel_count": count,
                "integrated_intensity": sum_i,
                "centroid_mu": cz,
                "centroid_chi": cx,
                "centroid_d": cy,
                "bbox_min_z": int(coords[:, 0].min()),
                "bbox_max_z": int(coords[:, 0].max()),
                "bbox_min_y": int(coords[:, 1].min()),
                "bbox_max_y": int(coords[:, 1].max()),
                "bbox_min_x": int(coords[:, 2].min()),
                "bbox_max_x": int(coords[:, 2].max()),
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
