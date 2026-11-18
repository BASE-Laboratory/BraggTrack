"""Feature extraction for segmented 3D instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import find_objects

if TYPE_CHECKING:
    from braggtrack.io.models import AxisCalibration


def _excess_kurtosis(delta: np.ndarray, weights: np.ndarray) -> float:
    """Standardised fourth central moment (Pearson excess kurtosis ``g_2 = mu_4 / mu_2^2 - 3``).

    Zero for an ideal Gaussian rocking profile; positive (leptokurtic) for heavy-tailed
    dislocation pile-ups/notches and negative (platykurtic) for incipient bimodal splitting.
    """
    m2 = float(np.dot(weights, delta * delta))
    if m2 <= 1e-12:
        return 0.0
    m4 = float(np.dot(weights, (delta * delta) ** 2))
    return float(m4 / (m2 * m2) - 3.0)


def extract_instance_table(
    labels: np.ndarray,
    intensity: np.ndarray,
    *,
    calibration: AxisCalibration | None = None,
    subtract_background: bool = True,
    detector_gain: float = 1.0,
) -> list[dict[str, float | int]]:
    """Compute centroid, bbox, integrated intensity, covariance, kurtosis, and eigenvalues.

    Voxel array axes are ``(z, y, x)`` mapping to reciprocal-space coordinates
    as **μ → z**, **d → y**, **χ → x** (STFC / operando convention).

    When *calibration* is provided, centroids, covariance components, and
    eigenvalues are transformed into physical units while ``bbox_*`` bounds
    remain integer voxel indices for 3D array slicing and ``vox_eig_*`` / ``step_*``
    preserve dimensionless step-normalised geometry for unit-invariant tracking.
    """
    labels = np.asarray(labels)
    intensity = np.asarray(intensity, dtype=np.float64)
    if labels.size == 0 or int(labels.max()) <= 0:
        return []

    nz, ny, nx = labels.shape
    gain = max(float(detector_gain), 1e-6)
    is_calibrated = int(calibration is not None and calibration.calibrated)
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

        # Estimate local detector pedestal from outer perimeter of 4-voxel padded window (labels == 0),
        # well outside the immediate Bragg flank so sub-Otsu rocking-curve wings are not subtracted.
        pz0, pz1 = max(0, sz.start - 4), min(nz, sz.stop + 4)
        py0, py1 = max(0, sy.start - 4), min(ny, sy.stop + 4)
        px0, px1 = max(0, sx.start - 4), min(nx, sx.stop + 4)
        lbl_pad = labels[pz0:pz1, py0:py1, px0:px1]
        vol_pad = intensity[pz0:pz1, py0:py1, px0:px1]
        outer_ring = np.ones(lbl_pad.shape, dtype=bool)
        if lbl_pad.shape[0] > 2 and lbl_pad.shape[1] > 2 and lbl_pad.shape[2] > 2:
            outer_ring[1:-1, 1:-1, 1:-1] = False
        bg_candidates = (lbl_pad == 0) & outer_ring
        if not np.any(bg_candidates):
            bg_candidates = lbl_pad == 0
        if np.any(bg_candidates):
            bg_level = max(0.0, float(np.median(vol_pad[bg_candidates])))
        else:
            bg_level = 0.0

        vals_net = np.maximum(vals - bg_level, 0.0)
        sum_net = float(vals_net.sum())
        w_vals = vals_net if (subtract_background and sum_net > 0.0) else vals
        w_sum = sum_net if (subtract_background and sum_net > 0.0) else sum_i

        z_coords = local_coords[:, 0].astype(np.float64) + float(sz.start)
        y_coords = local_coords[:, 1].astype(np.float64) + float(sy.start)
        x_coords = local_coords[:, 2].astype(np.float64) + float(sx.start)

        # Intensity-weighted centroid; fall back to geometric centroid
        # if total intensity is non-positive (avoids sign-flip).
        if w_sum > 0.0:
            w = w_vals / w_sum
            cz = float(np.dot(w, z_coords))
            cy = float(np.dot(w, y_coords))
            cx = float(np.dot(w, x_coords))
        else:
            w = np.ones(count, dtype=np.float64) / count
            cz = float(z_coords.mean())
            cy = float(y_coords.mean())
            cx = float(x_coords.mean())

        dz_vox = z_coords - cz
        dy_vox = y_coords - cy
        dx_vox = x_coords - cx

        if calibration is not None:
            c_mu, c_chi, c_d = calibration.voxel_to_physical(cz, cy, cx)
            dz = dz_vox * calibration.mu_step
            dy = dy_vox * calibration.d_step
            dx = dx_vox * calibration.chi_step
            step_mu = abs(calibration.mu_step)
            step_d = abs(calibration.d_step)
            step_chi = abs(calibration.chi_step)
        else:
            c_mu, c_chi, c_d = cz, cx, cy
            dz = dz_vox
            dy = dy_vox
            dx = dx_vox
            step_mu = 1.0
            step_d = 1.0
            step_chi = 1.0

        czz = float(np.dot(w, dz * dz))
        cyy = float(np.dot(w, dy * dy))
        cxx = float(np.dot(w, dx * dx))
        czy = float(np.dot(w, dz * dy))
        czx = float(np.dot(w, dz * dx))
        cyx = float(np.dot(w, dy * dx))

        # Analytical 1-sigma centroid standard error: under Poisson photon statistics
        # Var(c) = cov / N_photons (N_photons = w_sum / detector_gain); when intensities are
        # normalized (< n_eff), fall back to Kish effective sample size n_eff = 1 / sum(w_k^2),
        # floored by uniform voxel quantization variance step^2 / 12.
        w_sq = float(np.dot(w, w))
        n_eff = (1.0 / w_sq) if w_sq > 0.0 else float(max(1, count))
        eff_denom = max(w_sum / gain, n_eff, 1.0)
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

        if is_calibrated:
            v_czz = float(np.dot(w, dz_vox * dz_vox))
            v_cyy = float(np.dot(w, dy_vox * dy_vox))
            v_cxx = float(np.dot(w, dx_vox * dx_vox))
            v_czy = float(np.dot(w, dz_vox * dy_vox))
            v_czx = float(np.dot(w, dz_vox * dx_vox))
            v_cyx = float(np.dot(w, dy_vox * dx_vox))
            vox_cov = np.array(
                [
                    [v_czz, v_czy, v_czx],
                    [v_czy, v_cyy, v_cyx],
                    [v_czx, v_cyx, v_cxx],
                ]
            )
            vox_eigs = np.maximum(0.0, np.sort(np.linalg.eigvalsh(vox_cov))[::-1])
        else:
            vox_eigs = eigvals

        min_z, max_z = int(z_coords.min()), int(z_coords.max())
        min_y, max_y = int(y_coords.min()), int(y_coords.max())
        min_x, max_x = int(x_coords.min()), int(x_coords.max())
        touches_rocking = bool(min_z == 0 or max_z >= nz - 1)
        touches_any = bool(touches_rocking or min_y == 0 or max_y >= ny - 1 or min_x == 0 or max_x >= nx - 1)

        # Evaluate rocking-curve Pearson excess kurtosis g_2 = mu_4 / sigma^4 - 3 over a 4-voxel
        # padded local ROI (excluding foreign spot labels) so sub-Otsu diffuse scattering wings
        # (LeHoux2026) are integrated rather than hard-clipped at the watershed boundary.
        kz0, kz1 = max(0, min_z - 4), min(nz, max_z + 5)
        ky0, ky1 = max(0, min_y - 4), min(ny, max_y + 5)
        kx0, kx1 = max(0, min_x - 4), min(nx, max_x + 5)
        lbl_win = labels[kz0:kz1, ky0:ky1, kx0:kx1]
        vol_win = intensity[kz0:kz1, ky0:ky1, kx0:kx1]
        own_or_bg = (lbl_win == 0) | (lbl_win == lbl)
        net_win = np.where(own_or_bg, np.maximum(0.0, vol_win - bg_level), 0.0)
        net_win_sum = float(net_win.sum())
        if net_win_sum > 0.0:
            rz, ry, rx = np.nonzero(net_win > 0.0)
            w_kurt = net_win[rz, ry, rx] / net_win_sum
            cz_k = float(np.dot(w_kurt, rz + kz0))
            cy_k = float(np.dot(w_kurt, ry + ky0))
            cx_k = float(np.dot(w_kurt, rx + kx0))
            k_mu = _excess_kurtosis((rz + kz0) - cz_k, w_kurt)
            k_chi = _excess_kurtosis((rx + kx0) - cx_k, w_kurt)
            k_d = _excess_kurtosis((ry + ky0) - cy_k, w_kurt)
        else:
            k_mu = _excess_kurtosis(dz_vox, w)
            k_chi = _excess_kurtosis(dx_vox, w)
            k_d = _excess_kurtosis(dy_vox, w)

        rows.append(
            {
                "label": int(lbl),
                "voxel_count": count,
                "integrated_intensity": sum_i,
                "net_intensity": sum_net,
                "background_level": bg_level,
                "centroid_mu": c_mu,
                "centroid_chi": c_chi,
                "centroid_d": c_d,
                "centroid_mu_err": c_mu_err,
                "centroid_chi_err": c_chi_err,
                "centroid_d_err": c_d_err,
                "bbox_min_z": min_z,
                "bbox_max_z": max_z,
                "bbox_min_y": min_y,
                "bbox_max_y": max_y,
                "bbox_min_x": min_x,
                "bbox_max_x": max_x,
                "cov_zz": czz,
                "cov_yy": cyy,
                "cov_xx": cxx,
                "cov_zy": czy,
                "cov_zx": czx,
                "cov_yx": cyx,
                "eig_1": float(eigvals[0]),
                "eig_2": float(eigvals[1]),
                "eig_3": float(eigvals[2]),
                "vox_eig_1": float(vox_eigs[0]),
                "vox_eig_2": float(vox_eigs[1]),
                "vox_eig_3": float(vox_eigs[2]),
                "kurtosis_mu": k_mu,
                "kurtosis_chi": k_chi,
                "kurtosis_d": k_d,
                "step_mu": float(step_mu),
                "step_chi": float(step_chi),
                "step_d": float(step_d),
                "calibrated": is_calibrated,
                "touches_border": touches_any,
                "touches_rocking_border": touches_rocking,
            }
        )

    return rows
