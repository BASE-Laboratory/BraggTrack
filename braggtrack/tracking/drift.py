"""Collective rocking-slab drift registration and 6-D constant-velocity Kalman filtering.

Solves crowded-layer spatial aliasing in operando 3DXRD / near-field DFXM where rigid
sample drift or Debye-Scherrer ring motion (``12-25 px/scan``) equals or exceeds the
inter-reflection spacing (``12-20 px``) within a rocking band:

1. :func:`estimate_slab_shifts_fft`: Windowed 2D FFT phase cross-correlation on
   pedestal-subtracted rocking-band MIPs (``skimage.registration.phase_cross_correlation``),
   recovering coherent ``(d_mu, d_chi, d_d)`` translations even when adjacent spots
   coalesce in watershed segmentation.
2. :func:`estimate_band_shifts_consensus`: Spot-table-only morphology-weighted Parzen
   translation mode estimator across local rocking-layer clusters (zero raw volume I/O).
3. :class:`TrackKalmanState`: Decoupled 6-D constant-velocity Kalman filter
   ``[mu, chi, d, v_mu, v_chi, v_d]`` propagating analytical Poisson centroid variances
   ``centroid_*_err`` across frames once ``>= 2`` observations exist on a track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from skimage.registration import phase_cross_correlation

from .cost import _spot_intensity, _spot_scales, _spot_widths

DriftMode = Literal["none", "consensus", "fft", "fft_kalman"]


def is_single_slice_speck(spot: dict[str, Any], *, max_voxels: int = 15) -> bool:
    """Return True if ``spot`` is a single-rocking-frame cosmic-ray or hot-pixel artifact."""
    vox = int(spot.get("voxel_count", 999))
    if vox > max_voxels:
        return False
    e3 = float(spot.get("vox_eig_3", spot.get("eig_3", 1.0)))
    if "bbox_min_z" in spot and "bbox_max_z" in spot:
        z_span = int(spot["bbox_max_z"]) - int(spot["bbox_min_z"])
        return z_span == 0 and e3 < 1e-6
    return e3 < 1e-6


def _spot_step_pos(spot: dict[str, Any]) -> tuple[float, float, float]:
    sm, sc, sd = _spot_scales(spot)
    return (
        float(spot["centroid_mu"]) / sm,
        float(spot["centroid_chi"]) / sc,
        float(spot["centroid_d"]) / sd,
    )


def group_local_rocking_patches(
    spots: list[dict[str, Any]],
    *,
    band_gap_steps: float = 4.0,
    patch_radius_steps: float = 65.0,
    exclude_specks: bool = True,
) -> list[list[int]]:
    """Group spot indices into connected ``(mu, chi, d)`` rocking-band patches."""
    valid_indices = [idx for idx, s in enumerate(spots) if not (exclude_specks and is_single_slice_speck(s))]
    if not valid_indices:
        return []

    coords = {idx: _spot_step_pos(spots[idx]) for idx in valid_indices}
    visited: set[int] = set()
    patches: list[list[int]] = []

    for start in valid_indices:
        if start in visited:
            continue
        comp: list[int] = []
        queue = [start]
        visited.add(start)
        while queue:
            cur = queue.pop()
            comp.append(cur)
            mu_a, chi_a, d_a = coords[cur]
            for other in valid_indices:
                if other in visited:
                    continue
                mu_b, chi_b, d_b = coords[other]
                if abs(mu_a - mu_b) <= band_gap_steps and math.hypot(chi_a - chi_b, d_a - d_b) <= patch_radius_steps:
                    visited.add(other)
                    queue.append(other)
        patches.append(sorted(comp))
    return patches


def _robust_pedestal(arr: np.ndarray, bg_sigma: float = 4.0) -> float:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    if flat.size > 65_536:
        stride = max(1, flat.size // 65_536)
        flat = flat[::stride]
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    return med + float(bg_sigma) * 1.4826 * mad


def _normalized_overlap_after_shift(ref: np.ndarray, mov: np.ndarray, dy: float, dx: float) -> float:
    """Fast integer-shifted cosine overlap quality check in ``[0, 1]``."""
    h, w = ref.shape
    iy = int(round(dy))
    ix = int(round(dx))
    if abs(iy) >= h or abs(ix) >= w:
        return 0.0
    r_y0, r_y1 = max(0, iy), min(h, h + iy)
    m_y0, m_y1 = max(0, -iy), min(h, h - iy)
    r_x0, r_x1 = max(0, ix), min(w, w + ix)
    m_x0, m_x1 = max(0, -ix), min(w, w - ix)
    sub_r = ref[r_y0:r_y1, r_x0:r_x1]
    sub_m = mov[m_y0:m_y1, m_x0:m_x1]
    nr = float(np.linalg.norm(sub_r))
    nm = float(np.linalg.norm(sub_m))
    if nr <= 1e-9 or nm <= 1e-9:
        return 0.0
    return float(np.clip(np.sum(sub_r * sub_m) / (nr * nm), 0.0, 1.0))


def estimate_slab_shifts_fft(
    vol_t: np.ndarray,
    vol_t1: np.ndarray,
    spots_t: list[dict[str, Any]],
    *,
    band_gap_frames: float = 4.0,
    patch_radius_px: float = 65.0,
    pad_z: int = 2,
    pad_yx: int = 45,
    max_step_chi: float = 40.0,
    max_step_d: float = 40.0,
    upsample_factor: int = 4,
    min_overlap_cosine: float = 0.15,
    calibration: Any | None = None,
) -> tuple[list[tuple[float, float, float]], list[dict[str, Any]]]:
    """Estimate per-spot ``(d_mu, d_chi, d_d)`` drift via windowed 2D FFT cross-correlation.

    Groups reflections in ``spots_t`` into local rocking-band patches, projects pedestal-
    subtracted ``sqrt``-compressed 2D MIPs over ``[z0, z1]``, and registers ``vol_t`` onto
    ``vol_t1`` with sub-pixel FFT cross-correlation.

    Returns
    -------
    spot_shifts
        List of ``(d_mu, d_chi, d_d)`` shifts in native spot units (length ``len(spots_t)``).
    patch_records
        JSON-serializable summary dicts per rocking patch (for ``braggtrack-eval-archive`` caching).
    """
    n_spots = len(spots_t)
    shifts: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_spots
    if n_spots == 0:
        return shifts, []

    v0 = np.asarray(vol_t)
    v1 = np.asarray(vol_t1)
    if v0.ndim != 3 or v0.shape != v1.shape:
        return shifts, []

    nz, ny, nx = v0.shape
    patches = group_local_rocking_patches(
        spots_t,
        band_gap_steps=band_gap_frames,
        patch_radius_steps=patch_radius_px,
        exclude_specks=True,
    )
    records: list[dict[str, Any]] = []

    for p_idx, idxs in enumerate(patches):
        z_vals: list[int] = []
        y_vals: list[int] = []
        x_vals: list[int] = []
        for i in idxs:
            s = spots_t[i]
            sm, sc, sd = _spot_scales(s)
            if "bbox_min_z" in s and "bbox_max_z" in s:
                z_vals.extend([int(s["bbox_min_z"]), int(s["bbox_max_z"])])
                y_vals.extend([int(s["bbox_min_y"]), int(s["bbox_max_y"])])
                x_vals.extend([int(s["bbox_min_x"]), int(s["bbox_max_x"])])
            else:
                cz = int(round(float(s["centroid_mu"]) / sm))
                cx = int(round(float(s["centroid_chi"]) / sc))
                cy = int(round(float(s["centroid_d"]) / sd))
                z_vals.append(cz)
                y_vals.append(cy)
                x_vals.append(cx)

        z0 = max(0, min(z_vals) - int(pad_z))
        z1 = min(nz, max(z_vals) + int(pad_z) + 1)
        y0 = max(0, min(y_vals) - int(pad_yx))
        y1 = min(ny, max(y_vals) + int(pad_yx) + 1)
        x0 = max(0, min(x_vals) - int(pad_yx))
        x1 = min(nx, max(x_vals) + int(pad_yx) + 1)
        if z1 <= z0 or (y1 - y0) < 8 or (x1 - x0) < 8:
            continue

        slab0 = v0[z0:z1, y0:y1, x0:x1]
        slab1 = v1[z0:z1, y0:y1, x0:x1]
        ped0 = _robust_pedestal(slab0, bg_sigma=4.0)
        ped1 = _robust_pedestal(slab1, bg_sigma=4.0)
        mip0 = np.sqrt(np.maximum(np.max(slab0, axis=0).astype(np.float64) - ped0, 0.0))
        mip1 = np.sqrt(np.maximum(np.max(slab1, axis=0).astype(np.float64) - ped1, 0.0))

        if float(mip0.max()) <= 1e-9 or float(mip1.max()) <= 1e-9:
            continue

        # phase_cross_correlation(reference=mip1, moving=mip0) returns (dy, dx) such that
        # shifting mip0 by (+dy, +dx) aligns it with mip1, i.e. spot_t1 ~= spot_t + (dy, dx).
        shift_yx, _, _ = phase_cross_correlation(
            mip1,
            mip0,
            upsample_factor=max(1, int(upsample_factor)),
            normalization=None,
        )
        dy_px = float(shift_yx[0])
        dx_px = float(shift_yx[1])
        overlap = _normalized_overlap_after_shift(mip1, mip0, dy_px, dx_px)

        if abs(dx_px) > max_step_chi or abs(dy_px) > max_step_d or overlap < min_overlap_cosine:
            dy_px = 0.0
            dx_px = 0.0

        for i in idxs:
            _, sc_i, sd_i = _spot_scales(spots_t[i])
            if calibration is not None and bool(spots_t[i].get("calibrated", 0)):
                sc_signed = float(calibration.chi_step)
                sd_signed = float(calibration.d_step)
            else:
                sc_signed = sc_i
                sd_signed = sd_i
            shifts[i] = (0.0, dx_px * sc_signed, dy_px * sd_signed)

        records.append(
            {
                "patch_id": p_idx,
                "spot_indices": idxs,
                "bbox_zyx": [z0, z1, y0, y1, x0, x1],
                "shift_d_px": round(dy_px, 4),
                "shift_chi_px": round(dx_px, 4),
                "overlap_cosine": round(overlap, 4),
            }
        )

    return shifts, records


def estimate_band_shifts_consensus(
    spots_t: list[dict[str, Any]],
    spots_t1: list[dict[str, Any]],
    *,
    band_gap_steps: float = 4.0,
    patch_radius_steps: float = 65.0,
    max_step_mu: float = 6.0,
    max_step_chi: float = 40.0,
    max_step_d: float = 40.0,
    kernel_sigma_steps: float = 3.5,
    min_patch_spots: int = 2,
) -> list[tuple[float, float, float]]:
    """Estimate collective ``(d_mu, d_chi, d_d)`` drift purely from consecutive spot tables.

    For each co-moving cluster (``len >= min_patch_spots``) in ``spots_t``, evaluates all
    admissible candidate displacements to ``spots_t1`` weighted by intrinsic width and
    log-intensity similarity, then locates the Parzen Gaussian kernel density mode.
    """
    n_t = len(spots_t)
    shifts: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_t
    if n_t == 0 or not spots_t1:
        return shifts

    valid_t1 = [j for j, s in enumerate(spots_t1) if not is_single_slice_speck(s)]
    if not valid_t1:
        return shifts

    pos_t = [_spot_step_pos(s) for s in spots_t]
    pos_t1 = [_spot_step_pos(s) for s in spots_t1]
    widths_t = [_spot_widths(s, normalize_by_step=True) for s in spots_t]
    widths_t1 = [_spot_widths(s, normalize_by_step=True) for s in spots_t1]
    log_i_t = [math.log(_spot_intensity(s)) for s in spots_t]
    log_i_t1 = [math.log(_spot_intensity(s)) for s in spots_t1]

    patches = group_local_rocking_patches(
        spots_t,
        band_gap_steps=band_gap_steps,
        patch_radius_steps=patch_radius_steps,
        exclude_specks=True,
    )

    bw2 = max(float(kernel_sigma_steps) ** 2, 0.25)
    for idxs in patches:
        if len(idxs) < max(1, int(min_patch_spots)):
            continue

        disp_list: list[tuple[float, float, float]] = []
        weight_list: list[float] = []
        src_list: list[int] = []
        for i in idxs:
            mu_i, chi_i, d_i = pos_t[i]
            wi = widths_t[i]
            li = log_i_t[i]
            for j in valid_t1:
                mu_j, chi_j, d_j = pos_t1[j]
                dmu = mu_j - mu_i
                dchi = chi_j - chi_i
                dd = d_j - d_i
                if abs(dmu) > max_step_mu or abs(dchi) > max_step_chi or abs(dd) > max_step_d:
                    continue
                wj = widths_t1[j]
                shape_d2 = (wi[0] - wj[0]) ** 2 + (wi[1] - wj[1]) ** 2 + (wi[2] - wj[2]) ** 2
                int_d2 = (li - log_i_t1[j]) ** 2
                w_ij = math.exp(-0.25 * shape_d2 - 2.0 * int_d2)
                disp_list.append((dmu, dchi, dd))
                weight_list.append(w_ij)
                src_list.append(i)

        if not disp_list:
            continue

        disp_arr = np.asarray(disp_list, dtype=np.float64)
        w_arr = np.asarray(weight_list, dtype=np.float64)
        # Parzen density at each candidate displacement, requiring support across distinct source spots
        diff2 = np.sum((disp_arr[:, None, :] - disp_arr[None, :, :]) ** 2, axis=-1)
        kern = np.exp(-0.5 * diff2 / bw2) * w_arr[None, :]
        # Zero self-source inflation when multiple spots exist so 1 outlier pair can't dominate
        if len(idxs) >= 2:
            src_arr = np.asarray(src_list, dtype=np.int64)
            same_src = src_arr[:, None] == src_arr[None, :]
            # Keep best per source spot via soft downweighting of same-source duplicates
            kern = np.where(same_src, 0.35 * kern, kern)
        density = np.sum(kern, axis=1)
        best_k = int(np.argmax(density))

        # Locally refine around mode via kernel-weighted average of consistent neighbours
        in_peak = diff2[best_k] <= (2.25 * bw2)
        w_peak = kern[best_k, in_peak]
        w_sum = float(np.sum(w_peak))
        if w_sum <= 1e-12:
            continue
        mode_step = np.sum(disp_arr[in_peak] * (w_peak[:, None] / w_sum), axis=0)

        # Mild shrinkage for 2-spot pairs; near-unity for >=3 co-moving spots
        shrink = float(len(idxs)) / (float(len(idxs)) + 0.5)
        dmu_s, dchi_s, dd_s = (
            float(mode_step[0]) * shrink,
            float(mode_step[1]) * shrink,
            float(mode_step[2]) * shrink,
        )
        for i in idxs:
            sm_i, sc_i, sd_i = _spot_scales(spots_t[i])
            shifts[i] = (dmu_s * sm_i, dchi_s * sc_i, dd_s * sd_i)

    return shifts


@dataclass
class TrackKalmanState:
    """Decoupled 6-D constant-velocity Kalman filter ``[mu, chi, d, v_mu, v_chi, v_d]``."""

    pos: np.ndarray  # shape (3,) -> (mu, chi, d)
    vel: np.ndarray  # shape (3,) -> (v_mu, v_chi, v_d) per unit dt
    cov: np.ndarray  # shape (3, 2, 2) per-axis [pos, vel] covariance
    age: int = 1
    last_time: float = 0.0

    @classmethod
    def initialize(
        cls,
        spot: dict[str, Any],
        *,
        time_val: float = 0.0,
        init_vel_std_steps: float = 6.0,
    ) -> TrackKalmanState:
        sm, sc, sd = _spot_scales(spot)
        pos = np.array(
            [float(spot["centroid_mu"]), float(spot["centroid_chi"]), float(spot["centroid_d"])],
            dtype=np.float64,
        )
        vel = np.zeros(3, dtype=np.float64)
        errs = [
            max(float(spot.get("centroid_mu_err", 0.25 * sm)), 0.1 * sm),
            max(float(spot.get("centroid_chi_err", 0.25 * sc)), 0.1 * sc),
            max(float(spot.get("centroid_d_err", 0.25 * sd)), 0.1 * sd),
        ]
        v_stds = [init_vel_std_steps * sm, init_vel_std_steps * sc, init_vel_std_steps * sd]
        cov = np.zeros((3, 2, 2), dtype=np.float64)
        for a in range(3):
            cov[a, 0, 0] = errs[a] ** 2
            cov[a, 1, 1] = v_stds[a] ** 2
        return cls(pos=pos, vel=vel, cov=cov, age=1, last_time=float(time_val))

    def predict(
        self,
        dt: float = 1.0,
        *,
        accel_std_steps: tuple[float, float, float] = (0.6, 2.5, 2.5),
        scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return predicted ``(mean_pos, std_pos)`` after step ``dt`` without mutating state."""
        dt_f = max(float(dt), 1e-6)
        F = np.array([[1.0, dt_f], [0.0, 1.0]], dtype=np.float64)
        pred_pos = self.pos + self.vel * dt_f
        pred_std = np.zeros(3, dtype=np.float64)
        for a in range(3):
            sig_a = float(accel_std_steps[a]) * float(scales[a])
            q_mat = (sig_a**2) * np.array(
                [[0.25 * dt_f**4, 0.5 * dt_f**3], [0.5 * dt_f**3, dt_f**2]],
                dtype=np.float64,
            )
            p_pred = F @ self.cov[a] @ F.T + q_mat
            pred_std[a] = math.sqrt(max(float(p_pred[0, 0]), 1e-12))
        return (
            (float(pred_pos[0]), float(pred_pos[1]), float(pred_pos[2])),
            (float(pred_std[0]), float(pred_std[1]), float(pred_std[2])),
        )

    def update(
        self,
        spot: dict[str, Any],
        *,
        time_val: float | None = None,
        accel_std_steps: tuple[float, float, float] = (0.6, 2.5, 2.5),
    ) -> None:
        """Advance state to ``time_val`` and fuse observed ``spot`` centroid and error."""
        sm, sc, sd = _spot_scales(spot)
        scales = (sm, sc, sd)
        t_now = (self.last_time + 1.0) if time_val is None else float(time_val)
        dt_f = max(t_now - self.last_time, 1e-6)
        F = np.array([[1.0, dt_f], [0.0, 1.0]], dtype=np.float64)
        obs = [float(spot["centroid_mu"]), float(spot["centroid_chi"]), float(spot["centroid_d"])]
        obs_errs = [
            max(float(spot.get("centroid_mu_err", 0.25 * sm)), 0.1 * sm),
            max(float(spot.get("centroid_chi_err", 0.25 * sc)), 0.1 * sc),
            max(float(spot.get("centroid_d_err", 0.25 * sd)), 0.1 * sd),
        ]

        for a in range(3):
            sig_a = float(accel_std_steps[a]) * scales[a]
            q_mat = (sig_a**2) * np.array(
                [[0.25 * dt_f**4, 0.5 * dt_f**3], [0.5 * dt_f**3, dt_f**2]],
                dtype=np.float64,
            )
            x_pred = F @ np.array([self.pos[a], self.vel[a]], dtype=np.float64)
            p_pred = F @ self.cov[a] @ F.T + q_mat
            r_var = obs_errs[a] ** 2
            s_inn = float(p_pred[0, 0]) + r_var
            k_gain = p_pred[:, 0] / max(s_inn, 1e-18)
            innov = obs[a] - float(x_pred[0])
            x_new = x_pred + k_gain * innov
            # Joseph stabilized covariance update (I - K H) P (I - K H)^T + K R K^T
            ikh = np.array([[1.0 - k_gain[0], 0.0], [-k_gain[1], 1.0]], dtype=np.float64)
            p_new = ikh @ p_pred @ ikh.T + np.outer(k_gain, k_gain) * r_var
            self.pos[a] = float(x_new[0])
            self.vel[a] = float(x_new[1])
            self.cov[a] = 0.5 * (p_new + p_new.T)

        self.age += 1
        self.last_time = t_now


def apply_predicted_shifts(
    spots_t: list[dict[str, Any]],
    shifts_t: list[tuple[float, float, float]] | None,
) -> list[dict[str, Any]]:
    """Return shallow-copied ``spots_t`` with centroids translated by ``shifts_t``."""
    if not shifts_t:
        return spots_t
    shifted: list[dict[str, Any]] = []
    for s, (dmu, dchi, dd) in zip(spots_t, shifts_t):
        if dmu == 0.0 and dchi == 0.0 and dd == 0.0:
            shifted.append(s)
        else:
            cp = dict(s)
            cp["centroid_mu"] = float(s["centroid_mu"]) + float(dmu)
            cp["centroid_chi"] = float(s["centroid_chi"]) + float(dchi)
            cp["centroid_d"] = float(s["centroid_d"]) + float(dd)
            shifted.append(cp)
    return shifted
