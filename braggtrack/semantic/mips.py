"""Orthogonal maximum-intensity projections for 2.5D multi-view descriptors.

Volume layout is ``(z, y, x)`` with **μ → z**, **d → y**, **χ → x**.
"""

from __future__ import annotations

import numpy as np


def crop_spot_cube(
    volume: np.ndarray,
    labels: np.ndarray,
    label_id: int,
    bbox: dict[str, int],
    margin: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a padded sub-volume and binary mask for one instance.

    ``bbox`` uses keys ``bbox_min_z`` … ``bbox_max_x`` (array indices).
    """
    z0 = max(0, int(bbox["bbox_min_z"]) - margin)
    z1 = min(volume.shape[0], int(bbox["bbox_max_z"]) + margin + 1)
    y0 = max(0, int(bbox["bbox_min_y"]) - margin)
    y1 = min(volume.shape[1], int(bbox["bbox_max_y"]) + margin + 1)
    x0 = max(0, int(bbox["bbox_min_x"]) - margin)
    x1 = min(volume.shape[2], int(bbox["bbox_max_x"]) + margin + 1)

    sub_vol = volume[z0:z1, y0:y1, x0:x1].astype(np.float64, copy=False)
    sub_lab = labels[z0:z1, y0:y1, x0:x1]
    mask = (sub_lab == label_id).astype(np.float64)
    masked = sub_vol * mask
    return masked, mask


def orthogonal_mips(masked_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three MIPs: collapse μ, χ, and d respectively.

    Returns
    -------
    mip_mu, mip_chi, mip_d
        * ``mip_mu`` — max along **μ** (axis 0): plane **(d, χ)** shape ``(ny, nx)``.
        * ``mip_chi`` — max along **χ** (axis 2): plane **(μ, d)** shape ``(nz, ny)``.
        * ``mip_d`` — max along **d** (axis 1): plane **(μ, χ)** shape ``(nz, nx)``.
    """
    v = np.asarray(masked_volume, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError("masked_volume must be 3-D")
    mip_mu = np.max(v, axis=0)
    mip_chi = np.max(v, axis=2)
    mip_d = np.max(v, axis=1)
    return mip_mu, mip_chi, mip_d
