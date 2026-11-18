"""Shared type definitions for the BraggTrack pipeline."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class SpotRecord(TypedDict, total=False):
    """Feature record for one segmented Bragg spot.

    Required keys are produced by :func:`~braggtrack.segmentation.extract_instance_table`.
    Optional keys (``embedding``) are added by the semantic embedding stage.
    """

    # Required — segmentation stage
    label: int
    voxel_count: int
    integrated_intensity: float
    net_intensity: float
    background_level: float
    centroid_mu: float
    centroid_chi: float
    centroid_d: float
    centroid_mu_err: float
    centroid_chi_err: float
    centroid_d_err: float
    bbox_min_z: int
    bbox_max_z: int
    bbox_min_y: int
    bbox_max_y: int
    bbox_min_x: int
    bbox_max_x: int
    cov_zz: float
    cov_yy: float
    cov_xx: float
    cov_zy: float
    cov_zx: float
    cov_yx: float
    eig_1: float
    eig_2: float
    eig_3: float
    vox_eig_1: float
    vox_eig_2: float
    vox_eig_3: float
    kurtosis_mu: float
    kurtosis_chi: float
    kurtosis_d: float
    step_mu: float
    step_chi: float
    step_d: float
    calibrated: int
    touches_border: bool
    touches_rocking_border: bool

    # Optional — semantic stage
    embedding: np.ndarray
