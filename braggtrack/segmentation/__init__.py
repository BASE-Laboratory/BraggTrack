"""Segmentation utilities for Week 2 baselines."""

from .classical import (
    ClassicalSegmentationResult,
    gaussian_blur_3d,
    h_maxima_seeds,
    local_maxima_seeds,
    log_enhance_3d,
    segment_classical,
    watershed_from_seeds,
)
from .features import extract_instance_table
from .otsu import otsu_threshold
from .pipeline import SegmentationResult, connected_components_3d, segment_volume
from .postprocess import fill_holes_binary, relabel_sequential, remove_small_objects
from .projection import label_projection_by_intensity, otsu_floor_from_mip

__all__ = [
    "ClassicalSegmentationResult",
    "SegmentationResult",
    "connected_components_3d",
    "extract_instance_table",
    "fill_holes_binary",
    "gaussian_blur_3d",
    "h_maxima_seeds",
    "label_projection_by_intensity",
    "local_maxima_seeds",
    "log_enhance_3d",
    "otsu_floor_from_mip",
    "otsu_threshold",
    "relabel_sequential",
    "remove_small_objects",
    "segment_classical",
    "segment_volume",
    "watershed_from_seeds",
]
