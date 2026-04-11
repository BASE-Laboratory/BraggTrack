"""Data I/O helpers for BraggTrack."""

from .beamline import BeamlineAdapter
from .discovery import discover_operando_scans
from .models import AxisSpec, ExperimentSequence, ScanVolumeMeta
from .paths import default_dataset_root, resolve_dataset_root, sample_operando_root
from .nexus import MissingH5DependencyError, extract_scan_metadata, load_primary_volume, summarize_hdf5_tree
from .validation import ValidationIssue, validate_sequence

__all__ = [
    "default_dataset_root",
    "resolve_dataset_root",
    "sample_operando_root",
    "discover_operando_scans",
    "extract_scan_metadata",
    "summarize_hdf5_tree",
    "load_primary_volume",
    "MissingH5DependencyError",
    "AxisSpec",
    "ScanVolumeMeta",
    "ExperimentSequence",
    "BeamlineAdapter",
    "ValidationIssue",
    "validate_sequence",
]
