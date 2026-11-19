"""Multi-scan 4D grain tracking, lineage DAGs, kinematics, and fracture precursors."""

from .assignment import associate_frames, compute_assignment_margins
from .cost import CostFunction, GeometrySemanticCost, PositionShapeCost, WassersteinBuresCost
from .drift import (
    DriftMode,
    TrackKalmanState,
    apply_predicted_shifts,
    estimate_band_shifts_consensus,
    estimate_slab_shifts_fft,
    group_local_rocking_patches,
    is_single_slice_speck,
)
from .events import LineageEvent, SplitMergeConfig, events_to_table, resolve_splits_and_merges
from .gap_closing import GapCloseRecord, GapClosingConfig, resolve_track_gaps
from .kinematics import (
    GrainKinematics,
    KinematicsSummary,
    compute_grain_kinematics,
    kinematics_to_table,
    summarize_kinematics,
    summary_to_dict,
)
from .lifecycle import TrackEvent, build_tracks, tracks_to_table
from .metrics import compute_tracking_metrics
from .precursors import (
    ObservationPrecursorScore,
    PrecursorCandidate,
    PrecursorChannelMetrics,
    PrecursorConfig,
    compute_fracture_precursors,
    evaluate_precursor_channels_roc,
    precursors_to_table,
)
from .synthetic import generate_crossing_scenario, generate_fracture_scenario

__all__ = [
    "CostFunction",
    "DriftMode",
    "GapCloseRecord",
    "GapClosingConfig",
    "GeometrySemanticCost",
    "GrainKinematics",
    "KinematicsSummary",
    "LineageEvent",
    "ObservationPrecursorScore",
    "PositionShapeCost",
    "PrecursorCandidate",
    "PrecursorChannelMetrics",
    "PrecursorConfig",
    "SplitMergeConfig",
    "TrackEvent",
    "TrackKalmanState",
    "WassersteinBuresCost",
    "apply_predicted_shifts",
    "associate_frames",
    "build_tracks",
    "compute_assignment_margins",
    "compute_fracture_precursors",
    "compute_grain_kinematics",
    "compute_tracking_metrics",
    "estimate_band_shifts_consensus",
    "estimate_slab_shifts_fft",
    "evaluate_precursor_channels_roc",
    "events_to_table",
    "generate_crossing_scenario",
    "generate_fracture_scenario",
    "group_local_rocking_patches",
    "is_single_slice_speck",
    "kinematics_to_table",
    "precursors_to_table",
    "resolve_splits_and_merges",
    "resolve_track_gaps",
    "summarize_kinematics",
    "summary_to_dict",
    "tracks_to_table",
]
