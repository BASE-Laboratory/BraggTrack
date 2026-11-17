"""Multi-scan grain tracking with physics and semantic cost functions."""

from .assignment import associate_frames, compute_assignment_margins
from .cost import CostFunction, GeometrySemanticCost, PositionShapeCost
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
    PrecursorConfig,
    compute_fracture_precursors,
    precursors_to_table,
)
from .synthetic import generate_crossing_scenario, generate_fracture_scenario

__all__ = [
    "CostFunction",
    "GapCloseRecord",
    "GapClosingConfig",
    "GeometrySemanticCost",
    "GrainKinematics",
    "KinematicsSummary",
    "LineageEvent",
    "ObservationPrecursorScore",
    "PositionShapeCost",
    "PrecursorCandidate",
    "PrecursorConfig",
    "SplitMergeConfig",
    "TrackEvent",
    "associate_frames",
    "build_tracks",
    "compute_assignment_margins",
    "compute_fracture_precursors",
    "compute_grain_kinematics",
    "compute_tracking_metrics",
    "events_to_table",
    "generate_crossing_scenario",
    "generate_fracture_scenario",
    "kinematics_to_table",
    "precursors_to_table",
    "resolve_splits_and_merges",
    "resolve_track_gaps",
    "summarize_kinematics",
    "summary_to_dict",
    "tracks_to_table",
]
