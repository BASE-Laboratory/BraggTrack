"""Online causal streaming, GP forecasting, and autonomous DFXM "Scout-and-Zoom" engine."""

from .gp import (
    AcquisitionKind,
    GPForecast,
    evaluate_acquisition,
    forecast_1d_gp,
    forecast_grain_trajectory_gp,
    score_candidates_gp,
)
from .replay import PolicyReplayMetrics, ReplayBenchmarkResult, evaluate_causal_replay
from .streaming import CausalStreamingTracker, StreamingStepResult
from .trigger import (
    ExecutionMode,
    ZoomTarget,
    ZoomTriggerPolicy,
    clamp_setpoint_bounds,
    compute_target_geometry,
    select_zoom_targets,
)

__all__ = [
    "AcquisitionKind",
    "CausalStreamingTracker",
    "ExecutionMode",
    "GPForecast",
    "PolicyReplayMetrics",
    "ReplayBenchmarkResult",
    "StreamingStepResult",
    "ZoomTarget",
    "ZoomTriggerPolicy",
    "clamp_setpoint_bounds",
    "compute_target_geometry",
    "evaluate_acquisition",
    "evaluate_causal_replay",
    "forecast_1d_gp",
    "forecast_grain_trajectory_gp",
    "score_candidates_gp",
    "select_zoom_targets",
]
