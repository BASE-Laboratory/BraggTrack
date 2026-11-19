"""Online causal streaming, GP forecasting, and autonomous DFXM "Scout-and-Zoom" engine."""

from .gp import (
    AcquisitionKind,
    ForecastDomain,
    GPForecast,
    compute_electrochemical_state_coord,
    evaluate_acquisition,
    forecast_1d_gp,
    forecast_constant_velocity_2pt,
    forecast_grain_trajectory_gp,
    forecast_kalman_cv,
    forecast_persistence,
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
    "ForecastDomain",
    "GPForecast",
    "PolicyReplayMetrics",
    "ReplayBenchmarkResult",
    "StreamingStepResult",
    "ZoomTarget",
    "ZoomTriggerPolicy",
    "clamp_setpoint_bounds",
    "compute_electrochemical_state_coord",
    "compute_target_geometry",
    "evaluate_acquisition",
    "evaluate_causal_replay",
    "forecast_1d_gp",
    "forecast_constant_velocity_2pt",
    "forecast_grain_trajectory_gp",
    "forecast_kalman_cv",
    "forecast_persistence",
    "score_candidates_gp",
    "select_zoom_targets",
]
