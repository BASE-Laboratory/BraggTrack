"""Online causal streaming and autonomous DFXM "Scout-and-Zoom" engine."""

from .replay import PolicyReplayMetrics, ReplayBenchmarkResult, evaluate_causal_replay
from .streaming import CausalStreamingTracker, StreamingStepResult
from .trigger import ZoomTarget, ZoomTriggerPolicy, compute_target_geometry, select_zoom_targets

__all__ = [
    "CausalStreamingTracker",
    "PolicyReplayMetrics",
    "ReplayBenchmarkResult",
    "StreamingStepResult",
    "ZoomTarget",
    "ZoomTriggerPolicy",
    "compute_target_geometry",
    "evaluate_causal_replay",
    "select_zoom_targets",
]
