"""Online causal streaming and FAL Level 3 autonomous DFXM "Scout-and-Zoom" engine."""

from .odd import (
    CHI2_DF3_P95,
    CHI2_DF3_P99,
    DEFAULT_VALIDITY_EPSILON,
    BulkReferencePrior,
    ExecutionMode,
    FALActiveLevel,
    HypothesisMode,
    OperationalDesignDomain,
    SimToRealAuditRecord,
    compute_sim_to_real_gap,
    spot_kurtosis_non_gaussianity,
)
from .replay import PolicyReplayMetrics, ReplayBenchmarkResult, evaluate_causal_replay
from .streaming import CausalStreamingTracker, StreamingStepResult
from .trigger import (
    ZoomTarget,
    ZoomTriggerPolicy,
    compute_predictive_std,
    compute_target_geometry,
    select_zoom_targets,
)

__all__ = [
    "CHI2_DF3_P95",
    "CHI2_DF3_P99",
    "DEFAULT_VALIDITY_EPSILON",
    "BulkReferencePrior",
    "CausalStreamingTracker",
    "ExecutionMode",
    "FALActiveLevel",
    "HypothesisMode",
    "OperationalDesignDomain",
    "PolicyReplayMetrics",
    "ReplayBenchmarkResult",
    "SimToRealAuditRecord",
    "StreamingStepResult",
    "ZoomTarget",
    "ZoomTriggerPolicy",
    "compute_predictive_std",
    "compute_sim_to_real_gap",
    "compute_target_geometry",
    "evaluate_causal_replay",
    "select_zoom_targets",
    "spot_kurtosis_non_gaussianity",
]
