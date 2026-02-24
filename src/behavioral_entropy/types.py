"""
Shared data types for behavioral entropy analysis.

Defines the core dataclasses used across entropy, fingerprinting,
and profiling modules.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class BehavioralPattern(Enum):
    """Types of behavioral patterns captured from AI agents."""
    TIMING_RHYTHM = "timing_rhythm"
    DECISION_LATENCY = "decision_latency"
    PREFERENCE_CONSISTENCY = "preference_consistency"
    ERROR_HANDLING = "error_handling"
    RESOURCE_CONSUMPTION = "resource_consumption"
    ALGORITHM_BIAS = "algorithm_bias"
    COMMUNICATION_CADENCE = "communication_cadence"
    CONFIDENCE_TIMING = "confidence_timing"
    DATA_PROCESSING_RHYTHM = "data_processing_rhythm"
    REQUEST_PATTERN = "request_pattern"


@dataclass
class BehavioralDataPoint:
    """Individual behavioral measurement."""
    pattern_type: BehavioralPattern
    value: Any
    timestamp: float
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "direct_observation"


@dataclass
class AgentProfile:
    """Complete behavioral profile for an AI agent."""
    agent_id: str
    profile_created: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Core behavioral patterns
    timing_patterns: List[float] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)
    action_choices: Dict[str, int] = field(default_factory=dict)
    error_responses: List[str] = field(default_factory=list)
    resource_usage: Dict[str, List[float]] = field(
        default_factory=lambda: {"cpu": [], "memory": [], "network": []}
    )
    algorithm_preferences: Dict[str, int] = field(default_factory=dict)
    communication_intervals: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)

    # Statistical analysis results
    behavioral_entropy: float = 0.0
    pattern_consistency: float = 0.0
    uniqueness_score: float = 0.0
    authentication_strength: float = 0.0

    # Fingerprint
    fingerprint: Optional[bytes] = None
    fingerprint_hash: Optional[str] = None


@dataclass
class EntropyMeasurement:
    """
    Result of entropy measurement for a data segment.

    Attributes:
        value: Shannon entropy value (bits).
        normalized: Entropy normalized to 0-1 range.
        window_size: Number of tokens or data points analyzed.
        distribution: Frequency distribution of observed symbols.
        segment_entropies: Per-segment entropy values.
    """
    value: float
    normalized: float
    window_size: int
    distribution: Dict[str, int] = field(default_factory=dict)
    segment_entropies: List[float] = field(default_factory=list)

    @property
    def is_high_entropy(self) -> bool:
        """Check if entropy is above typical threshold."""
        return self.normalized > 0.6

    @property
    def is_low_entropy(self) -> bool:
        """Check if entropy is below typical threshold."""
        return self.normalized < 0.4


@dataclass
class EntropyProfile:
    """
    Configuration profile for entropy analysis.

    Attributes:
        baseline_entropy: Normal entropy for this context (0-1 normalized).
        entropy_std: Standard deviation of entropy.
        low_threshold: Entropy level considered low (normalized).
        high_threshold: Entropy level considered high (normalized).
        window_size: Number of tokens per analysis window.
    """
    baseline_entropy: float = 0.5
    entropy_std: float = 0.15
    low_threshold: float = 0.35
    high_threshold: float = 0.65
    window_size: int = 50


@dataclass
class AuthenticationResult:
    """Result of behavioral authentication."""
    agent_id: str
    authenticated: bool
    confidence_score: float
    behavioral_distance: float
    anomaly_score: float
    pattern_matches: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProfileComparison:
    """Result of comparing two behavioral profiles."""
    agent_id_a: str
    agent_id_b: str
    similarity: float
    timing_similarity: float = 0.0
    preference_similarity: float = 0.0
    entropy_distance: float = 0.0
