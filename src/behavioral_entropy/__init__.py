"""
behavioral-entropy: AI agent behavioral fingerprinting via entropy analysis.

Provides tools for measuring Shannon entropy of agent activity streams,
generating deterministic behavioral fingerprints, computing profile
similarity, and detecting anomalous behavior.

Example::

    from behavioral_entropy import (
        AgentProfile,
        shannon_entropy_numeric,
        generate_fingerprint,
        BehavioralProfiler,
    )

    profile = AgentProfile(agent_id="agent-1")
    profile.timing_patterns = [0.12, 0.11, 0.13, 0.12, 0.14]
    fp = generate_fingerprint(profile)

"""

__version__ = "0.1.0"

# Types
from .types import (
    BehavioralPattern,
    BehavioralDataPoint,
    AgentProfile,
    EntropyMeasurement,
    EntropyProfile,
    AuthenticationResult,
    ProfileComparison,
)

# Entropy
from .entropy import (
    shannon_entropy,
    shannon_entropy_numeric,
    normalize_entropy,
    measure_text_entropy,
    measure_word_entropy,
    measure_segment_entropies,
    behavioral_entropy,
    tokenize,
)

# Fingerprinting
from .fingerprint import (
    generate_fingerprint,
    fingerprint_hex,
    apply_fingerprint,
    sequence_similarity,
    dict_similarity,
    profile_similarity,
    uniqueness_score,
    authenticate,
)

# Profiler
from .profiler import (
    extract_timing_features,
    extract_interval_features,
    BehavioralProfiler,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "BehavioralPattern",
    "BehavioralDataPoint",
    "AgentProfile",
    "EntropyMeasurement",
    "EntropyProfile",
    "AuthenticationResult",
    "ProfileComparison",
    # Entropy
    "shannon_entropy",
    "shannon_entropy_numeric",
    "normalize_entropy",
    "measure_text_entropy",
    "measure_word_entropy",
    "measure_segment_entropies",
    "behavioral_entropy",
    "tokenize",
    # Fingerprinting
    "generate_fingerprint",
    "fingerprint_hex",
    "apply_fingerprint",
    "sequence_similarity",
    "dict_similarity",
    "profile_similarity",
    "uniqueness_score",
    "authenticate",
    # Profiler
    "extract_timing_features",
    "extract_interval_features",
    "BehavioralProfiler",
]
