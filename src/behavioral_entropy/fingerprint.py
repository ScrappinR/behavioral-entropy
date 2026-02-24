"""
Behavioral fingerprint generation and comparison.

Generates deterministic fingerprints (SHA-256) from an agent's behavioral
profile, and provides similarity / uniqueness scoring between profiles.
"""

import hashlib
import statistics
import struct
from typing import Dict, List, Optional, Tuple

from .types import AgentProfile, ProfileComparison

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ── Fingerprint generation ──────────────────────────────────────────

def generate_fingerprint(profile: AgentProfile) -> bytes:
    """
    Generate a SHA-256 fingerprint from an agent's behavioral signals.

    The fingerprint is a deterministic digest built from up to 32
    timing values, decision times, and sorted preference strings.

    Args:
        profile: Agent behavioral profile.

    Returns:
        32-byte SHA-256 digest.
    """
    components: List[bytes] = []

    if profile.timing_patterns:
        data = profile.timing_patterns[:32]
        components.append(struct.pack(f">{len(data)}f", *data))

    if profile.decision_times:
        data = profile.decision_times[:32]
        components.append(struct.pack(f">{len(data)}f", *data))

    if profile.action_choices:
        s = "".join(f"{k}:{v}" for k, v in sorted(profile.action_choices.items()))
        components.append(s.encode())

    if profile.algorithm_preferences:
        s = "".join(f"{k}:{v}" for k, v in sorted(profile.algorithm_preferences.items()))
        components.append(s.encode())

    combined = b"".join(components)
    return hashlib.sha256(combined).digest()


def fingerprint_hex(profile: AgentProfile) -> str:
    """Return the hex-encoded fingerprint string."""
    return generate_fingerprint(profile).hex()


def apply_fingerprint(profile: AgentProfile) -> AgentProfile:
    """
    Compute and store the fingerprint on the profile in-place.

    Args:
        profile: Profile to update.

    Returns:
        The same profile object, with .fingerprint and .fingerprint_hash set.
    """
    fp = generate_fingerprint(profile)
    profile.fingerprint = fp
    profile.fingerprint_hash = fp.hex()
    return profile


# ── Profile similarity ──────────────────────────────────────────────

def sequence_similarity(seq1: List[float], seq2: List[float]) -> float:
    """
    Statistical similarity between two numeric sequences.

    Compares distribution shape (mean and std) and returns a value
    in [0, 1] where 1 = identical distributions.
    """
    if not seq1 or not seq2:
        return 0.0
    try:
        mean1, mean2 = statistics.mean(seq1), statistics.mean(seq2)
        std1 = statistics.stdev(seq1) if len(seq1) > 1 else 0.0
        std2 = statistics.stdev(seq2) if len(seq2) > 1 else 0.0
        denom = max(std1, std2, 0.001)
        mean_diff = abs(mean1 - mean2) / denom
        std_diff = abs(std1 - std2) / denom
        return 1.0 / (1.0 + mean_diff + std_diff)
    except statistics.StatisticsError:
        return 0.0


def dict_similarity(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """
    Cosine similarity between two label->count dictionaries.

    Returns a value in [0, 1].
    """
    all_keys = set(dict1.keys()) | set(dict2.keys())
    if not all_keys:
        return 1.0
    vec1 = [dict1.get(k, 0) for k in all_keys]
    vec2 = [dict2.get(k, 0) for k in all_keys]
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def profile_similarity(a: AgentProfile, b: AgentProfile) -> ProfileComparison:
    """
    Compare two agent profiles across all available behavioral dimensions.

    Returns a ProfileComparison with per-dimension and aggregate similarity.
    """
    components: List[float] = []
    timing_sim = 0.0
    pref_sim = 0.0

    if a.timing_patterns and b.timing_patterns:
        timing_sim = sequence_similarity(
            a.timing_patterns[:100], b.timing_patterns[:100]
        )
        components.append(timing_sim)

    if a.decision_times and b.decision_times:
        components.append(
            sequence_similarity(a.decision_times[:100], b.decision_times[:100])
        )

    if a.action_choices and b.action_choices:
        pref_sim = dict_similarity(a.action_choices, b.action_choices)
        components.append(pref_sim)

    if a.algorithm_preferences and b.algorithm_preferences:
        components.append(
            dict_similarity(a.algorithm_preferences, b.algorithm_preferences)
        )

    overall = statistics.mean(components) if components else 0.0

    return ProfileComparison(
        agent_id_a=a.agent_id,
        agent_id_b=b.agent_id,
        similarity=overall,
        timing_similarity=timing_sim,
        preference_similarity=pref_sim,
        entropy_distance=abs(a.behavioral_entropy - b.behavioral_entropy),
    )


# ── Uniqueness scoring ──────────────────────────────────────────────

def uniqueness_score(
    profile: AgentProfile,
    population: List[AgentProfile],
) -> float:
    """
    Score how unique a profile is relative to a population.

    Returns 1.0 if the profile is completely distinct from every other
    profile, 0.0 if it is identical to at least one.

    Args:
        profile: The profile to evaluate.
        population: Other profiles to compare against.

    Returns:
        Uniqueness score in [0, 1].
    """
    similarities: List[float] = []
    for other in population:
        if other.agent_id == profile.agent_id:
            continue
        cmp = profile_similarity(profile, other)
        similarities.append(cmp.similarity)

    if not similarities:
        return 1.0
    return 1.0 - max(similarities)


# ── Authentication ──────────────────────────────────────────────────

def authenticate(
    stored: AgentProfile,
    observed: AgentProfile,
    threshold: float = 0.75,
) -> Tuple[bool, float]:
    """
    Authenticate an agent by comparing observed behavior to a stored profile.

    Args:
        stored: Reference profile.
        observed: Newly observed profile.
        threshold: Minimum similarity for authentication.

    Returns:
        (authenticated, similarity_score) tuple.
    """
    cmp = profile_similarity(stored, observed)
    return cmp.similarity >= threshold, cmp.similarity
