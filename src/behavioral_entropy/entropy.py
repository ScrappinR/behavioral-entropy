"""
Shannon entropy calculations for behavioral data.

Provides functions to compute, normalize, and analyze Shannon entropy
over both numeric sequences and text token streams. Operates on raw
Python data structures; numpy is used when available but not required.
"""

import math
import re
import statistics
from typing import Dict, List, Optional, Any
from collections import Counter

from .types import EntropyMeasurement, EntropyProfile

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ── Token frequency tiers (English) ────────────────────────────────
# Used by measure_word_entropy to estimate per-word surprisal.

HIGH_FREQUENCY_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "but", "or", "if", "then", "so", "this", "that", "these",
    "those", "it", "its", "they", "their", "we", "our", "you", "your",
}

MID_FREQUENCY_WORDS = {
    "important", "system", "process", "data", "information", "method",
    "result", "analysis", "approach", "solution", "problem", "issue",
    "user", "application", "service", "function", "component", "module",
    "security", "performance", "quality", "standard", "requirement",
    "development", "implementation", "design", "architecture", "structure",
}

LOW_FREQUENCY_WORDS = {
    "ameliorate", "coalesce", "ephemeral", "ubiquitous", "paradigm",
    "synergy", "leverage", "holistic", "granular", "robust", "scalable",
    "orthogonal", "heterogeneous", "homogeneous", "deterministic",
    "stochastic", "heuristic", "algorithmic", "cryptographic", "semantic",
}


# ── Core entropy functions ──────────────────────────────────────────

def shannon_entropy(data: List[Any]) -> float:
    """
    Calculate Shannon entropy of a discrete distribution.

    H = -sum(p(x) * log2(p(x)))

    Works for any hashable elements (strings, ints, discretized floats).

    Args:
        data: Sequence of discrete symbols.

    Returns:
        Entropy in bits.
    """
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def shannon_entropy_numeric(values: List[float], bins: Optional[int] = None) -> float:
    """
    Calculate Shannon entropy of a continuous numeric sequence by
    discretizing into histogram bins.

    Args:
        values: Numeric sequence.
        bins: Number of histogram bins. Defaults to min(len(values)//2, 20).

    Returns:
        Entropy in bits.
    """
    if not values or len(values) < 2:
        return 0.0

    n_bins = bins or min(len(values) // 2, 20)
    n_bins = max(n_bins, 2)

    if _HAS_NUMPY:
        arr = np.array(values)
        hist, _ = np.histogram(arr, bins=n_bins)
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist[hist > 0] / total
        return float(-np.sum(probs * np.log2(probs)))
    else:
        # Pure-Python fallback
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return 0.0
        bin_width = (max_val - min_val) / n_bins
        hist = [0] * n_bins
        for v in values:
            idx = int((v - min_val) / bin_width)
            idx = min(idx, n_bins - 1)
            hist[idx] += 1
        total = sum(hist)
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy


def normalize_entropy(entropy: float, vocab_size: int) -> float:
    """
    Normalize entropy to [0, 1] by dividing by log2(vocab_size).

    Args:
        entropy: Raw Shannon entropy (bits).
        vocab_size: Number of distinct symbols.

    Returns:
        Normalized entropy in [0, 1].
    """
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log2(vocab_size)
    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


# ── Text-oriented entropy analysis ─────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer. Returns lowercase tokens."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def measure_text_entropy(
    text: str,
    window_size: Optional[int] = None,
) -> EntropyMeasurement:
    """
    Measure Shannon entropy of text tokens.

    Args:
        text: Input text.
        window_size: If provided, also compute per-window segment entropies.

    Returns:
        EntropyMeasurement with overall and per-segment values.
    """
    tokens = tokenize(text)
    if not tokens:
        return EntropyMeasurement(
            value=0.0, normalized=0.0, window_size=0,
            distribution={}, segment_entropies=[],
        )

    overall = shannon_entropy(tokens)
    vocab_size = len(set(tokens))
    norm = normalize_entropy(overall, vocab_size)
    distribution = dict(Counter(tokens))

    segment_entropies: List[float] = []
    if window_size and len(tokens) > window_size:
        for i in range(0, len(tokens), window_size):
            window = tokens[i:i + window_size]
            if len(window) >= window_size // 2:
                seg_e = shannon_entropy(window)
                seg_v = len(set(window))
                segment_entropies.append(normalize_entropy(seg_e, seg_v))

    return EntropyMeasurement(
        value=overall,
        normalized=norm,
        window_size=len(tokens),
        distribution=distribution,
        segment_entropies=segment_entropies,
    )


def measure_word_entropy(word: str) -> float:
    """
    Estimate single-word surprisal based on English frequency tier.

    Returns a value in [0, 1] where 0 = maximally predictable,
    1 = maximally surprising.
    """
    w = word.lower()
    if w in HIGH_FREQUENCY_WORDS:
        return 0.2
    if w in MID_FREQUENCY_WORDS:
        return 0.5
    if w in LOW_FREQUENCY_WORDS:
        return 0.8
    char_entropy = len(set(w)) / max(len(w), 1)
    length_factor = min(len(w) / 10, 1.0)
    return 0.4 + 0.3 * char_entropy + 0.1 * length_factor


def measure_segment_entropies(
    text: str,
    window_size: int = 50,
) -> List[float]:
    """
    Compute normalized entropy for each non-overlapping window of tokens.

    Args:
        text: Input text.
        window_size: Tokens per segment.

    Returns:
        List of normalized entropy values, one per segment.
    """
    tokens = tokenize(text)
    if not tokens:
        return []
    entropies: List[float] = []
    for i in range(0, len(tokens), window_size):
        seg = tokens[i:i + window_size]
        if len(seg) >= window_size // 2:
            e = shannon_entropy(seg)
            v = len(set(seg))
            entropies.append(normalize_entropy(e, v))
    return entropies


# ── Behavioral-data entropy ────────────────────────────────────────

def behavioral_entropy(
    timing_patterns: Optional[List[float]] = None,
    decision_times: Optional[List[float]] = None,
    action_counts: Optional[Dict[str, int]] = None,
) -> float:
    """
    Calculate aggregate behavioral entropy across multiple signal sources.

    Averages the per-source Shannon entropy. Each continuous source is
    discretized via histogram binning.

    Args:
        timing_patterns: Inter-event timing values.
        decision_times: Decision latency values.
        action_counts: Action -> count mapping (already discrete).

    Returns:
        Mean entropy across provided sources, or 0.0 if none given.
    """
    sources: List[float] = []
    if timing_patterns:
        sources.append(shannon_entropy_numeric(timing_patterns))
    if decision_times:
        sources.append(shannon_entropy_numeric(decision_times))
    if action_counts:
        values = list(action_counts.values())
        sources.append(shannon_entropy_numeric([float(v) for v in values]))
    return statistics.mean(sources) if sources else 0.0
