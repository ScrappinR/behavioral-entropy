"""Tests for the entropy module."""

import math
import pytest

from behavioral_entropy.entropy import (
    shannon_entropy,
    shannon_entropy_numeric,
    normalize_entropy,
    measure_text_entropy,
    measure_word_entropy,
    measure_segment_entropies,
    behavioral_entropy,
    tokenize,
)


class TestShannonEntropy:
    def test_empty_returns_zero(self):
        assert shannon_entropy([]) == 0.0

    def test_single_symbol(self):
        assert shannon_entropy(["a", "a", "a"]) == 0.0

    def test_two_equiprobable(self):
        # H = log2(2) = 1.0
        result = shannon_entropy(["a", "b", "a", "b"])
        assert abs(result - 1.0) < 0.01

    def test_four_equiprobable(self):
        # H = log2(4) = 2.0
        result = shannon_entropy(["a", "b", "c", "d"] * 100)
        assert abs(result - 2.0) < 0.01

    def test_skewed_lower_entropy(self):
        # Heavily skewed should have lower entropy than uniform
        skewed = ["a"] * 90 + ["b"] * 10
        uniform = ["a"] * 50 + ["b"] * 50
        assert shannon_entropy(skewed) < shannon_entropy(uniform)


class TestShannonEntropyNumeric:
    def test_empty_returns_zero(self):
        assert shannon_entropy_numeric([]) == 0.0

    def test_single_value_returns_zero(self):
        assert shannon_entropy_numeric([5.0]) == 0.0

    def test_constant_returns_zero(self):
        # All same value goes in one bin
        result = shannon_entropy_numeric([1.0] * 100)
        assert result == 0.0

    def test_uniform_spread_positive_entropy(self):
        import random
        random.seed(42)
        values = [random.uniform(0, 100) for _ in range(200)]
        result = shannon_entropy_numeric(values)
        assert result > 0.0

    def test_custom_bins(self):
        values = list(range(100))
        e10 = shannon_entropy_numeric([float(v) for v in values], bins=10)
        e5 = shannon_entropy_numeric([float(v) for v in values], bins=5)
        # More bins can capture more entropy
        assert e10 >= e5


class TestNormalizeEntropy:
    def test_zero_vocab(self):
        assert normalize_entropy(1.0, 0) == 0.0

    def test_one_vocab(self):
        assert normalize_entropy(1.0, 1) == 0.0

    def test_max_entropy_normalizes_to_one(self):
        # log2(8) = 3.0
        assert normalize_entropy(3.0, 8) == 1.0

    def test_half_entropy(self):
        result = normalize_entropy(1.5, 8)  # 1.5 / 3.0 = 0.5
        assert abs(result - 0.5) < 0.01


class TestMeasureTextEntropy:
    def test_empty_string(self):
        m = measure_text_entropy("")
        assert m.value == 0.0
        assert m.window_size == 0

    def test_simple_sentence(self):
        m = measure_text_entropy("the cat sat on the mat")
        assert m.value > 0
        assert m.window_size == 6
        assert "the" in m.distribution

    def test_with_window(self):
        text = "alpha beta gamma delta " * 20
        m = measure_text_entropy(text, window_size=10)
        assert len(m.segment_entropies) > 0


class TestMeasureWordEntropy:
    def test_high_frequency_word(self):
        assert measure_word_entropy("the") == 0.2

    def test_mid_frequency_word(self):
        assert measure_word_entropy("system") == 0.5

    def test_low_frequency_word(self):
        assert measure_word_entropy("stochastic") == 0.8

    def test_unknown_word_in_range(self):
        result = measure_word_entropy("xylophone")
        assert 0.0 < result < 1.0


class TestBehavioralEntropy:
    def test_no_sources(self):
        assert behavioral_entropy() == 0.0

    def test_timing_only(self):
        result = behavioral_entropy(timing_patterns=[0.1, 0.2, 0.15, 0.12, 0.18])
        assert result >= 0.0

    def test_multiple_sources(self):
        result = behavioral_entropy(
            timing_patterns=[0.1, 0.2, 0.15, 0.12, 0.18],
            decision_times=[50.0, 55.0, 48.0, 52.0, 60.0],
            action_counts={"query": 10, "index": 5, "delete": 2},
        )
        assert result > 0.0


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert tokenize("hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []
