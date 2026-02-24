"""Tests for the profiler module."""

import pytest

from behavioral_entropy.profiler import (
    extract_timing_features,
    extract_interval_features,
    BehavioralProfiler,
)


class TestExtractTimingFeatures:
    def test_basic_extraction(self):
        timings = [0.10, 0.12, 0.11, 0.13, 0.10, 0.14, 0.11, 0.12]
        features = extract_timing_features(timings)
        assert "mean" in features
        assert "std" in features
        assert "entropy" in features
        assert "rhythm_consistency" in features
        assert features["mean"] > 0

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError):
            extract_timing_features([0.1, 0.2])

    def test_all_same_values(self):
        features = extract_timing_features([1.0, 1.0, 1.0, 1.0])
        assert features["std"] == 0.0


class TestExtractIntervalFeatures:
    def test_basic(self):
        timestamps = [1.0, 1.1, 1.25, 1.35, 1.5]
        features = extract_interval_features(timestamps)
        assert "mean" in features

    def test_too_few_timestamps_raises(self):
        with pytest.raises(ValueError):
            extract_interval_features([1.0, 2.0, 3.0])


class TestBehavioralProfiler:
    def test_build_profile(self):
        profiler = BehavioralProfiler()
        samples = [
            [0.10, 0.12, 0.11, 0.13, 0.10, 0.14, 0.11, 0.12],
            [0.11, 0.13, 0.12, 0.14, 0.11, 0.15, 0.12, 0.13],
            [0.09, 0.11, 0.10, 0.12, 0.09, 0.13, 0.10, 0.11],
        ]
        profile = profiler.build_profile(
            agent_id="test-agent",
            timing_samples=samples,
            action_choices={"query": 10, "index": 5},
        )
        assert profile.agent_id == "test-agent"
        assert profile.behavioral_entropy > 0
        assert profile.fingerprint is not None
        assert profile.fingerprint_hash is not None
        assert len(profile.timing_patterns) == 24  # 8 * 3

    def test_authenticate_known_agent(self):
        profiler = BehavioralProfiler()
        samples = [
            [0.10, 0.12, 0.11, 0.13, 0.10, 0.14, 0.11, 0.12],
            [0.11, 0.13, 0.12, 0.14, 0.11, 0.15, 0.12, 0.13],
        ]
        profiler.build_profile("test-agent", timing_samples=samples)

        result = profiler.authenticate(
            "test-agent",
            observed_timings=[0.10, 0.12, 0.11, 0.13, 0.10, 0.14, 0.11, 0.12],
        )
        assert result.agent_id == "test-agent"
        assert result.execution_time_ms >= 0

    def test_authenticate_unknown_agent(self):
        profiler = BehavioralProfiler()
        result = profiler.authenticate("unknown")
        assert result.authenticated is False
        assert result.confidence_score == 0.0
