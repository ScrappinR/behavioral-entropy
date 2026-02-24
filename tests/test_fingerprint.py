"""Tests for the fingerprint module."""

import pytest

from behavioral_entropy.types import AgentProfile
from behavioral_entropy.fingerprint import (
    generate_fingerprint,
    fingerprint_hex,
    apply_fingerprint,
    sequence_similarity,
    dict_similarity,
    profile_similarity,
    uniqueness_score,
    authenticate,
)


def _make_profile(agent_id: str, **kwargs) -> AgentProfile:
    p = AgentProfile(agent_id=agent_id)
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


class TestGenerateFingerprint:
    def test_deterministic(self):
        p = _make_profile("a", timing_patterns=[0.1, 0.2, 0.3])
        fp1 = generate_fingerprint(p)
        fp2 = generate_fingerprint(p)
        assert fp1 == fp2

    def test_different_profiles_different_fingerprints(self):
        p1 = _make_profile("a", timing_patterns=[0.1, 0.2, 0.3])
        p2 = _make_profile("b", timing_patterns=[0.5, 0.6, 0.7])
        assert generate_fingerprint(p1) != generate_fingerprint(p2)

    def test_length_is_32_bytes(self):
        p = _make_profile("a", timing_patterns=[0.1])
        assert len(generate_fingerprint(p)) == 32

    def test_empty_profile_still_produces_fingerprint(self):
        p = _make_profile("empty")
        fp = generate_fingerprint(p)
        assert len(fp) == 32


class TestFingerprintHex:
    def test_hex_length(self):
        p = _make_profile("a", timing_patterns=[0.1, 0.2])
        assert len(fingerprint_hex(p)) == 64


class TestApplyFingerprint:
    def test_sets_fields(self):
        p = _make_profile("a", timing_patterns=[0.1, 0.2])
        result = apply_fingerprint(p)
        assert result is p
        assert p.fingerprint is not None
        assert p.fingerprint_hash is not None
        assert len(p.fingerprint_hash) == 64


class TestSequenceSimilarity:
    def test_identical_sequences(self):
        s = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert sequence_similarity(s, s) > 0.9

    def test_different_sequences(self):
        s1 = [0.1, 0.1, 0.1, 0.1]
        s2 = [100.0, 200.0, 300.0, 400.0]
        assert sequence_similarity(s1, s2) < 0.5

    def test_empty_returns_zero(self):
        assert sequence_similarity([], [1.0, 2.0]) == 0.0
        assert sequence_similarity([1.0], []) == 0.0


class TestDictSimilarity:
    def test_identical(self):
        d = {"a": 5, "b": 3}
        assert abs(dict_similarity(d, d) - 1.0) < 0.01

    def test_disjoint(self):
        d1 = {"a": 5}
        d2 = {"b": 3}
        assert dict_similarity(d1, d2) == 0.0

    def test_empty(self):
        assert dict_similarity({}, {}) == 1.0


class TestProfileSimilarity:
    def test_identical_profiles(self):
        p = _make_profile("a", timing_patterns=[0.1, 0.2, 0.3, 0.4])
        cmp = profile_similarity(p, p)
        assert cmp.similarity > 0.9

    def test_different_profiles(self):
        p1 = _make_profile("a", timing_patterns=[0.1] * 10)
        p2 = _make_profile("b", timing_patterns=[100.0] * 10)
        cmp = profile_similarity(p1, p2)
        assert cmp.similarity < 0.5


class TestUniquenessScore:
    def test_single_profile_is_unique(self):
        p = _make_profile("a")
        assert uniqueness_score(p, [p]) == 1.0

    def test_identical_profiles_not_unique(self):
        p1 = _make_profile("a", timing_patterns=[0.1, 0.2, 0.3, 0.4])
        p2 = _make_profile("b", timing_patterns=[0.1, 0.2, 0.3, 0.4])
        score = uniqueness_score(p1, [p1, p2])
        assert score < 0.2


class TestAuthenticate:
    def test_same_profile_authenticates(self):
        p = _make_profile("a", timing_patterns=[0.1, 0.2, 0.3, 0.4])
        ok, score = authenticate(p, p)
        assert ok is True
        assert score > 0.75

    def test_different_profile_fails(self):
        stored = _make_profile("a", timing_patterns=[0.1] * 10)
        observed = _make_profile("b", timing_patterns=[100.0] * 10)
        ok, score = authenticate(stored, observed)
        assert ok is False
