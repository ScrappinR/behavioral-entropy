"""Tests for the CLI module."""

import json
import io
import pytest

from behavioral_entropy.cli import _parse_jsonl, _group_by_agent, _build_profile_from_records


SAMPLE_RECORDS = [
    {"agent_id": "a1", "timestamp": 1000.0, "action": "query", "latency_ms": 120.5},
    {"agent_id": "a1", "timestamp": 1001.0, "action": "query", "latency_ms": 115.3},
    {"agent_id": "a1", "timestamp": 1002.5, "action": "index", "latency_ms": 130.1},
    {"agent_id": "a2", "timestamp": 1000.5, "action": "index", "latency_ms": 85.1},
    {"agent_id": "a2", "timestamp": 1001.5, "action": "index", "latency_ms": 90.2},
]


class TestParseJsonl:
    def test_basic_parsing(self):
        lines = "\n".join(json.dumps(r) for r in SAMPLE_RECORDS)
        records = _parse_jsonl(io.StringIO(lines))
        assert len(records) == 5

    def test_skips_blank_lines(self):
        lines = json.dumps(SAMPLE_RECORDS[0]) + "\n\n" + json.dumps(SAMPLE_RECORDS[1])
        records = _parse_jsonl(io.StringIO(lines))
        assert len(records) == 2

    def test_skips_malformed_lines(self, capsys):
        lines = json.dumps(SAMPLE_RECORDS[0]) + "\n{bad json\n" + json.dumps(SAMPLE_RECORDS[1])
        records = _parse_jsonl(io.StringIO(lines))
        assert len(records) == 2


class TestGroupByAgent:
    def test_grouping(self):
        groups = _group_by_agent(SAMPLE_RECORDS)
        assert "a1" in groups
        assert "a2" in groups
        assert len(groups["a1"]) == 3
        assert len(groups["a2"]) == 2


class TestBuildProfile:
    def test_builds_from_records(self):
        profile = _build_profile_from_records("a1", SAMPLE_RECORDS[:3])
        assert profile.agent_id == "a1"
        assert len(profile.decision_times) == 3
        assert len(profile.timing_patterns) == 2
        assert "query" in profile.action_choices
        assert "index" in profile.action_choices
