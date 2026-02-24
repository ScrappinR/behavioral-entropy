"""
Command-line interface for behavioral-entropy.

Accepts JSONL agent activity data on stdin or from a file and outputs
entropy analysis and behavioral fingerprints.

Usage::

    # From a file
    behavioral-entropy analyze activity.jsonl

    # From stdin
    cat activity.jsonl | behavioral-entropy analyze -

    # Text entropy
    behavioral-entropy text-entropy "The quick brown fox jumps over the lazy dog"

Expected JSONL format (one JSON object per line)::

    {"agent_id": "agent-1", "timestamp": 1700000000.0, "action": "query", "latency_ms": 120.5}
    {"agent_id": "agent-1", "timestamp": 1700000001.2, "action": "query", "latency_ms": 115.3}
    {"agent_id": "agent-2", "timestamp": 1700000000.5, "action": "index", "latency_ms": 85.1}

"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Any, TextIO

from .types import AgentProfile
from .entropy import (
    shannon_entropy_numeric,
    behavioral_entropy,
    measure_text_entropy,
)
from .fingerprint import (
    generate_fingerprint,
    profile_similarity,
    uniqueness_score,
)
from .profiler import extract_timing_features


def _parse_jsonl(source: TextIO) -> List[Dict[str, Any]]:
    """Parse JSONL from a file-like object."""
    records: List[Dict[str, Any]] = []
    for lineno, line in enumerate(source, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
    return records


def _group_by_agent(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by agent_id."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        aid = rec.get("agent_id", "unknown")
        groups[aid].append(rec)
    return dict(groups)


def _build_profile_from_records(
    agent_id: str, records: List[Dict[str, Any]]
) -> AgentProfile:
    """Build an AgentProfile from parsed JSONL records."""
    profile = AgentProfile(agent_id=agent_id)

    latencies: List[float] = []
    timestamps: List[float] = []
    actions: Dict[str, int] = defaultdict(int)

    for rec in records:
        if "latency_ms" in rec:
            latencies.append(float(rec["latency_ms"]))
        if "timestamp" in rec:
            timestamps.append(float(rec["timestamp"]))
        if "action" in rec:
            actions[str(rec["action"])] += 1

    if latencies:
        profile.decision_times = latencies
    if timestamps:
        sorted_ts = sorted(timestamps)
        intervals = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
        if intervals:
            profile.timing_patterns = intervals
    if actions:
        profile.action_choices = dict(actions)

    return profile


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze JSONL agent activity data."""
    if args.input == "-":
        records = _parse_jsonl(sys.stdin)
    else:
        with open(args.input, "r") as f:
            records = _parse_jsonl(f)

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    groups = _group_by_agent(records)
    profiles: List[AgentProfile] = []

    output: Dict[str, Any] = {"agents": {}}

    for agent_id, agent_records in sorted(groups.items()):
        profile = _build_profile_from_records(agent_id, agent_records)

        # Compute entropy
        ent = behavioral_entropy(
            timing_patterns=profile.timing_patterns or None,
            decision_times=profile.decision_times or None,
            action_counts=profile.action_choices or None,
        )
        profile.behavioral_entropy = ent

        # Fingerprint
        fp = generate_fingerprint(profile)
        profile.fingerprint = fp
        profile.fingerprint_hash = fp.hex()

        profiles.append(profile)

        agent_out: Dict[str, Any] = {
            "records": len(agent_records),
            "behavioral_entropy": round(ent, 4),
            "fingerprint": fp.hex(),
        }

        if profile.timing_patterns and len(profile.timing_patterns) >= 3:
            try:
                features = extract_timing_features(profile.timing_patterns)
                agent_out["timing_features"] = {
                    k: round(v, 6) for k, v in features.items()
                }
            except ValueError:
                pass

        if profile.decision_times:
            agent_out["decision_entropy"] = round(
                shannon_entropy_numeric(profile.decision_times), 4
            )

        if profile.action_choices:
            agent_out["action_distribution"] = profile.action_choices

        output["agents"][agent_id] = agent_out

    # Cross-agent comparison
    if len(profiles) > 1:
        comparisons: List[Dict[str, Any]] = []
        for i, pa in enumerate(profiles):
            pa.uniqueness_score = uniqueness_score(pa, profiles)
            output["agents"][pa.agent_id]["uniqueness_score"] = round(
                pa.uniqueness_score, 4
            )
            for pb in profiles[i + 1:]:
                cmp = profile_similarity(pa, pb)
                comparisons.append({
                    "agents": [cmp.agent_id_a, cmp.agent_id_b],
                    "similarity": round(cmp.similarity, 4),
                    "timing_similarity": round(cmp.timing_similarity, 4),
                    "preference_similarity": round(cmp.preference_similarity, 4),
                })
        output["comparisons"] = comparisons

    output["summary"] = {
        "total_records": len(records),
        "agents": len(profiles),
    }

    json.dump(output, sys.stdout, indent=2)
    print()


def cmd_text_entropy(args: argparse.Namespace) -> None:
    """Measure entropy of a text string."""
    text = args.text
    window = args.window

    measurement = measure_text_entropy(text, window_size=window)

    output = {
        "entropy_bits": round(measurement.value, 4),
        "normalized": round(measurement.normalized, 4),
        "token_count": measurement.window_size,
        "unique_tokens": len(measurement.distribution),
        "is_high_entropy": measurement.is_high_entropy,
        "is_low_entropy": measurement.is_low_entropy,
    }

    if measurement.segment_entropies:
        output["segment_entropies"] = [
            round(s, 4) for s in measurement.segment_entropies
        ]

    json.dump(output, sys.stdout, indent=2)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="behavioral-entropy",
        description="AI agent behavioral fingerprinting via entropy analysis.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze JSONL agent activity data.",
    )
    p_analyze.add_argument(
        "input",
        help="Path to JSONL file, or '-' for stdin.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # text-entropy
    p_text = subparsers.add_parser(
        "text-entropy",
        help="Measure Shannon entropy of a text string.",
    )
    p_text.add_argument("text", help="Text to analyze.")
    p_text.add_argument(
        "--window", type=int, default=None,
        help="Window size in tokens for segment analysis.",
    )
    p_text.set_defaults(func=cmd_text_entropy)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
