# behavioral-entropy

AI agent behavioral fingerprinting via entropy analysis.

`behavioral-entropy` measures Shannon entropy across agent activity streams
(timing patterns, decision latencies, action distributions) and generates
deterministic behavioral fingerprints. It can detect when two agents
exhibit similar behavior, score how unique an agent's behavioral profile
is relative to a population, and flag anomalous deviations from an
established baseline.

## Installation

```bash
pip install behavioral-entropy
```

For ML-based anomaly detection (IsolationForest, DBSCAN, PCA):

```bash
pip install behavioral-entropy[ml]
```

The core library requires only `numpy`. Scikit-learn and scipy are
optional; the profiler falls back to pure statistical methods when
they are not installed.

## Quick start

### Python API

```python
from behavioral_entropy import (
    AgentProfile,
    shannon_entropy_numeric,
    generate_fingerprint,
    profile_similarity,
    BehavioralProfiler,
)

# Measure entropy of a timing sequence
timings = [0.12, 0.11, 0.13, 0.12, 0.14, 0.11, 0.13, 0.12]
entropy = shannon_entropy_numeric(timings)
print(f"Timing entropy: {entropy:.3f} bits")

# Build a profile and generate a fingerprint
profiler = BehavioralProfiler()
profile = profiler.build_profile(
    agent_id="agent-alpha",
    timing_samples=[timings],
    action_choices={"query": 15, "index": 8, "delete": 2},
)
print(f"Fingerprint: {profile.fingerprint_hash}")
print(f"Behavioral entropy: {profile.behavioral_entropy:.3f}")

# Compare two profiles
profile_b = profiler.build_profile(
    agent_id="agent-beta",
    timing_samples=[[0.50, 0.48, 0.52, 0.49, 0.51]],
    action_choices={"index": 20, "query": 3},
)
cmp = profile_similarity(profile, profile_b)
print(f"Similarity: {cmp.similarity:.3f}")
```

### CLI

Analyze JSONL agent activity logs:

```bash
behavioral-entropy analyze activity.jsonl
```

Expected JSONL format (one JSON object per line):

```json
{"agent_id": "agent-1", "timestamp": 1700000000.0, "action": "query", "latency_ms": 120.5}
{"agent_id": "agent-1", "timestamp": 1700000001.2, "action": "query", "latency_ms": 115.3}
```

Measure text entropy:

```bash
behavioral-entropy text-entropy "The quick brown fox jumps over the lazy dog"
```

## Concepts

### Shannon entropy

Shannon entropy measures the unpredictability of a distribution. For a
discrete random variable with probability mass function p(x):

```
H(X) = -sum(p(x) * log2(p(x)))
```

A uniform distribution maximizes entropy (high unpredictability). A
constant sequence has zero entropy (fully predictable).

### Behavioral fingerprinting

Each AI agent produces observable behavioral signals: inter-request
timing intervals, decision latencies, action-type distributions, and
resource consumption patterns. These signals form a statistical
fingerprint that can distinguish one agent from another, even when the
agents perform the same nominal task.

`behavioral-entropy` computes a SHA-256 digest over the agent's
behavioral features, producing a deterministic fingerprint. Two profiles
with identical behavioral patterns yield the same fingerprint.

### Anomaly detection

When scikit-learn is installed, the `BehavioralProfiler` trains an
IsolationForest on an agent's historical feature vectors. New
observations are scored against this model. Deviations from the learned
distribution indicate potential identity changes, compromised agents,
or behavioral drift.

## Modules

| Module | Purpose |
|--------|---------|
| `entropy` | Shannon entropy calculations, text entropy, behavioral entropy |
| `fingerprint` | Fingerprint generation, profile similarity, uniqueness scoring |
| `profiler` | Feature extraction, ML-based profiling, anomaly detection |
| `types` | Shared dataclasses (`AgentProfile`, `EntropyMeasurement`, etc.) |
| `cli` | Command-line interface for JSONL analysis |

## Development

```bash
git clone https://github.com/ScrappinR/behavioral-entropy.git
cd behavioral-entropy
pip install -e ".[all]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
