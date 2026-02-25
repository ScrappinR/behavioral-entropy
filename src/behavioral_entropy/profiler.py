"""
Statistical profiling, feature extraction, and anomaly detection.

Builds behavioral profiles from raw measurement sequences using
statistical descriptors and, when available, scikit-learn models
(IsolationForest, DBSCAN, PCA, StandardScaler).

All ML dependencies are optional. If scikit-learn or scipy are not
installed, the profiler falls back to pure-Python statistics.
"""

import os
import time
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple

from .types import AgentProfile, AuthenticationResult
from .entropy import shannon_entropy_numeric

# scipy.stats and sklearn hang on import in MINGW/Git Bash on Windows
# (threading internals deadlock). Detect MINGW and skip entirely.
_IS_MINGW = os.environ.get("MSYSTEM", "").startswith("MINGW")

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

_HAS_SCIPY = False
sp_stats = None  # type: ignore[assignment]


def _get_scipy_stats():
    """Lazy import scipy.stats — skipped entirely on MINGW."""
    global sp_stats, _HAS_SCIPY
    if _IS_MINGW:
        return None
    if _HAS_SCIPY:
        return sp_stats
    try:
        import scipy.stats as _sp
        sp_stats = _sp
        _HAS_SCIPY = True
        return sp_stats
    except ImportError:
        return None


_HAS_SKLEARN = False
StandardScaler = None  # type: ignore[assignment,misc]
PCA = None  # type: ignore[assignment,misc]
DBSCAN = None  # type: ignore[assignment,misc]
IsolationForest = None  # type: ignore[assignment,misc]


def _load_sklearn():
    """Lazy import sklearn — skipped entirely on MINGW."""
    global StandardScaler, PCA, DBSCAN, IsolationForest, _HAS_SKLEARN
    if _IS_MINGW:
        return False
    if _HAS_SKLEARN:
        return True
    try:
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.decomposition import PCA as _PCA
        from sklearn.cluster import DBSCAN as _DB
        from sklearn.ensemble import IsolationForest as _IF
        StandardScaler = _SS
        PCA = _PCA
        DBSCAN = _DB
        IsolationForest = _IF
        _HAS_SKLEARN = True
        return True
    except ImportError:
        return False

logger = logging.getLogger(__name__)


# ── Feature extraction ──────────────────────────────────────────────

def extract_timing_features(timings: List[float]) -> Dict[str, float]:
    """
    Extract statistical features from a timing sequence.

    Works with pure Python; adds skewness/kurtosis when scipy is available.

    Args:
        timings: List of inter-event timing values.

    Returns:
        Dictionary of named features.
    """
    if len(timings) < 3:
        raise ValueError("Need at least 3 timing values for feature extraction")

    mean_val = statistics.mean(timings)
    std_val = statistics.stdev(timings) if len(timings) > 1 else 0.0
    median_val = statistics.median(timings)
    min_val = min(timings)
    max_val = max(timings)
    cv = std_val / (mean_val + 1e-10)
    q75 = _percentile(timings, 75)
    q25 = _percentile(timings, 25)
    iqr = q75 - q25

    features: Dict[str, float] = {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "coefficient_variation": cv,
        "interquartile_range": iqr,
        "entropy": shannon_entropy_numeric(timings),
        "rhythm_consistency": _rhythm_consistency(timings),
        "autocorrelation_lag1": _autocorrelation_lag1(timings),
    }

    _sp = _get_scipy_stats()
    if _sp is not None and _HAS_NUMPY:
        arr = np.array(timings)
        features["skewness"] = float(_sp.skew(arr))
        features["kurtosis"] = float(_sp.kurtosis(arr))

    return features


def extract_interval_features(timestamps: List[float]) -> Dict[str, float]:
    """
    Extract features from a sequence of absolute timestamps.

    First computes inter-event intervals, then delegates to
    extract_timing_features.
    """
    if len(timestamps) < 4:
        raise ValueError("Need at least 4 timestamps")
    sorted_ts = sorted(timestamps)
    intervals = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
    return extract_timing_features(intervals)


# ── Profile building ────────────────────────────────────────────────

class BehavioralProfiler:
    """
    Builds and evaluates behavioral profiles.

    Uses sklearn when available for clustering and anomaly detection;
    falls back to statistical thresholds otherwise.
    """

    def __init__(self, anomaly_sensitivity: float = 0.1):
        self.anomaly_sensitivity = anomaly_sensitivity

        # Per-agent ML state (only populated when sklearn is available)
        self._scalers: Dict[str, Any] = {}
        self._anomaly_detectors: Dict[str, Any] = {}
        self._pca_models: Dict[str, Any] = {}
        self._cluster_centers: Dict[str, List[List[float]]] = {}

        # Profile storage
        self.profiles: Dict[str, AgentProfile] = {}

    def build_profile(
        self,
        agent_id: str,
        timing_samples: Optional[List[List[float]]] = None,
        decision_samples: Optional[List[List[float]]] = None,
        action_choices: Optional[Dict[str, int]] = None,
        algorithm_preferences: Optional[Dict[str, int]] = None,
    ) -> AgentProfile:
        """
        Build a behavioral profile from raw measurement samples.

        Args:
            agent_id: Unique agent identifier.
            timing_samples: Multiple timing sequences.
            decision_samples: Multiple decision-latency sequences.
            action_choices: Aggregated action -> count map.
            algorithm_preferences: Aggregated algorithm -> count map.

        Returns:
            Populated AgentProfile with entropy, consistency, and
            fingerprint metrics computed.
        """
        profile = AgentProfile(agent_id=agent_id)

        # Flatten timing samples
        if timing_samples:
            flat_timings: List[float] = []
            for seq in timing_samples:
                flat_timings.extend(seq)
            profile.timing_patterns = flat_timings

        if decision_samples:
            flat_decisions: List[float] = []
            for seq in decision_samples:
                flat_decisions.extend(seq)
            profile.decision_times = flat_decisions

        if action_choices:
            profile.action_choices = dict(action_choices)

        if algorithm_preferences:
            profile.algorithm_preferences = dict(algorithm_preferences)

        # Compute derived metrics
        self._compute_metrics(profile)

        # Train ML models if sklearn available and we have enough data
        if timing_samples and _HAS_NUMPY and _load_sklearn():
            self._train_ml_models(agent_id, timing_samples)

        self.profiles[agent_id] = profile
        return profile

    def authenticate(
        self,
        agent_id: str,
        observed_timings: Optional[List[float]] = None,
        observed_actions: Optional[Dict[str, int]] = None,
        threshold: float = 0.75,
    ) -> AuthenticationResult:
        """
        Authenticate an agent against its stored profile.

        Uses ML anomaly detection when available, falls back to
        statistical distance comparison.
        """
        start = time.time()
        stored = self.profiles.get(agent_id)
        if not stored:
            return AuthenticationResult(
                agent_id=agent_id,
                authenticated=False,
                confidence_score=0.0,
                behavioral_distance=float("inf"),
                anomaly_score=-1.0,
                execution_time_ms=(time.time() - start) * 1000,
            )

        anomaly_score = 0.0
        distance = float("inf")

        # ML path
        if _HAS_NUMPY and _load_sklearn() and agent_id in self._anomaly_detectors:
            if observed_timings and len(observed_timings) >= 3:
                features = extract_timing_features(observed_timings)
                fv = np.array([list(features.values())])
                if agent_id in self._scalers:
                    fv = self._scalers[agent_id].transform(fv)
                    if agent_id in self._pca_models:
                        fv = self._pca_models[agent_id].transform(fv)
                score = self._anomaly_detectors[agent_id].score_samples(fv)
                anomaly_score = float(score[0])

                # Distance to nearest cluster center
                if agent_id in self._cluster_centers:
                    for center in self._cluster_centers[agent_id]:
                        ca = np.array(center).reshape(1, -1)
                        if ca.shape[1] == fv.shape[1]:
                            d = float(np.linalg.norm(fv[0] - ca[0]))
                            distance = min(distance, d)

        # Statistical fallback
        if distance == float("inf") and observed_timings and stored.timing_patterns:
            from .fingerprint import sequence_similarity
            sim = sequence_similarity(stored.timing_patterns[:100], observed_timings[:100])
            distance = 1.0 - sim

        confidence_factors = []
        if distance < float("inf"):
            confidence_factors.append(max(0.0, 1.0 - distance / 5.0))
        if anomaly_score != 0.0:
            confidence_factors.append(max(0.0, min(1.0, (anomaly_score + 1.0) / 2.0)))

        # Action-choice similarity
        if observed_actions and stored.action_choices:
            from .fingerprint import dict_similarity
            confidence_factors.append(dict_similarity(stored.action_choices, observed_actions))

        confidence = statistics.mean(confidence_factors) if confidence_factors else 0.0
        authenticated = confidence >= threshold

        return AuthenticationResult(
            agent_id=agent_id,
            authenticated=authenticated,
            confidence_score=confidence,
            behavioral_distance=distance if distance < float("inf") else -1.0,
            anomaly_score=anomaly_score,
            execution_time_ms=(time.time() - start) * 1000,
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _compute_metrics(self, profile: AgentProfile) -> None:
        """Compute entropy, consistency, and authentication strength."""
        from .entropy import behavioral_entropy as _be
        from .fingerprint import generate_fingerprint

        profile.behavioral_entropy = _be(
            timing_patterns=profile.timing_patterns or None,
            decision_times=profile.decision_times or None,
            action_counts=profile.action_choices or None,
        )

        # Pattern consistency (inverse of coefficient of variation)
        cv_scores: List[float] = []
        if len(profile.timing_patterns) > 1:
            m = statistics.mean(profile.timing_patterns)
            s = statistics.stdev(profile.timing_patterns)
            cv_scores.append(1.0 - min(s / (m + 1e-10), 1.0))
        if len(profile.decision_times) > 1:
            m = statistics.mean(profile.decision_times)
            s = statistics.stdev(profile.decision_times)
            cv_scores.append(1.0 - min(s / (m + 1e-10), 1.0))
        profile.pattern_consistency = statistics.mean(cv_scores) if cv_scores else 0.0

        # Authentication strength
        factors = [
            profile.behavioral_entropy / 10.0,
            profile.pattern_consistency,
            min(len(profile.timing_patterns) / 100.0, 1.0),
            min(len(profile.action_choices) / 10.0, 1.0),
        ]
        profile.authentication_strength = statistics.mean(factors)

        # Fingerprint
        fp = generate_fingerprint(profile)
        profile.fingerprint = fp
        profile.fingerprint_hash = fp.hex()

    def _train_ml_models(
        self,
        agent_id: str,
        timing_samples: List[List[float]],
    ) -> None:
        """Train sklearn models for anomaly detection and clustering."""
        feature_rows: List[List[float]] = []
        for seq in timing_samples:
            if len(seq) >= 3:
                try:
                    feats = extract_timing_features(seq)
                    feature_rows.append(list(feats.values()))
                except ValueError:
                    continue

        if len(feature_rows) < 2:
            return

        X = np.array(feature_rows)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self._scalers[agent_id] = scaler

        # PCA if many features
        if Xs.shape[1] > 10:
            pca = PCA(n_components=min(10, Xs.shape[0] - 1))
            Xs = pca.fit_transform(Xs)
            self._pca_models[agent_id] = pca

        # Clustering
        if len(Xs) >= 3:
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(Xs)
            centers = []
            for label in set(labels):
                if label != -1:
                    pts = Xs[labels == label]
                    centers.append(np.mean(pts, axis=0).tolist())
            if not centers:
                centers = [np.mean(Xs, axis=0).tolist()]
            self._cluster_centers[agent_id] = centers
        else:
            self._cluster_centers[agent_id] = [np.mean(Xs, axis=0).tolist()]

        # Anomaly detector
        detector = IsolationForest(
            contamination=self.anomaly_sensitivity, random_state=42
        )
        detector.fit(Xs)
        self._anomaly_detectors[agent_id] = detector


# ── Utility functions ───────────────────────────────────────────────

def _percentile(data: List[float], pct: float) -> float:
    """Pure-Python percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _rhythm_consistency(timings: List[float]) -> float:
    """Rhythm consistency: inverse of normalized interval variance."""
    if len(timings) < 4:
        return 0.0
    intervals = [timings[i + 1] - timings[i] for i in range(len(timings) - 1)]
    mean_i = statistics.mean(intervals)
    if mean_i == 0:
        return 0.0
    std_i = statistics.stdev(intervals)
    return 1.0 / (1.0 + std_i / abs(mean_i))


def _autocorrelation_lag1(values: List[float]) -> float:
    """Lag-1 autocorrelation."""
    if len(values) < 4:
        return 0.0
    if _HAS_NUMPY:
        arr = np.array(values)
        normed = (arr - np.mean(arr)) / (np.std(arr) + 1e-10)
        ac = float(np.corrcoef(normed[:-1], normed[1:])[0, 1])
        return ac if not (ac != ac) else 0.0  # nan check
    else:
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 1.0
        normed = [(v - m) / (s + 1e-10) for v in values]
        corr = sum(normed[i] * normed[i + 1] for i in range(len(normed) - 1))
        return corr / (len(normed) - 1)
