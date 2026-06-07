"""
Unsupervised anomaly detection via Isolation Forest.

The "Critic Agent": operates independently from the supervised NB + DT ensemble.
Trains online on a rolling buffer of healthy-state stable windows. When a new
stable window arrives, the critic scores it. If the critic says anomalous while
the supervised ensemble says healthy, that disagreement is a signal worth logging.

Why this matters: the supervised models were trained on Stage 1 Colab T4 data.
When deployed on a DGX B200 with different thermal characteristics, or on a GPU
whose thermal paste has degraded, the training distribution has shifted. The
supervised models remain confidently wrong. The Isolation Forest — which learned
only what healthy looks like on THIS specific GPU — will catch the drift.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import GPUState, AlertEvent
from .window import WindowResult

log = logging.getLogger(__name__)

# Feature vector: [rtheta_mean, rtheta_std, last_power, last_util, last_pstate]
FEATURE_DIM = 5

MIN_SAMPLES_TO_ACTIVATE = 60     # won't score until we've seen 60 healthy windows
RETRAIN_EVERY_N         = 30     # retrain IF every N new healthy samples
MAX_BASELINE_SAMPLES    = 500    # rolling buffer cap
IF_CONTAMINATION        = 0.05   # expected fraction of anomalies in healthy data
IF_N_ESTIMATORS         = 100
ALERT_COOLDOWN_S        = 300    # 5 min between critic alerts per GPU


@dataclass
class CriticResult:
    gpu_index:    int
    is_anomalous: bool
    score:        float   # negative = more anomalous; sklearn convention
    activated:    bool    # False if not enough data yet


class IsolationForestCritic:
    """
    Per-GPU unsupervised critic.

    Accumulates feature vectors from healthy (under_load, clean_idle) stable
    windows and periodically retrains an Isolation Forest. Scores every new
    stable window regardless of health state.

    Used by the daemon to detect distribution shift — when a GPU's healthy
    behaviour deviates from its own historical baseline.
    """

    def __init__(self):
        self._buffers:    dict[int, deque]  = {}   # healthy feature vectors
        self._models:     dict[int, object] = {}   # fitted Pipeline per GPU
        self._score_mean: dict[int, float]  = {}   # mean anomaly score on training data
        self._score_std:  dict[int, float]  = {}   # std of anomaly scores on training data
        self._since_last_retrain: dict[int, int] = {}
        self._last_alert_ts: dict[int, float] = {}
        self._n_scored: dict[int, int] = {}

    def _features(self, window: WindowResult) -> np.ndarray:
        return np.array([[
            window.rtheta_mean,
            window.rtheta_std,
            window.last_power,
            window.last_util,
            float(window.last_pstate),
        ]], dtype=float)

    def update_healthy(self, gpu: int, window: WindowResult) -> None:
        """Feed a confirmed healthy window to grow the baseline buffer."""
        if gpu not in self._buffers:
            self._buffers[gpu] = deque(maxlen=MAX_BASELINE_SAMPLES)
            self._since_last_retrain[gpu] = 0

        self._buffers[gpu].append(self._features(window)[0])
        self._since_last_retrain[gpu] += 1

        n = len(self._buffers[gpu])
        if n >= MIN_SAMPLES_TO_ACTIVATE and self._since_last_retrain[gpu] >= RETRAIN_EVERY_N:
            self._retrain(gpu)
            self._since_last_retrain[gpu] = 0

    def _retrain(self, gpu: int) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        X = np.array(list(self._buffers[gpu]))
        # RobustScaler (median + IQR) handles near-constant features better than
        # StandardScaler, which produces near-zero std on features like power_w
        # during pure-load training windows.
        model = Pipeline([
            ("scaler", RobustScaler()),
            ("if", IsolationForest(
                n_estimators  = IF_N_ESTIMATORS,
                contamination = IF_CONTAMINATION,
                random_state  = 42,
                n_jobs        = 1,
            )),
        ])
        model.fit(X)
        self._models[gpu] = model

        # Compute training-score statistics for a secondary threshold
        scores = model.score_samples(X)
        self._score_mean[gpu] = float(scores.mean())
        self._score_std[gpu]  = float(scores.std())
        log.debug("if_retrained gpu=%d n_samples=%d score_mean=%.4f score_std=%.4f",
                  gpu, len(X), self._score_mean[gpu], self._score_std[gpu])

    def score(self, gpu: int, window: WindowResult) -> CriticResult:
        """Score a stable window. Returns CriticResult with activated=False if not ready."""
        model = self._models.get(gpu)
        if model is None:
            return CriticResult(gpu_index=gpu, is_anomalous=False, score=0.0, activated=False)

        X = self._features(window)
        raw_score = float(model.score_samples(X)[0])  # more negative = more anomalous

        # Two-signal anomaly detection:
        # 1. sklearn predict() — contamination-based threshold
        # 2. z-score vs training distribution — catches extreme out-of-distribution points
        #    even when predict() threshold is poorly calibrated on small datasets
        predict_anomaly = model.predict(X)[0] == -1
        score_mean = self._score_mean.get(gpu, 0.0)
        score_std  = self._score_std.get(gpu, 1.0)
        zscore_anomaly = (raw_score - score_mean) / max(score_std, 1e-6) < -2.5
        is_anomalous = predict_anomaly or zscore_anomaly

        self._n_scored[gpu] = self._n_scored.get(gpu, 0) + 1
        return CriticResult(
            gpu_index    = gpu,
            is_anomalous = bool(is_anomalous),
            score        = round(raw_score, 4),
            activated    = True,
        )

    def maybe_alert(
        self,
        gpu:           int,
        window:        WindowResult,
        supervised_state: GPUState,
        timestamp:     float,
    ) -> Optional[AlertEvent]:
        """
        Emit an alert when the critic disagrees with the supervised classifier.

        Disagreement = supervised says CLEAN_IDLE or UNDER_LOAD, critic says anomalous.
        Agreement (both anomalous, or both healthy) → no critic alert.
        """
        result = self.score(gpu, window)
        if not result.activated or not result.is_anomalous:
            return None

        healthy_supervised = supervised_state in (GPUState.CLEAN_IDLE, GPUState.UNDER_LOAD)
        if not healthy_supervised:
            return None   # supervised already caught it — no need for critic to pile on

        last_alert = self._last_alert_ts.get(gpu, 0.0)
        if timestamp - last_alert < ALERT_COOLDOWN_S:
            return None

        self._last_alert_ts[gpu] = timestamp
        log.warning("critic_disagrees gpu=%d score=%.4f supervised=%s", gpu, result.score, supervised_state.name)

        return AlertEvent(
            gpu_index       = gpu,
            timestamp       = timestamp,
            state           = supervised_state,
            prev_state      = supervised_state,
            rtheta          = window.rtheta_mean,
            rtheta_baseline = None,
            drift_sigma     = None,
            confidence      = 0.70,
            message         = (
                f"[WARNING] GPU {gpu} — unsupervised critic disagrees with supervised classifier. "
                f"Isolation Forest scores this window as anomalous (score={result.score:.3f}) "
                f"while ensemble classifies it as {supervised_state.name}. "
                f"Possible distribution shift: GPU hardware, firmware, or thermal environment "
                f"has changed from the training baseline. Monitor R_θ trend."
            ),
            context = {
                "severity":         "warning",
                "critic_score":     result.score,
                "critic_anomalous": True,
                "supervised_state": supervised_state.name,
                "rtheta_mean":      window.rtheta_mean,
                "rtheta_std":       window.rtheta_std,
            },
        )
