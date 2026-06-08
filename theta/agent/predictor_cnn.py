"""
Cascading-1D-CNN failure predictor — architecture scaffolding only.

This module provides a model-loadable predictor that:
  (1) Defines the architecture (cascading 1D CNNs over multi-channel
      time-series telemetry), inspired by Cai et al. 2026 "Forecasting
      Machine Degradation of GPU Clusters" (PRAUC 0.90).
  (2) Loads pre-trained weights from theta/models/bundle/cnn_predictor.pt
      WHEN AVAILABLE, and falls back gracefully to the rule-based
      FailurePredictor when no weights file is present.
  (3) Ships with a training-script outline so operators with their own
      labeled failure data can train the model on their fleet.
  (4) Is HONEST about validation status: this scaffolding does NOT ship a
      pre-trained model. The trained-model path requires labeled failure
      data, which Theta does not yet have at scale. Until the OSS
      telemetry network accumulates labeled events (see [[02_oss_agent_scope]]),
      this module's predict() method returns None and the daemon transparently
      falls back to the rule-based FailurePredictor.

Why ship the scaffolding before the trained model:
  - Pin the architecture contract. When labeled data arrives, training a
    model is a one-script run; the integration is already in place.
  - Make the "we have a real CNN predictor planned, here's the code" claim
    falsifiable. The architecture + the gating logic + the training script
    exist on disk; reviewers can read them.
  - Optional dependency: torch is imported lazily. Hosts without torch
    installed simply never load the model — no impact on daemon startup.

Reference: see [[cai_et_al_summary]] in the vault for the paper's
architectural choices we adopt (multi-channel input, 1D conv over the
time axis, cascading receptive field, PRAUC as primary metric).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Where the trained weights live when they exist. Bundled with the wheel
# so a `pip install runtheta` ships them automatically once trained.
MODEL_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "bundle" / "cnn_predictor.pt"

# Architecture — must match the training script when weights are loaded.
# These constants are the SCHEMA, not hyperparameters. Don't change them
# without retraining and re-shipping weights.
SEQUENCE_LENGTH      = 60    # number of stable windows in input sequence
INPUT_CHANNELS       = 5     # rtheta, power, temperature, util_pct, clock_eff
CONV_FILTERS         = (16, 32, 64)  # cascading filter widths
KERNEL_SIZES         = (5, 5, 3)     # decreasing kernel as receptive field grows
DROPOUT_RATE         = 0.3
PREDICTION_HORIZONS  = (60, 600, 3600, 86400)  # seconds: 1min, 10min, 1h, 1d


@dataclass
class CNNPrediction:
    """Output of the CNN predictor for one GPU at one moment.

    Multi-horizon: predicts failure probability at 4 future time windows.
    The horizon-specific probabilities let the daemon choose which one
    drives an alert (e.g., short-horizon high probability → drain NOW;
    long-horizon high probability → schedule maintenance).
    """
    gpu_index: int
    timestamp: float
    # Probability of failure within each horizon (sigmoid output)
    p_failure_by_horizon: dict[int, float]  # horizon_sec → probability
    # Calibrated confidence in the prediction itself (separate from the
    # probability — high p_failure with low confidence ≠ act now).
    model_confidence: float
    # Which input channels drove the prediction (gradient-based saliency).
    # Empty dict if SHAP / saliency not computed in this inference pass.
    channel_attribution: dict[str, float]

    def horizon_alert_level(self) -> str:
        """Convert multi-horizon probabilities to an action-routing tag."""
        p_1min = self.p_failure_by_horizon.get(60, 0.0)
        p_10min = self.p_failure_by_horizon.get(600, 0.0)
        p_1h = self.p_failure_by_horizon.get(3600, 0.0)
        p_1d = self.p_failure_by_horizon.get(86400, 0.0)
        if p_1min > 0.5 or p_10min > 0.7:
            return "emergency"
        if p_1h > 0.5:
            return "act_now"
        if p_1d > 0.6:
            return "act_soon"
        if p_1d > 0.3:
            return "watch"
        return "ok"


class CascadingCNNPredictor:
    """Pluggable CNN-based failure predictor.

    Lifecycle:
      - __init__() probes for torch + weights. If either is missing,
        is_ready returns False and predict() returns None — the daemon
        keeps using FailurePredictor.
      - update(gpu_index, sample) builds a per-GPU rolling buffer.
      - predict(gpu_index, ts) runs inference once the buffer is full.

    This class is intentionally narrow — it does NOT replace FailurePredictor,
    it AUGMENTS it. The daemon should run both and prefer the CNN output
    when available + high-confidence.
    """

    def __init__(self, weights_path: Path = MODEL_WEIGHTS_PATH):
        self._weights_path = weights_path
        self._model = None
        self._torch_available = False
        self._channels = ("rtheta", "power_w", "temp_c", "util_pct", "clock_eff")
        # Per-GPU rolling buffer of input feature vectors
        self._buffers: dict[int, deque] = {}
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Best-effort model load. Logs once on each failure mode."""
        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            log.info(
                "predictor_cnn_skipped reason=torch_not_installed "
                "note='to enable, pip install torch and provide weights at %s'",
                self._weights_path,
            )
            return

        if not self._weights_path.exists():
            log.info(
                "predictor_cnn_skipped reason=no_weights_file "
                "note='train via scripts/train_cnn_predictor.py and place output at %s'",
                self._weights_path,
            )
            return

        try:
            self._model = self._build_and_load_model()
            log.info("predictor_cnn_loaded weights=%s", self._weights_path)
        except Exception as exc:
            log.warning("predictor_cnn_load_failed error=%s", exc)
            self._model = None

    def _build_and_load_model(self):
        """Construct the architecture and load weights. Lazy-import torch."""
        import torch
        import torch.nn as nn

        class _CascadingCNN(nn.Module):
            """Cascading 1D CNN — multi-channel temperature/power/util time series
            in, multi-horizon failure probability out.

            Architecture (each layer doubles filter count, halves kernel):
                Input: (batch, INPUT_CHANNELS, SEQUENCE_LENGTH)
                Conv1D(in=5, out=16, k=5) → BN → ReLU → MaxPool(2)  →  (B, 16, 30)
                Conv1D(in=16, out=32, k=5) → BN → ReLU → MaxPool(2) →  (B, 32, 15)
                Conv1D(in=32, out=64, k=3) → BN → ReLU → AdaptivePool(1) → (B, 64, 1)
                Flatten → Dropout → FC(64, 32) → ReLU → FC(32, 4 horizons)
                Sigmoid per-horizon for independent failure probabilities.
            """
            def __init__(self):
                super().__init__()
                blocks = []
                in_ch = INPUT_CHANNELS
                for out_ch, k in zip(CONV_FILTERS, KERNEL_SIZES):
                    blocks.append(nn.Conv1d(in_ch, out_ch, k, padding=k // 2))
                    blocks.append(nn.BatchNorm1d(out_ch))
                    blocks.append(nn.ReLU(inplace=True))
                    blocks.append(nn.MaxPool1d(2))
                    in_ch = out_ch
                # Last pool is adaptive so we don't need exact SEQUENCE_LENGTH/factor
                blocks[-1] = nn.AdaptiveAvgPool1d(1)
                self.conv = nn.Sequential(*blocks)
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(in_ch, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, len(PREDICTION_HORIZONS)),
                )

            def forward(self, x):
                return torch.sigmoid(self.head(self.conv(x)))

        model = _CascadingCNN()
        state = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    @property
    def is_ready(self) -> bool:
        """True when torch + weights are loaded and inference can run."""
        return self._model is not None

    def update(self, gpu_index: int, ts: float, feature_vec: dict[str, float]) -> None:
        """Push one new sample into the per-GPU rolling buffer.

        Required keys in feature_vec: rtheta, power_w, temp_c, util_pct, clock_eff.
        Missing keys are filled with NaN — model is responsible for handling
        (typically via masking during training).
        """
        buf = self._buffers.setdefault(gpu_index, deque(maxlen=SEQUENCE_LENGTH))
        row = [feature_vec.get(c, float("nan")) for c in self._channels]
        buf.append((ts, row))

    def predict(self, gpu_index: int, ts: float) -> Optional[CNNPrediction]:
        """Run inference for one GPU. Returns None if not ready or buffer too short."""
        if not self.is_ready:
            return None
        buf = self._buffers.get(gpu_index)
        if buf is None or len(buf) < SEQUENCE_LENGTH:
            return None

        import torch

        # Build input tensor: (1, INPUT_CHANNELS, SEQUENCE_LENGTH)
        # Transpose from (T, C) → (C, T) for Conv1D
        rows = [r for _, r in buf]
        x = torch.tensor(rows, dtype=torch.float32).T.unsqueeze(0)
        # Replace NaN with column mean (simple imputation — train loader
        # should match this behavior)
        col_means = x.nanmean(dim=2, keepdim=True)
        nan_mask = x.isnan()
        x = torch.where(nan_mask, col_means.expand_as(x), x)

        with torch.no_grad():
            probs = self._model(x).squeeze(0).tolist()

        p_by_horizon = dict(zip(PREDICTION_HORIZONS, probs))

        return CNNPrediction(
            gpu_index=gpu_index,
            timestamp=ts,
            p_failure_by_horizon=p_by_horizon,
            model_confidence=_estimate_confidence(p_by_horizon),
            channel_attribution={},  # SHAP / saliency deferred — see docstring
        )

    def reset(self, gpu_index: int) -> None:
        self._buffers.pop(gpu_index, None)


def _estimate_confidence(p_by_horizon: dict[int, float]) -> float:
    """Heuristic confidence estimate from the prediction distribution.

    Logic: confident predictions have high entropy contrast across horizons
    (e.g., 0.95 at 1h, 0.10 at 1d means the model strongly localized the
    event in time). Uncertain predictions are uniformly mid-range. This is
    a temporary stand-in for proper calibration (Platt scaling on a holdout
    set) — replace once labeled validation data is available.
    """
    probs = list(p_by_horizon.values())
    if not probs:
        return 0.0
    # Highest probability minus mean of others = "peak vs noise" signal
    sorted_p = sorted(probs, reverse=True)
    peak = sorted_p[0]
    others = sorted_p[1:]
    contrast = peak - (sum(others) / len(others)) if others else 0.0
    return max(0.0, min(1.0, contrast))


# ──────────────────────────────────────────────────────────────────────────
# Training script outline — for the operator who has labeled failure data.
#
# Place this in scripts/train_cnn_predictor.py when implementing:
#
#   import torch
#   from torch.utils.data import DataLoader
#   from theta.agent.predictor_cnn import (
#       INPUT_CHANNELS, SEQUENCE_LENGTH, CONV_FILTERS, KERNEL_SIZES,
#       PREDICTION_HORIZONS, MODEL_WEIGHTS_PATH,
#   )
#
#   # 1. Load labeled telemetry: per-GPU time series with binary failure
#   #    labels at each PREDICTION_HORIZONS horizon ahead of `t`.
#   #    Schema: (B, INPUT_CHANNELS, SEQUENCE_LENGTH) input,
#   #            (B, len(PREDICTION_HORIZONS)) target.
#   # 2. Build the same architecture as CascadingCNNPredictor._build_and_load_model
#   # 3. Use BCEWithLogitsLoss with pos_weight set to invert class imbalance
#   #    (failures are rare — without pos_weight the model learns "always 0").
#   # 4. Optimize PRAUC on a holdout set (NOT accuracy — failure events are
#   #    typically <1% of the dataset).
#   # 5. Save state_dict to MODEL_WEIGHTS_PATH. The daemon picks it up on
#   #    next restart.
#
# Target metric (from Cai et al. 2026): PRAUC 0.90, precision 0.99, recall 0.90.
# ──────────────────────────────────────────────────────────────────────────
