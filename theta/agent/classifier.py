"""
GPU state classifier using pre-trained Naive Bayes + Decision Tree models.

Models are trained on Stage 1 data (4,570 rows, Tesla T4) with the
15-second steady-state filter applied. Accuracy: NB 99.8%, DT 100%.

At import time, this module tries to load bundled .pkl models. If they
don't exist yet, it falls back to the hard-coded decision tree rules
derived from the Orange Data Mining analysis (2026-06-04).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .metrics import GPUState, CLASS_INDEX_TO_STATE
from .window import WindowResult

log = logging.getLogger(__name__)

BUNDLE_DIR = Path(__file__).parent.parent / "models" / "bundle"
NB_MODEL_PATH = BUNDLE_DIR / "nb_steady_state.pkl"
DT_MODEL_PATH = BUNDLE_DIR / "dt_steady_state.pkl"

# T4 defaults — overridden per-GPU when calibration is present
_T4_LOAD_THRESHOLD = 0.87
_T4_IDLE_THRESHOLD = 1.50


# ── Hard-coded fallback rules from Orange DT analysis ─────────────────────
# Decision Tree (depth ≤ 5, 100% accuracy on steady-state data):
#
#   IF R_theta ≤ load_threshold:
#     → under_load
#   ELSE IF P-state = P0:
#     → zombie_recovery
#   ELSE (P-state ≥ P1):
#     IF R_theta ≤ idle_threshold:
#       IF power ≤ 12.83W → child_exit_recovery OR clean_idle (power < 10.06W → clean_idle)
#     ELSE (R_theta > idle_threshold):
#       → child_exit_recovery

def _rule_classify(
    rtheta: float,
    power: float,
    pstate: int,
    load_threshold: float = _T4_LOAD_THRESHOLD,
    idle_threshold: float = _T4_IDLE_THRESHOLD,
) -> tuple[GPUState, float]:
    """Decision tree rules with optional calibrated thresholds."""
    if rtheta <= load_threshold:
        return GPUState.UNDER_LOAD, 0.99
    if pstate == 0:
        return GPUState.ZOMBIE_RECOVERY, 1.00
    # pstate ≥ 1
    if rtheta <= idle_threshold:
        if power <= 12.83:
            if power <= 10.06:
                return GPUState.CLEAN_IDLE, 1.00
            return GPUState.CHILD_EXIT_RECOVERY, 0.98
        return GPUState.CLEAN_IDLE, 0.95
    return GPUState.CHILD_EXIT_RECOVERY, 0.99


class StateClassifier:
    """
    Classifies GPU state from a stable WindowResult.

    When both models are available, runs ensemble voting: NB + DT must agree
    for full confidence. Disagreement caps confidence at 0.65 and logs the
    conflict — useful signal for distribution shift (new GPU hardware, updated
    firmware, workload patterns not in Stage 1 training data).

    When a CalibrationManager is provided and calibration exists for a GPU,
    the classifier always falls back to calibrated rule-based classification
    rather than using the T4-trained ML models (which would misclassify on
    hardware with different R_theta ranges).

    Priority: calibrated-rules (when cal present) → ensemble → single DT → single NB → T4 rules.
    """

    def __init__(self, prefer_interpretable: bool = True, calibration=None):
        self._prefer_dt    = prefer_interpretable
        self._nb_model     = None
        self._dt_model     = None
        self._mode         = "rules"
        self._calibration  = calibration  # Optional[CalibrationManager]
        self._load_models()

    def _load_models(self) -> None:
        try:
            import joblib
            # Always attempt to load BOTH models for ensemble
            if DT_MODEL_PATH.exists():
                self._dt_model = joblib.load(DT_MODEL_PATH)
                log.info("Loaded Decision Tree model from bundle")
            if NB_MODEL_PATH.exists():
                self._nb_model = joblib.load(NB_MODEL_PATH)
                log.info("Loaded Naive Bayes model from bundle")

            if self._dt_model and self._nb_model:
                self._mode = "ensemble"
            elif self._dt_model:
                self._mode = "dt"
            elif self._nb_model:
                self._mode = "nb"
            else:
                log.warning("No bundled models found — using hard-coded DT rules (100% Stage 1 accuracy)")
        except ImportError:
            log.warning("joblib not available — using hard-coded DT rules")

    def classify(self, window: WindowResult) -> tuple[GPUState, float]:
        """
        Returns (state, confidence) from a steady-state window.
        Only call when window.is_stable == True.

        When calibration is present for this GPU, calibrated rules are used
        and ML models are bypassed — the T4-trained models are not reliable
        on hardware with a different R_theta range.

        In ensemble mode (no calibration): both models vote. Agreement boosts
        confidence by 5% (capped at 1.0). Disagreement caps confidence at 0.65.
        """
        # Calibrated path — always rule-based with hardware-specific thresholds
        if self._calibration is not None:
            cal = self._calibration.get(window.gpu_index)
            if cal is not None:
                return _rule_classify(
                    window.rtheta_mean,
                    window.last_power,
                    window.last_pstate,
                    load_threshold=cal.load_threshold,
                    idle_threshold=cal.idle_threshold,
                )

        X = np.array([[
            window.rtheta_mean,
            window.last_power,
            window.last_util,
            float(window.last_pstate),
        ]])

        if self._mode == "ensemble":
            dt_pred  = int(self._dt_model.predict(X)[0])
            nb_pred  = int(self._nb_model.predict(X)[0])
            dt_proba = self._dt_model.predict_proba(X)[0]
            nb_proba = self._nb_model.predict_proba(X)[0]
            dt_conf  = float(dt_proba[dt_pred])
            nb_conf  = float(nb_proba[nb_pred])

            if dt_pred == nb_pred:
                state      = CLASS_INDEX_TO_STATE.get(dt_pred, GPUState.UNKNOWN)
                confidence = min(1.0, (dt_conf + nb_conf) / 2 * 1.05)
            else:
                # Models disagree — use DT (interpretable) but flag low confidence
                state      = CLASS_INDEX_TO_STATE.get(dt_pred, GPUState.UNKNOWN)
                confidence = min(dt_conf, 0.65)
                log.info(
                    "ensemble_disagree",
                    dt_pred=dt_pred, nb_pred=nb_pred,
                    dt_conf=round(dt_conf, 3), nb_conf=round(nb_conf, 3),
                    rtheta=window.rtheta_mean,
                )
            return state, confidence

        if self._mode == "dt" and self._dt_model is not None:
            pred = int(self._dt_model.predict(X)[0])
            try:
                proba = self._dt_model.predict_proba(X)[0]
                conf  = float(proba[pred])
            except Exception:
                conf  = 1.0
            return CLASS_INDEX_TO_STATE.get(pred, GPUState.UNKNOWN), conf

        if self._mode == "nb" and self._nb_model is not None:
            pred  = int(self._nb_model.predict(X)[0])
            proba = self._nb_model.predict_proba(X)[0]
            conf  = float(proba[pred])
            return CLASS_INDEX_TO_STATE.get(pred, GPUState.UNKNOWN), conf

        # Fallback: T4 rules derived from Orange analysis
        return _rule_classify(window.rtheta_mean, window.last_power, window.last_pstate)

    @property
    def mode(self) -> str:
        return self._mode

    def explain(self, window: WindowResult) -> str:
        """Return a human-readable explanation of the classification decision."""
        state, conf = self.classify(window)
        r = window.rtheta_mean
        p = window.last_power
        ps = window.last_pstate

        explanations = {
            GPUState.UNDER_LOAD: (
                f"R_θ={r:.3f} ≤ 0.87 C/W (load threshold) · "
                f"P={p:.1f}W · util={window.last_util:.0f}%"
            ),
            GPUState.ZOMBIE_RECOVERY: (
                f"R_θ={r:.3f} > 0.87 · P-state=P{ps} (P0 = CUDA context retained) · "
                f"P={p:.1f}W at 0% util — CUDA zombie"
            ),
            GPUState.CHILD_EXIT_RECOVERY: (
                f"R_θ={r:.3f} > 1.50 · P-state=P{ps}≥P1 · "
                f"thermal lag after process exit"
            ),
            GPUState.CLEAN_IDLE: (
                f"R_θ={r:.3f} · P={p:.1f}W · P-state=P{ps} · "
                f"low power, temperature settling"
            ),
        }
        reason = explanations.get(state, f"R_θ={r:.3f} P={p:.1f}W PS=P{ps}")
        return f"{state.name} (conf={conf:.2f}) — {reason}"
