"""
E-LT experiment runner: one trial end-to-end, and Monte Carlo over many trials.

One trial:
    build scenario -> integrate thermal ODE -> sensor model -> windowed R_theta
    -> fit healthy baseline -> sweep k -> lead_time = t_throttle - t_anomaly

Monte Carlo jitters the physical parameters, degradation severity/onset, ambient
drift and sensor noise across N trials to produce the lead-time DISTRIBUTION the
protocol asks for ("mean, std, min, max" and "median X before throttle, N trials").
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from . import params as P
from . import degradation as deg
from .thermal_model import simulate, SimResult
from .detector import (
    DetectorConfig, apply_sensor_model, windowed_rtheta,
    fit_baseline, detect_anomaly, Baseline,
)


@dataclass
class TrialResult:
    mode: str
    variant: str
    seed: int
    ambient_mode: str
    t_throttle: Optional[float]
    baseline: Baseline
    lead_times: dict          # k -> lead_time seconds (None if no detection / no throttle)
    t_anomaly: dict           # k -> t_anomaly seconds
    sim: Optional[SimResult] = field(default=None, repr=False)   # kept only when asked
    sensed: object = field(default=None, repr=False)
    rtheta: Optional[np.ndarray] = field(default=None, repr=False)
    stable: Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter jitter for Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
def _jitter_params(rng: np.random.Generator, frac: float = 0.08) -> P.ThermalParams:
    """Perturb physical params by +/- frac (1-sigma) — unit-to-unit variation."""
    def j(x):
        return x * (1.0 + rng.normal(0.0, frac))
    return replace(
        P.DEFAULT,
        r_jc=j(P.R_JC_CW), r_ct0=j(P.R_CT0_CW), r_sa_ref=j(P.R_SA_REF),
        c_j=j(P.C_J_JK), c_c=j(P.C_C_JK), c_s=j(P.C_S_JK),
    )


def _build_scenario(mode: str, variant: str, duration_s: float, baseline_s: float,
                    rng: np.random.Generator, jitter: bool):
    builder = deg.MODE_BUILDERS[mode]
    # default severities per mode (chosen to cross throttle)
    sev_default = {"tim": 2.4, "tim_arrhenius": 2.4, "airflow": 0.45, "fan": 0.40}[mode]
    sev = sev_default
    if jitter:
        # jitter severity ~5%, keeping airflow/fan caps in (0,1)
        if mode in ("tim", "tim_arrhenius"):
            sev = sev_default * (1.0 + rng.normal(0.0, 0.06))
        else:
            sev = float(np.clip(sev_default * (1.0 + rng.normal(0.0, 0.06)), 0.05, 0.95))
    # tim_arrhenius has only one variant ("emergent") — the trial harness's
    # default 'gradual'/'step' selection doesn't apply to an emergent process.
    eff_variant = "emergent" if mode == "tim_arrhenius" else variant
    scn, spec = builder(duration_s=duration_s, baseline_s=baseline_s,
                        severity=sev, variant=eff_variant)
    return scn, spec


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(mode: str, variant: str = "gradual",
              duration_s: Optional[float] = None, baseline_s: float = 600.0,
              seed: int = 0, ambient_mode: str = "true",
              ambient_drift_c_per_hr: float = 0.0,
              cfg: Optional[DetectorConfig] = None,
              jitter: bool = False, keep_traces: bool = False) -> TrialResult:
    """Run one E-LT trial and return lead times for each k."""
    cfg = cfg or DetectorConfig()
    rng = np.random.default_rng(seed)
    duration_s = duration_s or deg.DEFAULT_HORIZON_S[mode]

    prm = _jitter_params(rng) if jitter else P.DEFAULT
    scn, _spec = _build_scenario(mode, variant, duration_s, baseline_s, rng, jitter)

    # Integrate with the (possibly jittered) params
    sim = simulate(scn, prm)

    # Observe + detect
    sensed = apply_sensor_model(sim, rng, ambient_mode=ambient_mode,
                                ambient_drift_c_per_hr=ambient_drift_c_per_hr)
    rtheta, stable = windowed_rtheta(sensed, cfg)
    base = fit_baseline(rtheta, stable, sensed.t, baseline_s)

    lead_times, t_anoms = {}, {}
    for k in cfg.k_values:
        res = detect_anomaly(rtheta, stable, sensed.t, base, k, cfg)
        t_anoms[k] = res.t_anomaly
        if res.t_anomaly is not None and sim.t_throttle is not None:
            lead_times[k] = sim.t_throttle - res.t_anomaly
        else:
            lead_times[k] = None

    return TrialResult(
        mode=mode, variant=variant, seed=seed, ambient_mode=ambient_mode,
        t_throttle=sim.t_throttle, baseline=base,
        lead_times=lead_times, t_anomaly=t_anoms,
        sim=sim if keep_traces else None,
        sensed=sensed if keep_traces else None,
        rtheta=rtheta if keep_traces else None,
        stable=stable if keep_traces else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MonteCarloResult:
    mode: str
    variant: str
    ambient_mode: str
    n_trials: int
    k_values: tuple
    lead_times: dict          # k -> np.ndarray of lead times (s), None entries dropped
    detect_rate: dict         # k -> fraction of trials that detected before throttle
    throttle_times: np.ndarray
    trials: list = field(default_factory=list, repr=False)

    def summary(self, k: float) -> dict:
        lt = self.lead_times.get(k, np.array([]))
        if lt.size == 0:
            return {"k": k, "n": 0, "detect_rate": self.detect_rate.get(k, 0.0)}
        return {
            "k": k, "n": int(lt.size),
            "detect_rate": self.detect_rate.get(k, 0.0),
            "mean_s": float(np.mean(lt)), "std_s": float(np.std(lt)),
            "median_s": float(np.median(lt)),
            "min_s": float(np.min(lt)), "max_s": float(np.max(lt)),
            "p10_s": float(np.percentile(lt, 10)),
            "p90_s": float(np.percentile(lt, 90)),
        }


def run_monte_carlo(mode: str, variant: str = "gradual", n_trials: int = 50,
                    duration_s: Optional[float] = None, baseline_s: float = 600.0,
                    ambient_mode: str = "true", ambient_drift_c_per_hr: float = 0.0,
                    cfg: Optional[DetectorConfig] = None,
                    base_seed: int = 1000, keep_first_trace: bool = True) -> MonteCarloResult:
    """Run N jittered trials of one degradation arm; aggregate lead-time stats."""
    cfg = cfg or DetectorConfig()
    trials: list[TrialResult] = []
    per_k_lt = {k: [] for k in cfg.k_values}
    per_k_detect = {k: 0 for k in cfg.k_values}
    throttle_times = []

    for i in range(n_trials):
        keep = keep_first_trace and i == 0
        tr = run_trial(mode, variant=variant, duration_s=duration_s,
                       baseline_s=baseline_s, seed=base_seed + i,
                       ambient_mode=ambient_mode,
                       ambient_drift_c_per_hr=ambient_drift_c_per_hr,
                       cfg=cfg, jitter=True, keep_traces=keep)
        trials.append(tr)
        if tr.t_throttle is not None:
            throttle_times.append(tr.t_throttle)
        for k in cfg.k_values:
            lt = tr.lead_times[k]
            if lt is not None and lt > 0:
                per_k_lt[k].append(lt)
                per_k_detect[k] += 1

    lead_times = {k: np.array(v) for k, v in per_k_lt.items()}
    detect_rate = {k: per_k_detect[k] / n_trials for k in cfg.k_values}

    return MonteCarloResult(
        mode=mode, variant=variant, ambient_mode=ambient_mode,
        n_trials=n_trials, k_values=cfg.k_values,
        lead_times=lead_times, detect_rate=detect_rate,
        throttle_times=np.array(throttle_times), trials=trials,
    )
