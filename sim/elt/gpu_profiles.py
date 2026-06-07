"""
Cross-vendor R_theta baseline prediction — B200 / H100 / A100 / MI300X.

Answers the question raised in the vault (wiki/synthesis/ai_factory_readiness_gap.md):
*can the GPU-agnostic E-LT thermal model be re-pointed at non-T4 hardware to predict
what `theta calibrate` would measure, ahead of physical access?*

Method (a scaling-law derivation, NOT a fitted-to-data calibration like params.DEFAULT):
  1. Anchor on published TDP and a thermal-design-target operating temperature
     (vendors size cooling so sustained full-load T_j sits some margin below the
     hardware throttle limit — the "design margin" below is an engineering estimate,
     not a spec value).
  2. That anchor pins the *total* load R_theta = (T_j_target - T_amb) / TDP.
  3. Split the total across the three Cauer resistances (R_jc : R_ct0 : R_sa) in the
     same *proportions* the T4 calibration found — i.e. assume a "geometrically
     similar cooling solution, scaled for power," which is the standard engineering
     starting point when no unit-specific data exists yet.
  4. Scale thermal capacitances by an empirical package-mass power-law in TDP
     (bigger chips get bigger heatsinks, but not linearly — mass grows slower than
     power because density and packaging efficiency improve with each generation).

EVERY number below carries the same provenance convention as params.py:
  [LIT-pub]  published vendor spec (TDP) — solid
  [EST]      engineering estimate of a design target not publicly specified
  [SCALED]   derived from a T4 ratio + a scaling assumption — the weakest link

> CONFLICT-AVOIDANCE: outputs of this module are PREDICTIONS TO CONFIRM, exactly
> the same epistemic status as the E-LT lead-time numbers in elt_simulation — i.e.
> "physically plausible, not validated." They do not resolve Q_cross_vendor_calibration;
> only `theta calibrate` run against real silicon does that. See
> wiki/synthesis/ai_factory_readiness_gap.md for the full framing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from . import params as P
from . import degradation as deg
from .detector import (
    DetectorConfig, apply_sensor_model, windowed_rtheta, fit_baseline, detect_anomaly,
)
from .thermal_model import simulate


# ─────────────────────────────────────────────────────────────────────────────
# Published specs  [LIT-pub] TDP is a vendor datasheet number — solid.
# Junction throttle limits for datacenter parts are rarely published; the values
# below are [EST] drawn from the same 83-95 C band NVIDIA/AMD datacenter silicon
# has occupied across the last three generations (T4's own limit, 93 C, anchors
# the low end of that band).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GPUSpec:
    name:           str
    vendor:         str
    tdp_w:          float   # [LIT-pub] sustained full-load power draw
    idle_w:         float   # [EST] big datacenter dies idle far higher than T4's 13.6 W —
                            #       active memory controllers, NVLink/Infinity Fabric links,
                            #       and management cores keep a power floor regardless of compute load
    throttle_c:     float   # [EST] junction thermal limit
    design_margin_c: float  # [EST] vendor's target headroom below throttle at sustained TDP load
    cooling:        str     # "air" | "liquid" — informs which T4 ratio assumptions still apply
    note:           str = ""


# Ambient is held at the same 25 C the T4/Colab baseline used — datacenter cold-aisle
# targets are often a couple degrees warmer (27-30 C), but holding it constant keeps
# this an apples-to-apples R_theta comparison rather than conflating two unknowns.
T_AMBIENT_C = P.T_AMBIENT_C   # [STAGE1] 25.0 — held constant for comparability, see note above

GPU_SPECS: dict[str, GPUSpec] = {
    "T4": GPUSpec(   # the calibration anchor — included for direct ratio comparison
        name="Tesla T4", vendor="NVIDIA", tdp_w=P.LOAD_POWER_W, idle_w=P.IDLE_POWER_W,
        throttle_c=P.THROTTLE_TEMP_C, design_margin_c=P.THROTTLE_TEMP_C - P.LOAD_TEMP_C,
        cooling="air",
        note="[STAGE1] reference point — not predicted, measured (E001-E004)",
    ),
    "A100": GPUSpec(
        name="A100 SXM4 80GB", vendor="NVIDIA", tdp_w=400.0, idle_w=55.0,
        throttle_c=89.0, design_margin_c=6.0, cooling="liquid",
        note="[LIT-pub] 400 W TDP (Ampere datasheet); [EST] throttle ~89 C, "
             "idle floor ~55 W (large HBM2e + NVLink links keep idle well above T4's 13.6 W)",
    ),
    "H100": GPUSpec(
        name="H100 SXM5 80GB", vendor="NVIDIA", tdp_w=700.0, idle_w=70.0,
        throttle_c=87.0, design_margin_c=5.0, cooling="liquid",
        note="[LIT-pub] 700 W TDP (Hopper datasheet, SXM5); [EST] throttle ~87 C — "
             "tighter design margin than air-cooled parts because liquid loops run "
             "closer to their limit by design (smaller delta-T budget, higher flow rates)",
    ),
    "B200": GPUSpec(
        name="B200 (Blackwell, DGX/HGX)", vendor="NVIDIA", tdp_w=1000.0, idle_w=110.0,
        throttle_c=85.0, design_margin_c=5.0, cooling="liquid",
        note="[LIT-pub] ~1000 W TDP (Blackwell datasheet, dual-die package); [EST] "
             "throttle ~85 C, idle ~110 W — the dual-reticle die and always-on "
             "NVLink-5 fabric are the largest idle-floor assumptions in this table. "
             "DIRECTLY RELEVANT: this is the Cal Poly AI Factory's hardware (4x DGX B200 nodes).",
    ),
    "MI300X": GPUSpec(
        name="MI300X", vendor="AMD", tdp_w=750.0, idle_w=90.0,
        throttle_c=100.0, design_margin_c=8.0, cooling="liquid",
        note="[LIT-pub] 750 W TDP (CDNA3 datasheet); [EST] throttle ~100 C — AMD's "
             "publicly documented HBM-stacked junction limits run ~10-15 C above "
             "NVIDIA's equivalent-class parts (different process/packaging choices); "
             "idle ~90 W for the CDNA3 chiplet complex + 8-stack HBM3. "
             "WIDEST error bars in this table — least public thermal documentation of the four.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# T4 calibrated ratios — what we scale from. These three numbers fully determine
# how the predicted total R_theta gets split across the Cauer chain for every
# other GPU (step 3 in the module docstring).
# ─────────────────────────────────────────────────────────────────────────────
_T4_R_TOTAL = P.R_JC_CW + P.R_CT0_CW + P.R_SA_REF       # total conductive+convective at the anchor
_T4_FRAC_JC = P.R_JC_CW / _T4_R_TOTAL                    # [DERIVED] ~0.12  junction->case share
_T4_FRAC_CT = P.R_CT0_CW / _T4_R_TOTAL                   # [DERIVED] ~0.25  TIM share
_T4_FRAC_SA = P.R_SA_REF / _T4_R_TOTAL                   # [DERIVED] ~0.63  convection share

# Capacitance scaling exponent: package thermal mass grows sub-linearly with TDP
# (each generation packs more power into proportionally less material via better
# vapor chambers, denser fin stacks, direct-die liquid cold plates). 0.6 is a
# conservative mid-point of the 0.5-0.7 range typical of "scale factor" rules
# used in electronics-cooling sizing heuristics. [EST] — the single biggest lever
# on predicted *lead times* (it sets the system time constants), and the part of
# this model with the least empirical grounding.
_MASS_SCALE_EXPONENT = 0.6


def derive_thermal_params(spec: GPUSpec) -> P.ThermalParams:
    """
    [SCALED] Build a ThermalParams for `spec` by anchoring total load R_theta to
    the spec's (throttle - margin - ambient)/TDP operating point, splitting it in
    T4's calibrated proportions, and scaling capacitances by a TDP power-law.

    Returns a params object the existing thermal_model.simulate() consumes
    unmodified — the Cauer network and ODE integration are GPU-agnostic by
    construction (this is the structural claim ai_factory_readiness_gap.md makes).
    """
    t_load_target = spec.throttle_c - spec.design_margin_c
    r_total_load  = (t_load_target - T_AMBIENT_C) / spec.tdp_w

    r_jc  = r_total_load * _T4_FRAC_JC
    r_ct0 = r_total_load * _T4_FRAC_CT
    r_sa_ref = _solve_r_sa_ref_for_target(
        target_tj=t_load_target, power_w=spec.tdp_w, r_cond=r_jc + r_ct0,
    )

    mass_ratio = (spec.tdp_w / P.LOAD_POWER_W) ** _MASS_SCALE_EXPONENT
    return replace(
        P.DEFAULT,
        t_amb_c=T_AMBIENT_C,
        throttle_c=spec.throttle_c,
        r_jc=r_jc, r_ct0=r_ct0, r_sa_ref=r_sa_ref,
        r_natural=P.NATURAL_CONV_RATIO * r_sa_ref,
        c_j=P.C_J_JK * mass_ratio, c_c=P.C_C_JK * mass_ratio, c_s=P.C_S_JK * mass_ratio,
    )


def _steady_temp_generic(power_w: float, prm: P.ThermalParams) -> float:
    """Self-consistent steady T_j for an arbitrary ThermalParams (generic form of
    params._steady_temp_full, which is hard-wired to the T4 module constants)."""
    # Upper bracket is intentionally generous (not a physical ceiling): the search
    # explores r_sa_ref guesses that may correspond to "absurdly poor cooling for
    # this TDP" candidates, which only balance energy at very high (non-physical)
    # temperatures. T^4 radiation guarantees a sign change is found well before
    # this ceiling for any candidate that brentq needs to evaluate.
    _T_CEILING = 4000.0

    def inner_ts_residual(ts: float, rsa: float) -> float:
        q_conv = (ts - prm.t_amb_c) / rsa
        q_rad  = P.q_radiative(ts, prm.t_amb_c, prm.rad_emissivity, prm.rad_area_m2)
        return (q_conv + q_rad) - power_w

    def outer_tj_residual(tj: float) -> float:
        duty = P.fan_duty(tj, prm.fan_duty_min, prm.fan_duty_max, prm.fan_knee_lo, prm.fan_knee_hi)
        rsa  = P.r_sa(duty, prm.r_sa_ref, prm.r_natural)
        ts = brentq(lambda ts: inner_ts_residual(ts, rsa), prm.t_amb_c, _T_CEILING, xtol=1e-7)
        return tj - (ts + power_w * (prm.r_jc + prm.r_ct0))

    return brentq(outer_tj_residual, prm.t_amb_c, _T_CEILING, xtol=1e-6)


def _solve_r_sa_ref_for_target(target_tj: float, power_w: float, r_cond: float) -> float:
    """[CALIB-style] Solve r_sa_ref so the full self-consistent steady state lands
    on `target_tj` at `power_w` — the same nested-brentq approach params.py uses
    to fit the T4, just parameterized for an arbitrary spec."""
    probe = replace(P.DEFAULT, r_jc=0.0, r_ct0=r_cond)  # r_cond folded into r_ct0 for the probe

    def residual(r_sa_ref: float) -> float:
        prm = replace(probe, r_sa_ref=r_sa_ref, r_natural=P.NATURAL_CONV_RATIO * r_sa_ref)
        return _steady_temp_generic(power_w, prm) - target_tj

    return brentq(residual, 1e-4, 8.0, xtol=1e-9)


@dataclass(frozen=True)
class PredictedBaseline:
    spec_name:    str
    rtheta_idle:  float
    rtheta_load:  float
    t_idle_c:     float
    t_load_c:     float
    params:       P.ThermalParams


def predict_baseline(spec: GPUSpec) -> PredictedBaseline:
    """Run the full derivation + steady-state solve and return the predicted
    two-point R_theta fingerprint — i.e. what `theta calibrate` would most likely
    measure on first contact with this hardware."""
    prm = derive_thermal_params(spec)
    t_load = _steady_temp_generic(spec.tdp_w, prm)
    t_idle = _steady_temp_generic(spec.idle_w, prm)
    return PredictedBaseline(
        spec_name=spec.name,
        rtheta_idle=(t_idle - prm.t_amb_c) / spec.idle_w,
        rtheta_load=(t_load - prm.t_amb_c) / spec.tdp_w,
        t_idle_c=t_idle, t_load_c=t_load,
        params=prm,
    )


def predict_all() -> dict[str, PredictedBaseline]:
    return {key: predict_baseline(spec) for key, spec in GPU_SPECS.items()}


def format_table(predictions: dict[str, PredictedBaseline]) -> str:
    lines = [
        f"{'GPU':<22} {'TDP(W)':>7} {'R_th idle':>10} {'R_th load':>10} "
        f"{'T_idle':>7} {'T_load':>7}  {'cooling':<7}",
        "-" * 78,
    ]
    for key, pb in predictions.items():
        spec = GPU_SPECS[key]
        lines.append(
            f"{pb.spec_name:<22} {spec.tdp_w:>7.0f} {pb.rtheta_idle:>10.4f} "
            f"{pb.rtheta_load:>10.4f} {pb.t_idle_c:>6.1f}C {pb.t_load_c:>6.1f}C  {spec.cooling:<7}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-vendor lead-time prediction — generic counterpart to experiment.run_trial/
# run_monte_carlo, which are hard-wired to params.DEFAULT (T4). Reuses the exact
# same Cauer integration + sensor model + windowed-R_theta + baseline + anomaly-
# detection pipeline; only the physical params and workload power differ.
#
# This is the "lead time" half of the prediction: not just "what would the
# fingerprint look like" but "if Theta were watching this GPU with thresholds
# derived from THAT fingerprint, how much warning would it give before a TIM
# dry-out throttles it?" — directly extends elt_simulation's T4 answer to the
# other three GPU classes.
# ─────────────────────────────────────────────────────────────────────────────
TIM_HORIZON_S   = deg.DEFAULT_HORIZON_S["tim"]
TIM_BASELINE_S  = 600.0
TIM_SEVERITY    = 2.4   # [PROTOCOL] same severity elt_simulation uses for the T4 "gradual" arm


def _jitter_spec_params(rng: np.random.Generator, prm: P.ThermalParams,
                        frac: float = 0.08) -> P.ThermalParams:
    """Same +/-8% unit-to-unit jitter experiment._jitter_params applies, just
    centered on the SPEC's derived params instead of the T4 module constants."""
    def j(x):
        return x * (1.0 + rng.normal(0.0, frac))
    return replace(prm, r_jc=j(prm.r_jc), r_ct0=j(prm.r_ct0), r_sa_ref=j(prm.r_sa_ref),
                   c_j=j(prm.c_j), c_c=j(prm.c_c), c_s=j(prm.c_s))


def run_cross_vendor_trial(spec: GPUSpec, prm: P.ThermalParams, seed: int,
                           cfg: Optional["DetectorConfig"] = None,
                           jitter: bool = True):
    """One TIM-dryout (gradual) trial against `spec`'s derived thermal params.
    Returns (t_throttle, lead_time_dict) — mirrors experiment.TrialResult's
    fields that matter for Monte Carlo aggregation."""
    cfg = cfg or DetectorConfig()
    rng = np.random.default_rng(seed)
    prm_run = _jitter_spec_params(rng, prm) if jitter else prm

    sev = TIM_SEVERITY * (1.0 + rng.normal(0.0, 0.06)) if jitter else TIM_SEVERITY
    scn, _ = deg.tim_degradation(duration_s=TIM_HORIZON_S, baseline_s=TIM_BASELINE_S,
                                 severity=sev, variant="gradual", workload_w=spec.tdp_w)

    sim = simulate(scn, prm_run)
    sensed = apply_sensor_model(sim, rng, ambient_mode="true")
    rtheta, stable = windowed_rtheta(sensed, cfg)
    base = fit_baseline(rtheta, stable, sensed.t, TIM_BASELINE_S)

    lead_times = {}
    for k in cfg.k_values:
        res = detect_anomaly(rtheta, stable, sensed.t, base, k, cfg)
        if res.t_anomaly is not None and sim.t_throttle is not None:
            lead_times[k] = sim.t_throttle - res.t_anomaly
        else:
            lead_times[k] = None
    return sim.t_throttle, lead_times, base


@dataclass(frozen=True)
class CrossVendorMC:
    spec_name: str
    n_trials:  int
    k_values:  tuple
    detect_rate: dict      # k -> fraction detected before throttle
    median_lead_s: dict    # k -> median lead time (s) where detected
    throttle_rate: float   # fraction of trials that ever throttled


def run_cross_vendor_mc(spec: GPUSpec, n_trials: int = 30, base_seed: int = 5000,
                        cfg: Optional["DetectorConfig"] = None) -> CrossVendorMC:
    """Monte Carlo TIM-dryout lead-time prediction for one GPU spec."""
    cfg = cfg or DetectorConfig()
    prm = derive_thermal_params(spec)
    per_k_lt = {k: [] for k in cfg.k_values}
    per_k_detect = {k: 0 for k in cfg.k_values}
    n_throttled = 0

    for i in range(n_trials):
        t_throttle, lead_times, _ = run_cross_vendor_trial(spec, prm, seed=base_seed + i, cfg=cfg)
        if t_throttle is not None:
            n_throttled += 1
        for k in cfg.k_values:
            lt = lead_times[k]
            if lt is not None and lt > 0:
                per_k_lt[k].append(lt)
                per_k_detect[k] += 1

    return CrossVendorMC(
        spec_name=spec.name, n_trials=n_trials, k_values=cfg.k_values,
        detect_rate={k: per_k_detect[k] / n_trials for k in cfg.k_values},
        median_lead_s={k: (float(np.median(v)) if v else None) for k, v in per_k_lt.items()},
        throttle_rate=n_throttled / n_trials,
    )


def format_mc_table(results: dict[str, CrossVendorMC]) -> str:
    lines = []
    for key, mc in results.items():
        lines.append(f"\n{mc.spec_name}  (TIM dry-out, gradual, severity~{TIM_SEVERITY}x, N={mc.n_trials}, throttle_rate={mc.throttle_rate:.0%})")
        lines.append(f"  {'k':>5}  {'detect%':>8}  {'median lead':>14}")
        for k in mc.k_values:
            med = mc.median_lead_s[k]
            med_s = f"{med:.0f}s ({med/60:.1f} min)" if med is not None else "—"
            lines.append(f"  {k:>5g}  {mc.detect_rate[k]*100:>7.0f}%  {med_s:>14}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    preds = predict_all()
    print(format_table(preds))
    print()
    for key, pb in preds.items():
        if key == "T4":
            continue
        print(f"{pb.spec_name}: {pb.params.describe()}")

    if "--mc" in sys.argv:
        n = 30
        print(f"\n\n=== TIM dry-out Monte Carlo, N={n} per GPU (this takes a few minutes) ===")
        mc_results = {}
        for key, spec in GPU_SPECS.items():
            print(f"  running {spec.name}...", flush=True)
            mc_results[key] = run_cross_vendor_mc(spec, n_trials=n)
        print(format_mc_table(mc_results))
