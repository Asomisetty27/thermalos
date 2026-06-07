"""
Physical parameters for the E-LT thermal simulation.

EVERY number here has a provenance tag:
  [STAGE1]  measured directly from raw/experiments/ThermalOS_Measurements_Raw.csv
  [DERIVED] computed from Stage 1 measurements via the relations below
  [CALIB]   solved numerically (see calibrate_convection) to hit a Stage 1 operating point
  [LIT]     literature / engineering value for a ~70 W GPU package + heatsink
  [TESTBED] a knob Sam's heater-block testbed controls; default chosen to match the GPU

The thermal path is modelled as a 3-node Cauer (physical ladder) network:

    P(t) --> [T_j] --Rjc--> [T_c] --Rct(TIM)--> [T_s] --Rsa(airflow)--> T_amb
              C_j             C_c                 C_s

  T_j  junction (die)        — what the GPU sensor reports (integer-quantised)
  T_c  case / IHS
  T_s  heatsink base
  T_amb ambient (boundary)

Steady state reduces to  T_j = T_amb + P*(Rjc + Rct + Rsa),  i.e. R_theta = Rjc+Rct+Rsa,
exactly the quantity Theta computes as (T_j - T_ref)/P.

Degradation modes act on specific resistances:
  TIM dry-out        -> Rct rises
  airflow restriction-> Rsa rises (airflow term throttled)
  fan/pump reduction -> Rsa rises (fan duty capped)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 operating points  (Tesla T4, Google Colab)
# Source: raw/experiments/ThermalOS_Measurements_Raw.csv
# ─────────────────────────────────────────────────────────────────────────────
T_AMBIENT_C        = 25.0     # [STAGE1] ambient_assumed_c column (Colab assumption)
THROTTLE_TEMP_C    = 93.0     # [STAGE1] throttle_temp_c column (T4 thermal limit)

IDLE_POWER_W       = 13.6     # [STAGE1] clean_idle median power_w (13.629)
IDLE_TEMP_C        = 39.0     # [STAGE1] clean_idle temp_c (integer sensor)

LOAD_POWER_W       = 68.0     # [STAGE1] under_load median power_w (~68.0)
LOAD_TEMP_C        = 81.0     # [STAGE1] under_load steady temp_c (~81)

# Derived effective thermal resistances at the two operating points
R_THETA_IDLE = (IDLE_TEMP_C - T_AMBIENT_C) / IDLE_POWER_W   # [DERIVED] ~1.029 C/W
R_THETA_LOAD = (LOAD_TEMP_C - T_AMBIENT_C) / LOAD_POWER_W   # [DERIVED] ~0.824 C/W


# ─────────────────────────────────────────────────────────────────────────────
# Conduction resistances (junction -> case -> heatsink), roughly constant
# ─────────────────────────────────────────────────────────────────────────────
R_JC_CW   = 0.15     # [LIT] junction-to-case, silicon die + solder, ~70 W package
R_CT0_CW  = 0.30     # [LIT] healthy case-to-sink TIM resistance (degradation target)
R_COND_CW = R_JC_CW + R_CT0_CW   # = 0.45  [DERIVED] total healthy conduction


# ─────────────────────────────────────────────────────────────────────────────
# Convection (heatsink -> ambient) with a temperature-following fan curve.
#   airflow_norm in (0,1]; Rsa = R_SA_REF / airflow_norm**CONV_EXPONENT
#   CONV_EXPONENT ~0.8: forced-convection Nusselt ~ Re^0.8 (Dittus-Boelter).
# ─────────────────────────────────────────────────────────────────────────────
CONV_EXPONENT = 0.8          # [LIT] turbulent forced-convection exponent

# Fan curve: duty rises linearly with junction temp between two knees.
FAN_DUTY_MIN   = 0.40        # [TESTBED] idle/floor duty fraction
FAN_DUTY_MAX   = 1.00        # [TESTBED] full duty
FAN_KNEE_LO_C  = 45.0        # [TESTBED] below this -> FAN_DUTY_MIN
FAN_KNEE_HI_C  = 88.0        # [TESTBED] at/above this -> FAN_DUTY_MAX


def fan_duty(temp_j_c: float,
             duty_min: float = FAN_DUTY_MIN,
             duty_max: float = FAN_DUTY_MAX,
             knee_lo: float = FAN_KNEE_LO_C,
             knee_hi: float = FAN_KNEE_HI_C) -> float:
    """GPU auto fan curve: duty fraction as a function of junction temperature."""
    if temp_j_c <= knee_lo:
        return duty_min
    if temp_j_c >= knee_hi:
        return duty_max
    frac = (temp_j_c - knee_lo) / (knee_hi - knee_lo)
    return duty_min + (duty_max - duty_min) * frac


RAD_EMISSIVITY   = 0.80      # [LIT] anodised-aluminium heatsink surface (typical 0.7-0.9)
RAD_AREA_M2      = 0.010     # [LIT] effective radiating area incl. view-factor de-rate
                             #       (small blower-style cooler fin pack totals roughly
                             #        0.02-0.05 m^2; the shroud/enclosure leaves only a
                             #        fraction with a clear view of ambient)
STEFAN_BOLTZMANN = 5.670374e-8   # [LIT] W / (m^2 K^4)

# Natural (buoyancy-driven) convection persists even at zero forced airflow — it
# is the *physical* reason R_sa stays finite, not a numerical floor. Engineering
# range for small finned heatsinks: h_natural ~ 5-10 W/m^2K vs h_forced,max ~
# 30-80 W/m^2K (Incropera free/forced convection coefficient tables for this
# size class), i.e. R_natural is roughly 4-10x the fully-forced reference R.
NATURAL_CONV_RATIO = 6.0     # [LIT] R_natural = NATURAL_CONV_RATIO * R_SA_REF


def q_radiative(t_s_c: float, t_amb_c: float,
                 emissivity: float = RAD_EMISSIVITY,
                 area_m2: float = RAD_AREA_M2) -> float:
    """[LIT] Radiative heat loss from the heatsink surface — Stefan-Boltzmann (T in K)."""
    t_s_k, t_amb_k = t_s_c + 273.15, t_amb_c + 273.15
    return emissivity * STEFAN_BOLTZMANN * area_m2 * (t_s_k**4 - t_amb_k**4)


def r_sa(airflow_norm: float, r_sa_ref: float,
         r_natural: Optional[float] = None) -> float:
    """
    Convective sink-to-ambient resistance. Forced convection (fan-driven, scales
    with airflow^CONV_EXPONENT per Dittus-Boelter turbulent scaling) and natural
    convection (buoyancy-driven, airflow-independent) are parallel heat-loss
    paths, so their *conductances* (1/R) add. This gives a physically-grounded
    finite floor as airflow -> 0 (a stalled fan still cools by natural
    convection) rather than the previous numerical max(airflow, eps) guard.
    """
    if r_natural is None:
        r_natural = NATURAL_CONV_RATIO * r_sa_ref
    airflow_norm = max(airflow_norm, 0.0)
    g_natural = 1.0 / r_natural
    g_forced = (airflow_norm ** CONV_EXPONENT) / r_sa_ref if airflow_norm > 0.0 else 0.0
    return 1.0 / (g_natural + g_forced)


def _steady_temp_full(power_w: float, r_sa_ref: float,
                      r_cond: float = R_COND_CW) -> float:
    """
    Self-consistent steady state with radiative coupling. The same heat flow
    `power_w` crosses the conductive chain in series (junction -> case -> sink),
    but at the sink it splits into a convective branch (linear in dT, through
    R_sa) and a radiative branch (T^4 law) that together must balance it. Solve
    by nested root-find: inner solves T_s given T_j (and the convection/
    radiation split), outer solves T_j so the conductive drop matches.
    """
    def inner_ts_residual(ts: float, rsa: float) -> float:
        q_conv = (ts - T_AMBIENT_C) / rsa
        q_rad = q_radiative(ts, T_AMBIENT_C)
        return (q_conv + q_rad) - power_w

    def outer_tj_residual(tj: float) -> float:
        duty = fan_duty(tj)
        rsa = r_sa(duty, r_sa_ref)
        # NOTE: do not bracket by [T_amb, tj] — during the outer search, trial
        # tj values are not yet self-consistent, so no Ts <= tj may balance the
        # energy equation at that duty. The energy balance alone pins Ts; the
        # outer equation (tj = ts + power*r_cond) is what enforces consistency.
        ts = brentq(lambda ts: inner_ts_residual(ts, rsa),
                    T_AMBIENT_C, 600.0, xtol=1e-7)
        return tj - (ts + power_w * r_cond)

    return brentq(outer_tj_residual, T_AMBIENT_C, 600.0, xtol=1e-6)


def calibrate_convection() -> float:
    """
    [CALIB] Solve for R_SA_REF so the simulated LOAD operating point reproduces
    the measured load junction temperature (81 C at 68 W) — now with radiative
    and natural-convection paths sharing the heat-rejection burden. Forced
    convection only has to supply the *remaining* flow, so R_SA_REF comes out
    larger (less aggressive) than a convection-only fit would require. Idle
    then falls out of the same self-consistent model; its residual is reported
    by validate.py.
    """
    def load_residual(r_sa_ref: float) -> float:
        return _steady_temp_full(LOAD_POWER_W, r_sa_ref) - LOAD_TEMP_C
    # R_SA_REF physically positive; widened bracket since the fitted point
    # shifts up once radiative + natural paths absorb part of the load
    return brentq(load_residual, 1e-3, 4.0, xtol=1e-9)


# Calibrated once at import — exact load-point match by construction, now
# jointly with the radiative + natural-convection heat-loss paths.
R_SA_REF     = calibrate_convection()             # [CALIB] forced-convection reference resistance
R_NATURAL_CW = NATURAL_CONV_RATIO * R_SA_REF      # [DERIVED] natural-convection resistance


# ─────────────────────────────────────────────────────────────────────────────
# Thermal capacitances  ->  time constants
#   tau_node ~ R_node * C_node.  Chosen to match observed dynamics (fast
#   junction response sub-second, slow heatsink tens-of-seconds) AND
#   cross-checked against bounds derived from component geometry + material
#   specific heats below — i.e. these are not free curve-fit constants, they
#   must additionally fall inside a physically-plausible mass*c_p envelope.
# ─────────────────────────────────────────────────────────────────────────────
C_J_JK = 2.0      # [LIT+CHECK] die+package thermal mass (tau_jc = Rjc*Cj ~ 0.3 s)
C_C_JK = 15.0     # [LIT+CHECK] case/IHS thermal mass
C_S_JK = 120.0    # [LIT+CHECK] heatsink base thermal mass (tau_sa ~ Rsa*Cs ~ tens of s)

# Material specific heats [LIT, J/(kg*K)] and density-derived mass ranges for a
# package of this class (~70 W single-slot GPU). Used only as a plausibility
# bound — NOT to back out an exact C (true package geometry is undocumented).
_CP_SILICON_J_KGK = 700.0    # die (silicon)
_CP_COPPER_J_KGK  = 385.0    # IHS / vapor-chamber base (copper or Cu-Ni plated)
_CP_ALUMINIUM_J_KGK = 900.0  # finned heatsink body (aluminium)

# (mass_low_kg, mass_high_kg) engineering ranges for each node's lumped mass,
# from typical component dimensions for a single-slot ~70 W blower-cooled card:
_MASS_RANGE_J = (0.5e-3, 4.0e-3)   # die + solder + substrate sliver, grams-scale
_MASS_RANGE_C = (10.0e-3, 40.0e-3) # lumped "case" node: IHS + mounting plate +
                                    # thermal-pad backing + fasteners, tens-of-grams
_MASS_RANGE_S = (0.08, 0.35)       # finned heatsink body, ~100-350 g


def capacitance_plausibility_range() -> dict[str, tuple[float, float]]:
    """
    [DERIVED] Physically-plausible C ranges (J/K) computed as mass_range * c_p
    for each node's dominant material. Returns {"C_j": (lo, hi), ...} so
    validate.py can assert the chosen C_* values are not just curve-fit
    constants but sit inside a range a real component's mass and material
    would produce.
    """
    return {
        "C_j": (_MASS_RANGE_J[0] * _CP_SILICON_J_KGK,   _MASS_RANGE_J[1] * _CP_SILICON_J_KGK),
        "C_c": (_MASS_RANGE_C[0] * _CP_COPPER_J_KGK,    _MASS_RANGE_C[1] * _CP_COPPER_J_KGK),
        "C_s": (_MASS_RANGE_S[0] * _CP_ALUMINIUM_J_KGK, _MASS_RANGE_S[1] * _CP_ALUMINIUM_J_KGK),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fan rotor dynamics — a commanded duty change does not produce an instantaneous
# airflow change. Rotor inertia + motor torque give first-order spin-up/down
# lag. Engineering range for small (40-90mm) high-speed blower fans responding
# to a PWM step: ~0.5-3 s to settle near the new operating point.
# ─────────────────────────────────────────────────────────────────────────────
FAN_TAU_S = 1.5      # [LIT] first-order airflow response time constant (s)


# ─────────────────────────────────────────────────────────────────────────────
# Arrhenius kinetics for thermally-activated TIM degradation (pump-out / dry-out
# are diffusion- and oxidation-driven processes whose rates follow the Arrhenius
# law). This is what makes TIM dry-out *emergent* rather than an externally
# imposed schedule: hotter TIM degrades faster, which raises R_ct, which raises
# local temperature further — a real positive-feedback mechanism a prescribed
# ramp cannot represent.
#
#   dx/dt = k(T) * (1 - x),   k(T) = k_ref * exp[-(Ea/Rgas)*(1/T - 1/T_ref)]
#
# x in [0,1] is degradation progress; R_ct(t) = R_ct0 * (1 + (severity-1)*x(t)).
# T is the case/TIM-interface temperature T_c (where the material actually sits).
# ─────────────────────────────────────────────────────────────────────────────
GAS_CONSTANT_J_MOLK = 8.314462618   # [LIT] universal gas constant
TIM_ACTIVATION_ENERGY_J_MOL = 75_000.0   # [LIT] polymer/grease pump-out Ea, accelerated-aging
                                          #       literature commonly reports 50-100 kJ/mol
TIM_T_REF_K = (LOAD_TEMP_C - 1.0) + 273.15   # [DERIVED] reference ~ T_c at sustained load
TIM_K_REF_PER_S = 1.0 / (2.0 * 3600.0)       # [LIT] rate constant at T_ref: ~2 h to 63% progress
                                              #       (chosen so unaccelerated dry-out lands in the
                                              #        "hours" band the protocol expects to be useful)


def tim_arrhenius_rate(t_c_celsius: float,
                       ea_j_mol: float = TIM_ACTIVATION_ENERGY_J_MOL,
                       k_ref: float = TIM_K_REF_PER_S,
                       t_ref_k: float = TIM_T_REF_K) -> float:
    """[LIT] Temperature-dependent TIM degradation rate constant k(T), Arrhenius form."""
    t_k = t_c_celsius + 273.15
    return k_ref * np.exp(-(ea_j_mol / GAS_CONSTANT_J_MOLK) * (1.0 / t_k - 1.0 / t_ref_k))


# ─────────────────────────────────────────────────────────────────────────────
# Sensor model — the detector only ever sees these (NOT the true state)
# ─────────────────────────────────────────────────────────────────────────────
TEMP_QUANT_C      = 1.0    # [STAGE1] junction temp reported as integer degrees
TEMP_NOISE_C      = 0.3    # [LIT] sub-degree sensor noise (1-sigma, pre-quantisation)
POWER_NOISE_W     = 0.5    # [STAGE1] power_w jitter observed (~+/-0.5 W, 1-sigma)
SAMPLE_PERIOD_S   = 1.0    # [STAGE1] per-second telemetry cadence


# ─────────────────────────────────────────────────────────────────────────────
# Throttle behaviour: once T_j >= THROTTLE_TEMP_C the GPU clock-limits, reducing
# effective power to hold the junction near the limit (thermal governor).
# ─────────────────────────────────────────────────────────────────────────────
THROTTLE_HYSTERESIS_C = 1.0    # [LIT] re-engage band
THROTTLE_POWER_FLOOR  = 0.55   # [LIT] fraction of demanded power under hard throttle


@dataclass(frozen=True)
class ThermalParams:
    """Immutable bundle of the calibrated physical parameters for one GPU/testbed."""
    t_amb_c: float       = T_AMBIENT_C
    throttle_c: float    = THROTTLE_TEMP_C
    r_jc: float          = R_JC_CW
    r_ct0: float         = R_CT0_CW
    r_sa_ref: float      = R_SA_REF
    r_natural: float     = R_NATURAL_CW
    conv_exp: float      = CONV_EXPONENT
    rad_emissivity: float = RAD_EMISSIVITY
    rad_area_m2: float   = RAD_AREA_M2
    c_j: float           = C_J_JK
    c_c: float           = C_C_JK
    c_s: float           = C_S_JK
    fan_duty_min: float  = FAN_DUTY_MIN
    fan_duty_max: float  = FAN_DUTY_MAX
    fan_knee_lo: float   = FAN_KNEE_LO_C
    fan_knee_hi: float   = FAN_KNEE_HI_C
    fan_tau_s: float     = FAN_TAU_S

    def describe(self) -> str:
        return (
            f"ThermalParams(amb={self.t_amb_c}C throttle={self.throttle_c}C "
            f"Rjc={self.r_jc} Rct0={self.r_ct0} Rsa_ref={self.r_sa_ref:.4f} "
            f"Rnat={self.r_natural:.4f} rad(eps={self.rad_emissivity},A={self.rad_area_m2}) "
            f"Cj={self.c_j} Cc={self.c_c} Cs={self.c_s} fan_tau={self.fan_tau_s}s)"
        )


DEFAULT = ThermalParams()
