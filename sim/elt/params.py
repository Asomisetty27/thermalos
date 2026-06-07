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


def r_sa(airflow_norm: float, r_sa_ref: float) -> float:
    """Convective sink-to-ambient resistance at a given normalised airflow."""
    airflow_norm = max(airflow_norm, 1e-3)   # guard div-by-zero at total fan loss
    return r_sa_ref / (airflow_norm ** CONV_EXPONENT)


def _steady_temp(power_w: float, r_sa_ref: float,
                 r_cond: float = R_COND_CW) -> float:
    """
    Self-consistent steady junction temperature: the fan duty depends on T_j,
    and T_j depends on R_sa(duty). Solve the fixed point T_j = amb + P*R_total(T_j).
    """
    def residual(tj: float) -> float:
        duty = fan_duty(tj)
        r_total = r_cond + r_sa(duty, r_sa_ref)
        return tj - (T_AMBIENT_C + power_w * r_total)
    # T_j is bracketed between ambient and well past any physical throttle point
    return brentq(residual, T_AMBIENT_C, 600.0, xtol=1e-6)


def calibrate_convection() -> float:
    """
    [CALIB] Solve for R_SA_REF so the simulated LOAD operating point reproduces
    the measured load junction temperature (81 C at 68 W). Idle then falls out of
    the same model and its residual is reported by validate.py.
    """
    def load_residual(r_sa_ref: float) -> float:
        return _steady_temp(LOAD_POWER_W, r_sa_ref) - LOAD_TEMP_C
    # R_SA_REF physically positive and small (~0.3 C/W); bracket conservatively
    return brentq(load_residual, 1e-3, 2.0, xtol=1e-9)


# Calibrated once at import — exact load-point match by construction.
R_SA_REF = calibrate_convection()        # [CALIB] convective coefficient


# ─────────────────────────────────────────────────────────────────────────────
# Thermal capacitances  ->  time constants
#   tau_node ~ R_node * C_node.  Set to match observed dynamics:
#   - fast junction response (sub-second), slow heatsink (tens of seconds).
# ─────────────────────────────────────────────────────────────────────────────
C_J_JK = 2.0      # [LIT] die thermal mass  (tau_jc = Rjc*Cj ~ 0.3 s)
C_C_JK = 15.0     # [LIT] case/IHS thermal mass
C_S_JK = 120.0    # [LIT] heatsink base thermal mass (tau_sa ~ Rsa*Cs ~ 60-90 s)


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
    t_amb_c: float      = T_AMBIENT_C
    throttle_c: float   = THROTTLE_TEMP_C
    r_jc: float         = R_JC_CW
    r_ct0: float        = R_CT0_CW
    r_sa_ref: float     = R_SA_REF
    conv_exp: float     = CONV_EXPONENT
    c_j: float          = C_J_JK
    c_c: float          = C_C_JK
    c_s: float          = C_S_JK
    fan_duty_min: float = FAN_DUTY_MIN
    fan_duty_max: float = FAN_DUTY_MAX
    fan_knee_lo: float  = FAN_KNEE_LO_C
    fan_knee_hi: float  = FAN_KNEE_HI_C

    def describe(self) -> str:
        return (
            f"ThermalParams(amb={self.t_amb_c}C throttle={self.throttle_c}C "
            f"Rjc={self.r_jc} Rct0={self.r_ct0} Rsa_ref={self.r_sa_ref:.4f} "
            f"Cj={self.c_j} Cc={self.c_c} Cs={self.c_s})"
        )


DEFAULT = ThermalParams()
