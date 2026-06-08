"""
Hardware profile registry — per-GPU-class thermal priors.

This is the single source of truth for "what we expect from this silicon"
before any per-unit calibration happens. It solves three audit findings at once:

  1. Cold-start T_ref bias — instead of defaulting to 25 °C for every GPU,
     we seed from a hardware-class-appropriate value (~22 °C inlet for
     hot-aisle GPU servers, ~28 °C for retrofitted air-cooled racks) AND
     emit an uncertainty band so downstream knows the seed is provisional.

  2. Silent T4-default misclassification — the classifier's hard-coded
     R_theta thresholds (0.87 load, 1.50 idle) are valid ONLY for Tesla T4.
     This module exposes per-class threshold seeds so an H100 fleet without
     calibration still gets reasonable detection until `theta calibrate`
     produces unit-level numbers.

  3. Multi-vendor scaffolding — AMD MI300, Intel Gaudi, future TPUs all
     have entries here. Even when the collector for that vendor is a stub,
     the profile tells the rest of the agent what to expect.

Numbers are sourced from:
  - NVIDIA datasheets (TDP, max temp, peak boost clock)
  - Public scaling-law estimates from Stage 1 T4 data extrapolated to
    larger dies (see wiki/synthesis/cross_vendor_thermal_predictions.md)
  - SemiAnalysis / TechInsights reports for HBM stack thermals
  - First-hand thermal-resistance measurements where available

ALL numbers are TYPED AS PRIORS. A locked calibration always wins over a
profile seed; the profile is the bootstrap value before that lock arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ThermalProfile:
    """Per-hardware-class thermal expectations.

    Use these as cold-start priors. They are NOT calibration; they are the
    'we don't know this specific unit yet, so here's what the class looks
    like on average' fallback.
    """

    # Identity
    family:           str    # "ampere", "ada", "hopper", "blackwell", "cdna3", "gaudi3"
    canonical_name:   str    # "A100-SXM4-80GB"
    vendor:           str    # "nvidia", "amd", "intel"

    # Power envelope (W)
    tdp_w:            float  # nameplate TDP
    idle_floor_w:     float  # typical idle power draw

    # Thermal envelope (°C)
    junction_max_c:   float  # T_j throttle point (NVML reports temp ≤ this)
    expected_ambient_c: float  # typical inlet temp for this class's deployment

    # R_theta priors (°C/W) — junction-to-ambient effective thermal resistance
    #   load_threshold:  R_theta below this implies UNDER_LOAD (busy, healthy)
    #   idle_threshold:  R_theta above this implies stuck/zombie (cooling
    #                    overheads dominate a small power signal)
    #   These are interpolation seeds; calibration replaces them per-unit.
    rtheta_load_threshold: float
    rtheta_idle_threshold: float
    rtheta_expected_under_load: float  # the typical R_theta during normal work
    rtheta_expected_idle:       float  # the typical R_theta when truly idle

    # Drift detection priors
    rtheta_drift_warn_c_per_day: float  # slow drift that warrants a notice
    rtheta_drift_crit_c_per_day: float  # fast drift that warrants an alert

    # Cooling architecture (affects fault classification priors)
    cooling: str  # "air-blower", "air-passive", "liquid-cold-plate", "immersion"

    # Confidence in this profile
    #   "measured":     derived from first-party Stage 1+ data
    #   "extrapolated": physics-based scaling from a measured family member
    #   "datasheet":    inferred from vendor datasheets only (not validated)
    confidence: str  # "measured" | "extrapolated" | "datasheet"

    # Notes for operators
    notes: tuple[str, ...] = field(default_factory=tuple)


# ──────────────────────────────────────────────────────────────────────────
# Registry — keyed by normalized GPU name (lower-cased, stripped of "Tesla "
# and "NVIDIA " prefixes for matching purposes).
# ──────────────────────────────────────────────────────────────────────────

_PROFILES: dict[str, ThermalProfile] = {
    # ── NVIDIA — Tesla T4 (Stage 1 baseline; the only MEASURED profile) ──
    "t4": ThermalProfile(
        family="turing",
        canonical_name="Tesla T4",
        vendor="nvidia",
        tdp_w=70.0,
        idle_floor_w=11.0,
        junction_max_c=85.0,
        expected_ambient_c=22.0,
        rtheta_load_threshold=0.87,
        rtheta_idle_threshold=1.50,
        rtheta_expected_under_load=0.72,
        rtheta_expected_idle=1.28,
        rtheta_drift_warn_c_per_day=0.001,
        rtheta_drift_crit_c_per_day=0.005,
        cooling="air-passive",
        confidence="measured",
        notes=(
            "Stage 1 dataset baseline (4,570 rows). 100% DT accuracy in CV.",
            "R_theta thresholds are first-party measured, not extrapolated.",
        ),
    ),

    # ── NVIDIA — A100 PCIe / SXM4 (Ampere, monolithic, blower or cold-plate) ──
    "a100": ThermalProfile(
        family="ampere",
        canonical_name="A100-SXM4-80GB",
        vendor="nvidia",
        tdp_w=400.0,
        idle_floor_w=45.0,
        junction_max_c=92.0,  # Ampere allows higher T_j than Turing
        expected_ambient_c=22.0,
        # A100 has ~5.7× T4's TDP on a die ~3× the area + better TIM.
        # Scaling law: R_theta ≈ T4_R_theta × (T4_TDP/A100_TDP)^0.4 ≈ 0.55×
        rtheta_load_threshold=0.55,
        rtheta_idle_threshold=0.95,
        rtheta_expected_under_load=0.42,
        rtheta_expected_idle=0.80,
        rtheta_drift_warn_c_per_day=0.0008,
        rtheta_drift_crit_c_per_day=0.004,
        cooling="air-blower",  # PCIe variant; SXM4 is cold-plate
        confidence="extrapolated",
        notes=(
            "Extrapolated from T4 measurements via die-area + TDP scaling.",
            "Calibrate on first deployment — predicted ~0.42 C/W load, ~0.80 idle.",
            "SXM4 variant has lower R_theta (cold-plate); recalibrate per variant.",
        ),
    ),

    # ── NVIDIA — L40S (Ada Lovelace, PCIe passive-fin) ──
    "l40s": ThermalProfile(
        family="ada",
        canonical_name="L40S",
        vendor="nvidia",
        tdp_w=350.0,
        idle_floor_w=40.0,
        junction_max_c=92.0,
        expected_ambient_c=24.0,  # often deployed in retrofitted inference racks
        rtheta_load_threshold=0.58,
        rtheta_idle_threshold=0.98,
        rtheta_expected_under_load=0.46,
        rtheta_expected_idle=0.82,
        rtheta_drift_warn_c_per_day=0.001,
        rtheta_drift_crit_c_per_day=0.005,
        cooling="air-passive",
        confidence="extrapolated",
        notes=(
            "PCIe passive-fin design — relies on server-chassis airflow.",
            "Higher ambient tolerance than SXM cards (deployed in older facilities).",
        ),
    ),

    # ── NVIDIA — H100 SXM5 (Hopper, monolithic, cold-plate) ──
    "h100": ThermalProfile(
        family="hopper",
        canonical_name="H100-SXM5-80GB",
        vendor="nvidia",
        tdp_w=700.0,
        idle_floor_w=70.0,
        junction_max_c=95.0,
        expected_ambient_c=20.0,  # liquid-cooled DGX H100 typically runs 18-22 °C
        # H100 SXM5 is liquid-cooled cold-plate → much lower R_theta than air-cooled
        rtheta_load_threshold=0.40,
        rtheta_idle_threshold=0.72,
        rtheta_expected_under_load=0.30,
        rtheta_expected_idle=0.62,
        rtheta_drift_warn_c_per_day=0.0006,  # liquid cooling is slower to drift
        rtheta_drift_crit_c_per_day=0.003,
        cooling="liquid-cold-plate",
        confidence="extrapolated",
        notes=(
            "Liquid-cooled cold-plate dominates thermal path; ambient less relevant.",
            "Use BMC inlet temp from Redfish as T_ref override when available.",
            "HBM3 stacks have separate thermal path — monitor independently if exposed.",
        ),
    ),

    # ── NVIDIA — B200 SXM6 (Blackwell, dual-die CoWoS-L) ──
    "b200": ThermalProfile(
        family="blackwell",
        canonical_name="B200-SXM6",
        vendor="nvidia",
        tdp_w=1000.0,  # nominal; some variants up to 1200W
        idle_floor_w=85.0,
        junction_max_c=100.0,
        expected_ambient_c=20.0,
        # Dual-die package + larger HBM3e stack count → larger effective surface,
        # but 2.5× T4's TDP density. Net R_theta predicted ~0.27 load.
        rtheta_load_threshold=0.35,
        rtheta_idle_threshold=0.65,
        rtheta_expected_under_load=0.27,
        rtheta_expected_idle=0.55,
        rtheta_drift_warn_c_per_day=0.0005,
        rtheta_drift_crit_c_per_day=0.0025,
        cooling="liquid-cold-plate",
        confidence="extrapolated",
        notes=(
            "Dual-die package — each die has its own R_theta; report aggregate.",
            "Inter-die thermal coupling may produce asymmetric loading patterns.",
            "Predicted values pending first-party measurement on Cal Poly DGX B200.",
        ),
    ),

    # ── AMD — MI300X (CDNA3, OAM, chiplet, liquid cold-plate) ──
    "mi300x": ThermalProfile(
        family="cdna3",
        canonical_name="MI300X-OAM",
        vendor="amd",
        tdp_w=750.0,
        idle_floor_w=80.0,
        junction_max_c=110.0,  # AMD allows higher T_j than NVIDIA
        expected_ambient_c=20.0,
        # 8-chiplet design distributes heat → R_theta similar to H100 despite
        # higher TDP. Predicted ~0.32 load based on chiplet scaling.
        rtheta_load_threshold=0.42,
        rtheta_idle_threshold=0.75,
        rtheta_expected_under_load=0.32,
        rtheta_expected_idle=0.62,
        rtheta_drift_warn_c_per_day=0.0007,
        rtheta_drift_crit_c_per_day=0.0035,
        cooling="liquid-cold-plate",
        confidence="datasheet",  # AMD telemetry not yet validated against ground truth
        notes=(
            "Chiplet architecture — each XCD has its own thermal trip.",
            "ROCm telemetry differs from NVML; collector layer must abstract.",
            "Higher T_j ceiling than NVIDIA → drift detection thresholds wider.",
        ),
    ),

    # ── Intel — Gaudi 3 (HL-325L, OAM) ──
    "gaudi3": ThermalProfile(
        family="gaudi3",
        canonical_name="Gaudi3 HL-325L",
        vendor="intel",
        tdp_w=900.0,
        idle_floor_w=75.0,
        junction_max_c=95.0,
        expected_ambient_c=22.0,
        rtheta_load_threshold=0.40,
        rtheta_idle_threshold=0.72,
        rtheta_expected_under_load=0.30,
        rtheta_expected_idle=0.60,
        rtheta_drift_warn_c_per_day=0.0007,
        rtheta_drift_crit_c_per_day=0.0035,
        cooling="liquid-cold-plate",
        confidence="datasheet",
        notes=(
            "Intel Gaudi telemetry via habanalabs-smi; collector not yet built.",
            "Profile is datasheet-derived; no first-party measurements yet.",
        ),
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Match / normalize / resolve
# ──────────────────────────────────────────────────────────────────────────

# Substrings that, when found in a normalized GPU name, map to a profile key.
# Order matters: more specific substrings checked first.
_MATCH_RULES: list[tuple[str, str]] = [
    # NVIDIA datacenter — most specific first
    ("h100",   "h100"),
    ("b100",   "b200"),    # same family
    ("b200",   "b200"),
    ("gb200",  "b200"),    # GB200 superchip uses 2× B200
    ("a100",   "a100"),
    ("a800",   "a100"),    # cut-down A100 export variant
    ("h800",   "h100"),    # cut-down H100 export variant
    ("l40s",   "l40s"),
    ("l40",    "l40s"),
    ("t4",     "t4"),

    # AMD
    ("mi300x", "mi300x"),
    ("mi300",  "mi300x"),
    ("mi325",  "mi300x"),  # incremental MI325X uses same architecture

    # Intel
    ("gaudi3", "gaudi3"),
    ("gaudi",  "gaudi3"),
    ("hl-325", "gaudi3"),
]


def _normalize(name: str) -> str:
    """Lowercase + strip vendor prefixes for matching."""
    n = name.lower().strip()
    for prefix in ("nvidia ", "tesla ", "amd ", "intel ", "habana "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def resolve_profile(gpu_name: str) -> Optional[ThermalProfile]:
    """
    Return the hardware profile for a GPU name, or None if unmatched.

    Matching is fuzzy: 'NVIDIA H100 80GB HBM3' → h100, 'AMD Instinct MI300X' → mi300x.
    """
    if not gpu_name:
        return None
    norm = _normalize(gpu_name)
    for substring, key in _MATCH_RULES:
        if substring in norm:
            return _PROFILES.get(key)
    return None


def resolve_or_default(gpu_name: str) -> ThermalProfile:
    """Resolve profile, falling back to T4 (the measured baseline) if unknown.

    Use this when downstream code needs SOME profile to proceed — but it should
    log a warning so operators know they're running on an unprofiled GPU class.
    """
    p = resolve_profile(gpu_name)
    return p if p is not None else _PROFILES["t4"]


def all_profiles() -> dict[str, ThermalProfile]:
    """Read-only view of every registered profile (for diagnostic CLI / API)."""
    return dict(_PROFILES)


def profile_summary(profile: ThermalProfile) -> dict:
    """Serialize a profile for JSON APIs / Prometheus labels / wizard display."""
    return {
        "family": profile.family,
        "canonical_name": profile.canonical_name,
        "vendor": profile.vendor,
        "tdp_w": profile.tdp_w,
        "idle_floor_w": profile.idle_floor_w,
        "junction_max_c": profile.junction_max_c,
        "expected_ambient_c": profile.expected_ambient_c,
        "rtheta_load_threshold": profile.rtheta_load_threshold,
        "rtheta_idle_threshold": profile.rtheta_idle_threshold,
        "rtheta_expected_under_load": profile.rtheta_expected_under_load,
        "rtheta_expected_idle": profile.rtheta_expected_idle,
        "rtheta_drift_warn_c_per_day": profile.rtheta_drift_warn_c_per_day,
        "rtheta_drift_crit_c_per_day": profile.rtheta_drift_crit_c_per_day,
        "cooling": profile.cooling,
        "confidence": profile.confidence,
        "notes": list(profile.notes),
    }
