"""
Analysis + plotting for E-LT results — produces the deliverables the protocol lists:

  * R_theta trajectory plot (R_theta vs time, t_anomaly + t_throttle marked,
    power overlaid flat to prove the rise is cooling, not workload).
  * Lead-time vs k tradeoff curve.
  * Lead-time distribution histogram (Monte Carlo).
  * Summary tables + the headline single number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt

from . import params as P
from .experiment import TrialResult, MonteCarloResult

# Brand-aligned palette (Theta thermal ramp)
_HEALTHY = "#27A05A"
_CAUTION = "#C8942A"
_RISING  = "#C85F2A"
_CRIT    = "#B83030"
_BP      = "#5878A8"
_FG      = "#E2E2EA"
_MUTED   = "#9A9AAA"
_BG      = "#0E0E13"


def _style(ax):
    ax.set_facecolor(_BG)
    for s in ax.spines.values():
        s.set_color("#2A2A38")
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color(_FG)
    ax.grid(True, color="#1C1C26", linewidth=0.6)


def plot_trajectory(tr: TrialResult, out_path: Path, k: float = 3.0) -> Path:
    """R_theta + T_j + flat power vs time, with anomaly/throttle markers."""
    if tr.sim is None or tr.rtheta is None:
        raise ValueError("trial has no retained traces; run with keep_traces=True")

    t_min = tr.sim.t / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_BG)

    # ── Top: R_theta (windowed, what the detector sees) + true ──
    ax1.plot(t_min, tr.sim.rtheta_true, color=_MUTED, lw=1.0, alpha=0.5,
             label="R_θ true (ground truth)")
    finite = np.isfinite(tr.rtheta)
    ax1.plot(t_min[finite], tr.rtheta[finite], color=_HEALTHY, lw=1.6,
             label="R_θ windowed (detector)")

    base = tr.baseline
    thr = base.mean + k * base.std
    ax1.axhline(base.mean, color=_BP, lw=1.0, ls="--", alpha=0.8,
                label=f"baseline μ={base.mean:.3f}")
    ax1.axhline(thr, color=_CAUTION, lw=1.2, ls=":",
                label=f"threshold μ+{k:g}σ={thr:.3f}")

    ta = tr.t_anomaly.get(k)
    if ta is not None:
        ax1.axvline(ta / 60.0, color=_CAUTION, lw=1.6,
                    label=f"t_anomaly={ta/60:.1f}min")
    if tr.t_throttle is not None:
        ax1.axvline(tr.t_throttle / 60.0, color=_CRIT, lw=1.6,
                    label=f"t_throttle={tr.t_throttle/60:.1f}min")
    if ta is not None and tr.t_throttle is not None:
        ax1.axvspan(ta / 60.0, tr.t_throttle / 60.0, color=_HEALTHY, alpha=0.08)
        lead = (tr.t_throttle - ta) / 60.0
        mid = (ta + tr.t_throttle) / 2 / 60.0
        ax1.annotate(f"lead time\n{lead:.1f} min",
                     xy=(mid, base.mean), color=_HEALTHY, ha="center",
                     fontsize=10, fontweight="bold")

    ax1.set_ylabel("R_θ  (°C/W)")
    ax1.set_title(f"E-LT: {tr.mode} / {tr.variant}  —  R_θ degradation precedes throttling")
    ax1.legend(loc="upper left", fontsize=8, facecolor=_BG, edgecolor="#2A2A38",
               labelcolor=_FG, ncol=2)
    _style(ax1)

    # ── Bottom: power (flat) + junction temp + throttle line ──
    ax2.plot(t_min, tr.sim.p_eff, color=_RISING, lw=1.3, label="GPU power (W)")
    ax2.axhline(tr.sim.scenario.workload_power_w, color=_RISING, lw=0.8, ls="--",
                alpha=0.5)
    ax2.set_ylabel("Power (W)", color=_RISING)
    ax2.tick_params(axis="y", colors=_RISING)

    ax3 = ax2.twinx()
    ax3.plot(t_min, tr.sim.tj_true, color=_BP, lw=1.3, label="T_junction (°C)")
    ax3.axhline(P.THROTTLE_TEMP_C, color=_CRIT, lw=1.0, ls=":")
    ax3.set_ylabel("T_junction (°C)", color=_BP)
    ax3.tick_params(axis="y", colors=_BP)

    ax2.set_xlabel("time (minutes)")
    ax2.set_title("Power held flat (the experimental control) — T_j rises as cooling degrades",
                  fontsize=9)
    _style(ax2)
    ax2.grid(False); ax3.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out_path


def plot_leadtime_vs_k(mc_by_mode: dict, out_path: Path) -> Path:
    """Lead-time vs k tradeoff curve (median +/- IQR) per mode."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(_BG)
    colors = {"tim": _HEALTHY, "tim_arrhenius": _BP, "airflow": _CAUTION, "fan": _RISING}

    for mode, mc in mc_by_mode.items():
        ks = list(mc.k_values)
        med, lo, hi = [], [], []
        for k in ks:
            lt = mc.lead_times.get(k, np.array([])) / 60.0
            if lt.size:
                med.append(np.median(lt))
                lo.append(np.percentile(lt, 25))
                hi.append(np.percentile(lt, 75))
            else:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        c = colors.get(mode, _BP)
        ax.plot(ks, med, "o-", color=c, lw=1.8, label=f"{mode} (median)")
        ax.fill_between(ks, lo, hi, color=c, alpha=0.15)

    ax.set_xlabel("anomaly sensitivity  k  (μ + k·σ)")
    ax.set_ylabel("lead time (minutes)")
    ax.set_title("Lead time vs detection sensitivity (median, IQR band)")
    ax.legend(fontsize=9, facecolor=_BG, edgecolor="#2A2A38", labelcolor=_FG)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out_path


def plot_distribution(mc: MonteCarloResult, out_path: Path, k: float = 3.0) -> Path:
    """Histogram of lead times across Monte Carlo trials at a fixed k."""
    lt = mc.lead_times.get(k, np.array([])) / 60.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(_BG)
    if lt.size:
        ax.hist(lt, bins=min(20, max(5, lt.size // 3)), color=_HEALTHY, alpha=0.75,
                edgecolor=_BG)
        ax.axvline(np.median(lt), color=_CAUTION, lw=1.6,
                   label=f"median {np.median(lt):.1f} min")
        ax.axvline(np.mean(lt), color=_BP, lw=1.4, ls="--",
                   label=f"mean {np.mean(lt):.1f} min")
    ax.set_xlabel("lead time (minutes)")
    ax.set_ylabel("trials")
    ax.set_title(f"Lead-time distribution — {mc.mode}/{mc.variant}, "
                 f"k={k:g}, N={mc.n_trials}, detect={mc.detect_rate.get(k,0)*100:.0f}%")
    ax.legend(fontsize=9, facecolor=_BG, edgecolor="#2A2A38", labelcolor=_FG)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Text deliverables
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_dur(s: float) -> str:
    if s >= 3600:
        return f"{s/3600:.1f} h"
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.0f} s"


def summary_table(mc_by_mode: dict, k_headline: float = 3.0) -> str:
    """Human-readable summary across modes for one k."""
    lines = []
    lines.append(f"{'mode/variant':<22}{'N':>4}{'detect%':>9}"
                 f"{'median':>11}{'mean':>11}{'min':>10}{'max':>10}")
    lines.append("-" * 77)
    for key, mc in mc_by_mode.items():
        s = mc.summary(k_headline)
        if s.get("n", 0) == 0:
            lines.append(f"{key:<22}{0:>4}{mc.detect_rate.get(k_headline,0)*100:>8.0f}%"
                         f"{'—':>11}{'—':>11}{'—':>10}{'—':>10}")
            continue
        lines.append(
            f"{key:<22}{s['n']:>4}{s['detect_rate']*100:>8.0f}%"
            f"{_fmt_dur(s['median_s']):>11}{_fmt_dur(s['mean_s']):>11}"
            f"{_fmt_dur(s['min_s']):>10}{_fmt_dur(s['max_s']):>10}"
        )
    return "\n".join(lines)


def headline(mc_by_mode: dict, k: float = 3.0) -> str:
    """The single sentence for the deck."""
    parts = []
    for key, mc in mc_by_mode.items():
        s = mc.summary(k)
        if s.get("n", 0):
            parts.append(f"{mc.mode} {_fmt_dur(s['median_s'])}")
    inner = ", ".join(parts)
    return (f"R_θ detected cooling degradation a median of [{inner}] "
            f"before thermal throttling (k={k:g}σ).")


def export_json(mc_by_mode: dict, out_path: Path) -> Path:
    """Machine-readable results for the vault / site / sheets."""
    payload = {}
    for key, mc in mc_by_mode.items():
        payload[key] = {
            "mode": mc.mode, "variant": mc.variant,
            "ambient_mode": mc.ambient_mode, "n_trials": mc.n_trials,
            "by_k": {str(k): mc.summary(k) for k in mc.k_values},
            "throttle_time_mean_s": float(np.mean(mc.throttle_times))
                if mc.throttle_times.size else None,
        }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
