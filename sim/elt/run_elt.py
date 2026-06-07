"""
E-LT simulation CLI.

Usage:
    python -m sim.elt.run_elt validate
    python -m sim.elt.run_elt single  --mode tim --variant gradual
    python -m sim.elt.run_elt mc      --mode tim --variant gradual --trials 50
    python -m sim.elt.run_elt full    --trials 50 --out sim/elt/out
        (runs all three modes, writes plots + JSON + summary; this is the deliverable)

All outputs go to --out (default sim/elt/out/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import degradation as deg
from .experiment import run_trial, run_monte_carlo
from .detector import DetectorConfig
from . import analysis
from . import validate as validate_mod


# Default per-mode run geometry: (variant, duration_s, baseline_s)
MODE_GEOMETRY = {
    "tim":           ("gradual",  deg.DEFAULT_HORIZON_S["tim"],           600.0),
    "tim_arrhenius": ("emergent", deg.DEFAULT_HORIZON_S["tim_arrhenius"], 600.0),
    "airflow":       ("gradual",  deg.DEFAULT_HORIZON_S["airflow"],       600.0),
    "fan":           ("step",     deg.DEFAULT_HORIZON_S["fan"],           180.0),
}


def _out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_validate(_args) -> int:
    ok, checks = validate_mod.run_all()
    print(validate_mod.format_report(checks))
    return 0 if ok else 1


def cmd_single(args) -> int:
    variant, dur, base = MODE_GEOMETRY[args.mode]
    variant = args.variant or variant
    out = _out_dir(args.out)
    tr = run_trial(args.mode, variant=variant, duration_s=dur, baseline_s=base,
                   seed=args.seed, ambient_mode=args.ambient, keep_traces=True)
    print(f"mode={tr.mode}/{tr.variant}  ambient={tr.ambient_mode}")
    print(f"t_throttle = {tr.t_throttle:.0f} s" if tr.t_throttle else "no throttle")
    print(f"baseline R_theta = {tr.baseline.mean:.4f} +/- {tr.baseline.std:.5f} "
          f"(n={tr.baseline.n})")
    for k in tr.lead_times:
        lt = tr.lead_times[k]
        print(f"  k={k:g}: t_anomaly={tr.t_anomaly[k]}  "
              f"lead_time={'%.0f s (%.1f min)' % (lt, lt/60) if lt else 'none'}")
    p = analysis.plot_trajectory(tr, out / f"trajectory_{tr.mode}_{tr.variant}.png")
    print(f"\nwrote {p}")
    return 0


def cmd_mc(args) -> int:
    variant, dur, base = MODE_GEOMETRY[args.mode]
    variant = args.variant or variant
    out = _out_dir(args.out)
    mc = run_monte_carlo(args.mode, variant=variant, n_trials=args.trials,
                         duration_s=dur, baseline_s=base, ambient_mode=args.ambient)
    print(f"=== Monte Carlo: {args.mode}/{variant}  N={args.trials}  "
          f"ambient={args.ambient} ===")
    for k in mc.k_values:
        s = mc.summary(k)
        if s.get("n", 0):
            print(f"  k={k:g}: detect={s['detect_rate']*100:.0f}%  "
                  f"median={analysis._fmt_dur(s['median_s'])}  "
                  f"mean={analysis._fmt_dur(s['mean_s'])}  "
                  f"[{analysis._fmt_dur(s['min_s'])}..{analysis._fmt_dur(s['max_s'])}]")
        else:
            print(f"  k={k:g}: no detections before throttle")
    p = analysis.plot_distribution(mc, out / f"dist_{args.mode}_{variant}.png",
                                   k=args.k)
    print(f"wrote {p}")
    return 0


def cmd_full(args) -> int:
    out = _out_dir(args.out)

    print("Validating model against Stage 1...")
    ok, checks = validate_mod.run_all()
    print(validate_mod.format_report(checks))
    if not ok:
        print("\nVALIDATION FAILED — aborting (lead-time numbers not trustworthy).")
        return 1

    mc_by_mode = {}
    first_trials = {}
    for mode, (variant, dur, base) in MODE_GEOMETRY.items():
        print(f"\nRunning {mode}/{variant}  N={args.trials} ...")
        mc = run_monte_carlo(mode, variant=variant, n_trials=args.trials,
                             duration_s=dur, baseline_s=base,
                             ambient_mode=args.ambient, keep_first_trace=True)
        key = f"{mode}/{variant}"
        mc_by_mode[key] = mc
        # one representative trajectory plot per mode (first trial keeps traces)
        tr0 = mc.trials[0]
        if tr0.sim is not None:
            analysis.plot_trajectory(tr0, out / f"trajectory_{mode}_{variant}.png",
                                     k=args.k)
        analysis.plot_distribution(mc, out / f"dist_{mode}_{variant}.png", k=args.k)
        first_trials[key] = tr0

    analysis.plot_leadtime_vs_k(mc_by_mode, out / "leadtime_vs_k.png")
    analysis.export_json(mc_by_mode, out / "elt_results.json")

    report = []
    report.append("E-LT SIMULATION RESULTS")
    report.append("=" * 77)
    report.append(f"Monte Carlo N={args.trials} per mode, ambient={args.ambient}, "
                  f"headline k={args.k:g}\n")
    report.append(analysis.summary_table(mc_by_mode, k_headline=args.k))
    report.append("")
    report.append("HEADLINE: " + analysis.headline(mc_by_mode, k=args.k))
    report.append("")
    report.append("Decision-rule reading (per protocol):")
    for key, mc in mc_by_mode.items():
        s = mc.summary(args.k)
        if not s.get("n"):
            continue
        med_min = s["median_s"] / 60.0
        if med_min >= 10:
            verdict = "STRONG predictive product (tens of minutes warning)"
        elif med_min >= 1:
            verdict = "useful for acute fast-onset faults (minutes)"
        else:
            verdict = "prediction fails — forensic/efficiency only (<60s)"
        report.append(f"  {key:<18} median {analysis._fmt_dur(s['median_s'])}  -> {verdict}")

    text = "\n".join(report)
    (out / "elt_summary.txt").write_text(text)
    print("\n" + text)
    print(f"\nAll deliverables written to {out}/")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E-LT lead-time thermal simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("validate", help="validate model vs Stage 1")
    sp.set_defaults(fn=cmd_validate)

    for name, fn in (("single", cmd_single), ("mc", cmd_mc)):
        sp = sub.add_parser(name)
        sp.add_argument("--mode", choices=list(MODE_GEOMETRY), default="tim")
        sp.add_argument("--variant", choices=["gradual", "step", "emergent"], default=None)
        sp.add_argument("--trials", type=int, default=50)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--ambient", choices=["true", "virtual"], default="true")
        sp.add_argument("--k", type=float, default=3.0)
        sp.add_argument("--out", default="sim/elt/out")
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("full", help="run all modes, write all deliverables")
    sp.add_argument("--trials", type=int, default=50)
    sp.add_argument("--ambient", choices=["true", "virtual"], default="true")
    sp.add_argument("--k", type=float, default=3.0)
    sp.add_argument("--out", default="sim/elt/out")
    sp.set_defaults(fn=cmd_full)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
