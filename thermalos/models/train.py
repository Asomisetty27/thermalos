"""
Train and bundle the Stage 1 classifier models.

Trains GaussianNB and DecisionTreeClassifier on Stage 1 data with
the 15-second steady-state filter applied, then saves to models/bundle/.

Run:
    python3 thermalos/models/train.py
    python3 thermalos/models/train.py --csv /path/to/ThermalOS_Measurements_Raw.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

BUNDLE_DIR = Path(__file__).parent / "bundle"

PHASE_TO_CLASS = {
    "clean_idle":                            "clean_idle",
    "under_load":                            "under_load",
    "under_load_e002":                       "under_load",
    "e003_separate_process_load":            "under_load",
    "e003_rerun_separate_process_load":      "under_load",
    "extended_post_load_recovery":           "zombie_recovery",
    "e003_recovery_after_child_exit":        "child_exit_recovery",
    "e003_rerun_recovery_after_child_exit":  "child_exit_recovery",
}
for t in range(1, 9):
    PHASE_TO_CLASS[f"e004_t{t}_separate_process_load"]       = "under_load"
    PHASE_TO_CLASS[f"e004_t{t}_recovery_after_child_exit"]   = "child_exit_recovery"
    PHASE_TO_CLASS[f"e004v2_t{t}_separate_process_load"]     = "under_load"
    PHASE_TO_CLASS[f"e004v2_t{t}_recovery_after_child_exit"] = "child_exit_recovery"

CLASS_NAMES = ["child_exit_recovery", "clean_idle", "under_load", "zombie_recovery"]

WINDOW_SEC  = 15.0
SIGMA_THRESH = 0.05


def load_steady_state(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import csv as csvmod

    phases = defaultdict(list)
    with open(csv_path) as f:
        for row in csvmod.DictReader(f):
            phase = row.get("phase", "")
            if phase not in PHASE_TO_CLASS:
                continue
            try:
                phases[phase].append({
                    "rtheta":  float(row["rtheta_cwatt"]),
                    "power":   float(row["power_w"]),
                    "util":    float(row["util_pct"]),
                    "pstate":  int(row["perf_state"].replace("P", "")),
                    "ts":      int(row["trial_second"]),
                    "cls":     PHASE_TO_CLASS[phase],
                })
            except (ValueError, KeyError):
                continue

    X_rows, y_rows = [], []
    for phase, rows in phases.items():
        rows_sorted = sorted(rows, key=lambda r: r["ts"])
        buf: list[float] = []
        for row in rows_sorted:
            buf.append(row["rtheta"])
            if len(buf) > int(WINDOW_SEC):
                buf.pop(0)
            if len(buf) < int(WINDOW_SEC):
                continue
            mean_r = sum(buf) / len(buf)
            std_r  = math.sqrt(sum((v - mean_r) ** 2 for v in buf) / len(buf))
            if std_r < SIGMA_THRESH and row["power"] > 5.0:
                X_rows.append([row["rtheta"], row["power"], row["util"], row["pstate"]])
                y_rows.append(CLASS_NAMES.index(row["cls"]))

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)


def train(csv_path: Path) -> None:
    print(f"Loading Stage 1 data from: {csv_path}")
    X, y = load_steady_state(csv_path)
    print(f"  {len(X)} steady-state samples  ·  {len(set(y))} classes")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Naive Bayes
    nb = GaussianNB()
    nb_scores = cross_val_score(nb, X, y, cv=skf, scoring="accuracy")
    nb.fit(X, y)
    print(f"\nNaive Bayes     5-fold CA: {nb_scores.mean():.4f} ± {nb_scores.std():.4f}")

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_scores = cross_val_score(dt, X, y, cv=skf, scoring="accuracy")
    dt.fit(X, y)
    print(f"Decision Tree   5-fold CA: {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(nb, BUNDLE_DIR / "nb_steady_state.pkl")
    joblib.dump(dt, BUNDLE_DIR / "dt_steady_state.pkl")
    print(f"\nModels saved to {BUNDLE_DIR}/")
    print("  nb_steady_state.pkl")
    print("  dt_steady_state.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent.parent.parent.parent /
                    "thermalos-vault" / "raw" / "experiments" / "ThermalOS_Measurements_Raw.csv"),
        help="Path to ThermalOS_Measurements_Raw.csv",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    train(csv_path)
