# E-LT Thermal Simulation

A physics-grounded simulation of the **E-LT lead-time testbed** — the make-or-break
experiment for Theta/Theta. It answers the one question that gates the entire
predictive-maintenance product claim:

> **Does R_θ (effective thermal resistance) rise *detectably* before thermal
> throttling, and by how much lead time, per cooling-degradation mode?**

This lets us derive the lead-time result *before* Sam's physical testbed exists
(fall 2026). When the hardware arrives, only the degradation knob changes — the
model, detector, and analysis are already validated.

---

## Why a lumped Cauer network (and not CFD or MATLAB)

The Theta detector only ever sees **scalar telemetry**: one junction
temperature, one power number. It computes `R_θ = (T_j − T_ref) / P` from those.
The lead-time question is therefore a **lumped transient** question, and the right
model fidelity is a multi-node RC thermal network — not 3D CFD (whose spatial
detail the scalar detector can't observe) and not MATLAB (which would fork the
detector). Python lets the simulation use the **same** steady-state-window +
baseline+kσ detector the agent ships, exactly as the protocol requires.

The one spatial risk CFD would capture — degradation moving the hot-spot away
from the sensor — is representable here as a sensor-coupling parameter, without a
mesh.

---

## The physics

A 3-node Cauer (physical ladder) network, calibrated to Stage 1 Tesla T4 data:

```
P(t) ──► [T_j] ──Rjc──► [T_c] ──Rct(TIM)──► [T_s] ──Rsa(airflow)──► T_amb
          C_j            C_c                 C_s
```

```
C_j dT_j/dt = P_eff      − (T_j−T_c)/Rjc
C_c dT_c/dt = (T_j−T_c)/Rjc − (T_c−T_s)/Rct(t)
C_s dT_s/dt = (T_c−T_s)/Rct(t) − (T_s−T_amb)/Rsa(airflow(t,T_j))
```

At steady state this reduces to `T_j = T_amb + P·(Rjc+Rct+Rsa)`, i.e.
`R_θ = Rjc+Rct+Rsa` — exactly the quantity the product computes.

Key modelling choices, all with provenance in `elt/params.py`:

- **Calibration.** `R_sa_ref` is solved numerically so the simulated *load* point
  reproduces the measured 81 °C @ 68 W exactly. The idle point then falls out of
  the same model (residual +1.9 °C, within the 1 °C sensor quantisation + the
  assumed-25 °C-ambient slack).
- **Temperature-following fan.** A real GPU ramps its fan with junction temp, so
  the convective resistance `Rsa = R_sa_ref / airflow^0.8` is itself
  temperature-dependent (Dittus-Boelter exponent 0.8).
- **Stiff integration.** The junction time constant (~0.3 s) and heatsink time
  constant (~45 s) span two decades — the system is stiff, so it's integrated with
  an implicit BDF solver. The exact throttle-crossing time comes from a solver
  *event*, not grid-snapping.
- **Real sensor model.** Junction temp is reported as **integer degrees** (as in
  the Stage 1 CSV), with sub-degree noise added before quantisation; power carries
  the ~±0.5 W jitter observed in the data. The detector never sees ground truth.
- **One-sided throttle governor.** Full clocks until T_j hits 93 °C; above that,
  power is reduced to hold the junction near the limit. The first 93 °C crossing
  is the ground-truth throttle event lead time is measured against.

## The detector

`elt/detector.py` mirrors `theta/agent/window.py` + the baseline+kσ drift rule:

1. Steady-state window — R_θ computed only when power is stable (Kundu's guidance).
2. Healthy baseline — mean & std of windowed R_θ over the pre-degradation phase.
3. Anomaly — R_θ > μ + k·σ, sustained for `persist_s`. Sweep k ∈ {2,3,4}.
4. `lead_time = t_throttle − t_anomaly`.

---

## Degradation modes

| Mode | Physical parameter | Default onset | Variants |
|------|--------------------|---------------|----------|
| `tim` | R_ct rises (exp. dry-out) | 6 h | gradual, step |
| `airflow` | airflow falls (occlusion) | 45 min | gradual, step |
| `fan` | fan duty capped (actuator loss) | 10 min | step, gradual |

---

## Usage

```bash
# from the theta-agent repo root, using the sim venv:
SIM=sim/.venv/bin/python

# 1. validate the model against Stage 1 (run this first)
$SIM -m sim.elt.validate

# 2. one trial + trajectory plot
$SIM -m sim.elt.run_elt single --mode tim --variant gradual

# 3. Monte Carlo for one mode
$SIM -m sim.elt.run_elt mc --mode airflow --trials 50

# 4. THE DELIVERABLE: all modes, all plots, JSON + summary
$SIM -m sim.elt.run_elt full --trials 50 --out sim/elt/out

# virtual-ambient (product, no thermocouple) vs true-ambient (lab) comparison
$SIM -m sim.elt.run_elt full --ambient virtual --out sim/elt/out_virtual
```

Outputs (in `--out`): per-mode trajectory plots, lead-time distributions,
`leadtime_vs_k.png`, `elt_results.json`, `elt_summary.txt`.

---

## Representative result (N=40, k=3σ, true ambient)

| mode/variant | detect % | median lead | range | protocol verdict |
|---|---|---|---|---|
| tim/gradual | 100% | ~1.2 h | 39 min – 4.5 h | STRONG predictive |
| airflow/gradual | 92% | ~27 min | 18 – 33 min | STRONG predictive |
| fan/step | 100% | ~1.2 min | 30 s – 3 min | acute-fault early warning |

These map directly onto the protocol's decision rules: slow degradation (TIM,
airflow) gives tens of minutes to hours of warning — a genuine predictive-
maintenance product; abrupt fan loss gives ~1 minute — useful only for acute
faults. **The simulation is a prediction to be confirmed on Sam's testbed, not a
substitute for it.** Every number is reproducible and every parameter has a
provenance tag.

---

## Files

| File | Role |
|------|------|
| `elt/params.py` | calibrated physical parameters (provenance-tagged) |
| `elt/thermal_model.py` | Cauer network + stiff ODE + fan/throttle |
| `elt/degradation.py` | the three degradation modes |
| `elt/detector.py` | sensor model + steady-state window + baseline+kσ |
| `elt/experiment.py` | single trial + Monte Carlo |
| `elt/analysis.py` | plots + summary tables + JSON export |
| `elt/validate.py` | 7-check validation against Stage 1 |
| `elt/run_elt.py` | CLI (`validate` / `single` / `mc` / `full`) |
| `tests/test_elt.py` | pytest suite |

## Limitations (be honest, Kundu/Peuker will ask)

- Lumped model: no spatial hot-spot field (captured as a parameter, not a mesh).
- Calibrated to one GPU (T4 Colab). Testbed re-calibration expected per `calibrate`.
- Degradation trajectories are plausible functional forms, not measured dry-out
  curves — the testbed will replace them with real ones.
- Single workload (fixed power). Multi-workload baselines are future work.
