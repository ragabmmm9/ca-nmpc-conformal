# ca-nmpc-conformal

Simulation and analysis code that accompanies three research manuscripts on uncertainty quantification for learning-based vehicle control under distribution shift. **The manuscripts themselves are not included in this repository** (they are unpublished); this code supports the experiments described in them.

| # | Manuscript title | Code in this repository |
|---|---|---|
| 1 | *Delay–Gain Stability of Adaptive Conformal Inference with Delayed Feedback* | `paper1_delay_gain_aci/` |
| 2 | *When Calibrated Margins Do Not Help: Conformal Uncertainty Tubes for Learning-Based Predictive Vehicle Control under Distribution Shift* | `ca_nmpc/` |
| 3 | *Which Uncertainty Estimates Survive Dynamics Shift? A Multi-Step Benchmark of Ensembles, Dropout, Heteroscedastic, Last-Layer Bayesian, Quantile and Conformal Estimators for Hybrid Physics–Neural Vehicle Models* | `paper3_uq_benchmark/` |

## Overview

**CA-NMPC** (Paper 2) is a nonlinear MPC for obstacle avoidance that combines
a physical nominal model with an online friction estimate (RLS), a trust-gated neural residual
(ensemble distilled into one student), and horizon-grouped **adaptive conformal inference** margins
that enter the optimal-control problem as soft control-barrier-function constraints (CasADi / IPOPT).
The closed-loop study compares it with tuned non-conformal baselines on two vehicle plants
(own 8-state single-track model and the CommonRoad 29-state multi-body model) under hand-designed
friction, mass, sensor and obstacle-motion shifts, using paired exact tests with Holm correction.
Paper 1 contains the delayed-feedback ACI theory with small numerical checks; Paper 3 benchmarks
nine uncertainty estimators on 25-step hybrid-model predictions.

## Repository layout

```
ca_nmpc/                              Paper 2
  notebooks/CA_NMPC_Pilot_Step1_Step2_v3_1.ipynb
        full simulator: plants, neural-residual ensemble, CasADi NMPC, friction RLS, trust gate,
        grouped ACI tubes, all benchmark controllers, moving-obstacle world, CBF filter,
        pilot protocol (Step 1 diagnosis, Step 2 moving obstacles), statistics, report export
  experiments/matched_margin/         post-hoc matched-margin experiment (+ PREDICTIONS.md written before the run)
  experiments/success_sensitivity_power/   post-hoc success-definition sensitivity and power analysis
  legacy_scripts/                     early development scripts (assembly of the first benchmark notebook)
paper1_delay_gain_aci/numerics/       numerical checks g4 (comparison of delayed recursions), g5, g6
paper3_uq_benchmark/
  code/  PREREG.md                    simulator, training, evaluation, analysis; predictions written before evaluation
```

## Running the Paper 2 notebook

Requirements: Python 3, `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `casadi` (with IPOPT) and
`commonroad-vehicle-models` (the first cell installs `casadi` and the vehicle models if missing).
Edit the choices in the second cell, then run all cells:

| setting | values |
|---|---|
| `PLANT_CHOICE` | `"mb"` (29-state multi-body plant) or `"home"` (single-track plant) |
| `PILOT_CHOICE` | `"smoke"` (wiring check), `"quick"` (8 test seeds), `"pilot30"` (30 test seeds) |
| `NN_MODE` | `"smoke"`, `"fast"`, `"standard"` (size of offline data / network training) |
| `DELAY_CHOICE` | computational delay in control steps (default 1) |

Runs are resumable. Development seeds `0–3` are used for tuning only; test seeds start at `1000`.

## Status and known gaps (please read)

- The notebook is stored **without outputs**. Its own header notes that parts of it were assembled from earlier validated code and include a regression test of the extended simulator against the validated one.
- The scripts in `experiments/` load a flattened copy of the notebook (`prefix_flat.py`) and a stored pilot run from hard-coded paths (e.g. `./pilotrun/...`); these paths must be adapted. `paper3_uq_benchmark/code` also contains hard-coded relative paths (e.g. `./paper_uq/results`) and expects an input episode file from an earlier experiment (`episodes_full.pkl`) that is **not** included.
- The runner and statistics for the 18-controller benchmark (two plants, 27 shifted episodes per controller) are not separate scripts here; the controllers are defined in the notebook, but a stand-alone benchmark runner is missing.
- `paper1_delay_gain_aci/numerics` contains only the generators for the g4–g6 checks; generators for the earlier tables (G1–G3, E1) are not included.
- Result files (episode CSVs, tables, trained models) are not included.
- Everything is simulation; no real-vehicle data are used. No license has been chosen yet.
