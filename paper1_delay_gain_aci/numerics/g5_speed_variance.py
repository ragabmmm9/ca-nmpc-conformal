"""G5: stationary spread of the level versus per-step adaptation speed at delay D=25 (tau=D+1=26),
i.i.d. Exp(1) scores, frozen oracle quantile map, no shift (kappa=1).
integrator: speed v = eta;  phase-decoupled tau-DACI: speed v = gamma/tau (each chain moves once per tau steps)."""
import numpy as np, csv, os, math
from g4_compare import sim_integrator, sim_phase, kstar, alpha
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(23)
D = 25; tau = D + 1; R, T, burn = 400, 6000, 2000
sig2 = alpha * (1 - alpha)
rows = []
for v in [0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.045, 0.06, 0.08]:
    S = rng.exponential(1.0, (R, T))
    A, E = sim_integrator(v, D, S)
    rows.append(dict(method="integrator", speed=v, step=v, sd_sim=float(A[:, burn:].std()), pinf=float((A[:, burn:] <= 0).mean()),
                     miss=float(E[:, burn:].mean()), stable=bool(v < kstar(D))))
for gamma in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]:
    S = rng.exponential(1.0, (R, T))
    A, E = sim_phase(gamma, tau, S)
    th = math.sqrt(gamma * sig2 / (2 - gamma))
    rows.append(dict(method="phase", speed=gamma / tau, step=gamma, sd_sim=float(A[:, burn:].std()), pinf=float((A[:, burn:] <= 0).mean()),
                     miss=float(E[:, burn:].mean()), stable=bool(gamma < 2), sd_theory_linear=th))
keys = ["method", "speed", "step", "sd_sim", "pinf", "miss", "stable", "sd_theory_linear"]
with open(os.path.join(OUT, "G5_speed_variance.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
