# %% [markdown]
# # CA-NMPC: Conformally-Calibrated, Online-Adapting Neural-Residual MPC with CBF-form Constraints
# ### Safe learning-based lateral/longitudinal vehicle control under distribution shift
#
# **What this notebook does (end-to-end, runs on a free Colab CPU):**
# 1. Builds a *high-fidelity* vehicle "plant" (dynamic bicycle, Pacejka tyres, friction ellipse, steering-rate limit,
#    actuator lags, process + measurement noise) and a *mismatched nominal model* (linear tyres) used by the controller.
# 2. Learns a **deep-ensemble neural residual** `f_true - f_nominal` offline on *high-friction* data only.
# 3. Embeds the residual inside a **CasADi NMPC**, and adapts its **last layer online with RLS**.
# 4. Wraps the MPC with a **horizon-wise, online-calibrated conformal tube** (Adaptive Conformal Inference + a
#    shift-propagation rule that fixes the *delayed-feedback* problem of multi-step conformal prediction) used to
#    tighten **CBF-form** obstacle / road constraints.
# 5. Evaluates against 6 baselines + ablations on 4 shift scenarios with paired statistics, coverage diagnostics, and
#    exports figures, CSV and LaTeX tables.
#
# Set `MODE` in the config cell: `"smoke"` (~minutes, sanity check), `"fast"` (default), `"full"` (paper-grade seeds).

# %% [markdown]
# ## 0. Research gap and contributions
#
# **Gap.** Learning-based MPC (neural / GP residual dynamics) is usually validated *in-distribution*. When tyre-road
# friction or mass shifts, (i) the offline residual is wrong exactly where it matters (near the handling limit),
# (ii) ensemble disagreement is *not calibrated* and does not translate into a position-level safety margin, and
# (iii) conformal prediction for multi-step prediction suffers from **delayed feedback**: the error at horizon *j* is only
# observed *j* steps later, so far-horizon margins react to a shift after seconds, which is too late for safety.
#
# **Contributions (what the experiments below test):**
# * **C1 – Online last-layer adaptation** of an ensemble-mean residual inside the MPC (RLS, no re-compilation of the NLP).
# * **C2 – Horizon-wise conformal tube with ACI**: per-horizon-group Adaptive Conformal Inference on *normalised*
#   multi-step position-error scores, computed model-side (re-rolling the model along the *executed* inputs) so scores are
#   independent of the controller.
# * **C3 – Shift-propagation rule**: the inflation factor observed at short horizons (available quickly) lower-bounds the
#   inflation at longer horizons, closing most of the delayed-feedback gap.
# * **C4 – CBF-form constraints with conformal margin** (`h_j >= (1-gamma) h_{j-1}`) to keep the NLP feasible while remaining safe.
# * **C5 – Reproducible shift benchmark** + coverage/safety/efficiency metrics + paired statistical tests.
#
# **Proposition (informal, proved in the Method section).** If the j-step position error is bounded by the margin `q_j`
# (event whose long-run frequency ACI controls to `1-alpha` for arbitrary shift) and the MPC solution satisfies the tightened constraints
# with zero slack, then the *true* trajectory under the same input sequence satisfies the original constraints over the horizon.
#
# > Honest scope: this is an *open-loop-tube* guarantee (no terminal invariant set), validated in simulation. Section 11 lists
# > what is still needed for a Q1 submission (real data, hardware-in-the-loop, stronger theory).

# %%
import os, sys, time, math, json, subprocess, warnings, random, itertools, pickle
from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    import casadi as ca
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)
    import casadi as ca
import torch
import torch.nn as nn

# ------------------------------------------------------------------ CONFIG
MODE = os.environ.get("CA_MODE", "fast")      # "smoke" | "fast" | "full"
CFGS = {
    "smoke": dict(seeds=[0],               scenarios=["nominal", "ice"],                          T_sim=4.0, n_train=40,  n_val=10, n_cal=20, epochs=40),
    "fast":  dict(seeds=[0, 1, 2],         scenarios=["nominal", "wet_heavy", "ice", "mu_switch"], T_sim=8.0, n_train=120, n_val=20, n_cal=40, epochs=120),
    "full":  dict(seeds=list(range(10)),   scenarios=["nominal", "wet_heavy", "ice", "mu_switch"], T_sim=8.0, n_train=300, n_val=40, n_cal=80, epochs=200),
}
CFG = CFGS[MODE]
OUT = "results"; os.makedirs(OUT, exist_ok=True)

DT = 0.1            # control / model step [s]
N = 25              # MPC horizon steps (2.5 s)
NX, NU = 8, 2       # states [x y psi vx vy r delta ax], inputs [delta_cmd, ax_cmd]
DYN = [3, 4, 5, 6, 7]   # states on which the residual acts
ND = len(DYN)
K_ENS, HID = 5, 32
SEED0 = 12345

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
seed_all(SEED0)
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
print("MODE:", MODE, CFG)
print("casadi", ca.__version__, "| torch", torch.__version__, "| cpu cores", os.cpu_count())

# %% [markdown]
# ## 1. Vehicle models
# *Plant (ground truth):* dynamic single-track model with **Pacejka magic-formula tyres whose peak scales with friction `mu`**, a
# **friction ellipse** (braking/accelerating reduces lateral capacity), **steering-rate limit**, first-order actuator lags,
# aerodynamic drag, rolling resistance, and the road-friction-limited acceleration.
#
# *Nominal model (what the controller knows):* **linear tyres**, different mass/inertia/cornering stiffness, different actuator
# time constants, no rate limit, no friction ellipse, no knowledge of `mu`. This is a deliberately realistic structural + parametric mismatch.

# %%
@dataclass
class Veh:
    lf: float = 1.2; lr: float = 1.5
    m0: float = 1500.0; Iz0: float = 2500.0
    g: float = 9.81
    # Pacejka (true)
    B: float = 9.0; C: float = 1.3; E: float = -0.5
    # true actuators / resistances
    tau_d: float = 0.12; tau_a: float = 0.25; rate_max: float = 0.8
    cd: float = 0.42; crr: float = 0.015
    # nominal model parameters (mismatched on purpose)
    m_nom: float = 1500.0; Iz_nom: float = 2500.0
    Cf_nom: float = 80000.0; Cr_nom: float = 72000.0
    tau_d_nom: float = 0.20; tau_a_nom: float = 0.35
    cd_nom: float = 0.30; crr_nom: float = 0.010
    # limits
    d_max: float = 0.35; ax_min: float = -6.0; ax_max: float = 3.0
V = Veh()
L_WB = V.lf + V.lr

def _smooth_vx(vx):
    return 0.5 * (vx + 1.0 + ca.sqrt((vx - 1.0) ** 2 + 0.1))

def _kin(s):
    x, y, psi, vx, vy, r = s[0], s[1], s[2], s[3], s[4], s[5]
    return ca.vertcat(vx * ca.cos(psi) - vy * ca.sin(psi), vx * ca.sin(psi) + vy * ca.cos(psi), r)

def _pac(alpha, D):
    Ba = V.B * alpha
    return D * ca.sin(V.C * ca.atan(Ba - V.E * (Ba - ca.atan(Ba))))

def f_plant_ct(s, u, th):
    """true continuous-time model. th = [mu, mass]"""
    mu, m = th[0], th[1]
    Iz = V.Iz0 * m / V.m0
    vx, vy, r, dl, ax = s[3], s[4], s[5], s[6], s[7]
    vxs = _smooth_vx(vx)
    Fzf = m * V.g * V.lr / L_WB; Fzr = m * V.g * V.lf / L_WB
    axlim = 0.95 * mu * V.g
    ax_eff = axlim * ca.tanh(ax / axlim)
    rho = ax_eff / (mu * V.g)
    ell = ca.sqrt(1.0 - rho ** 2)                      # friction ellipse factor
    af = dl - ca.atan((vy + V.lf * r) / vxs)
    ar = -ca.atan((vy - V.lr * r) / vxs)
    Fyf = ell * _pac(af, mu * Fzf); Fyr = ell * _pac(ar, mu * Fzr)
    dvx = ax_eff + vy * r - V.cd * vx ** 2 / m - V.crr * V.g
    dvy = (Fyf * ca.cos(dl) + Fyr) / m - vx * r
    dr = (V.lf * Fyf * ca.cos(dl) - V.lr * Fyr) / Iz
    ddl = V.rate_max * ca.tanh((u[0] - dl) / (V.tau_d * V.rate_max))
    dax = (u[1] - ax) / V.tau_a
    return ca.vertcat(_kin(s), dvx, dvy, dr, ddl, dax)

def f_nom_ct(s, u, th=None):
    """nominal (controller) model: linear tyres, wrong constants, no mu."""
    m, Iz = V.m_nom, V.Iz_nom
    vx, vy, r, dl, ax = s[3], s[4], s[5], s[6], s[7]
    vxs = _smooth_vx(vx)
    Fyf = V.Cf_nom * (dl - (vy + V.lf * r) / vxs)
    Fyr = V.Cr_nom * (-(vy - V.lr * r) / vxs)
    dvx = ax + vy * r - V.cd_nom * vx ** 2 / m - V.crr_nom * V.g
    dvy = (Fyf * ca.cos(dl) + Fyr) / m - vx * r
    dr = (V.lf * Fyf * ca.cos(dl) - V.lr * Fyr) / Iz
    ddl = (u[0] - dl) / V.tau_d_nom
    dax = (u[1] - ax) / V.tau_a_nom
    return ca.vertcat(_kin(s), dvx, dvy, dr, ddl, dax)

def make_step(f_ct, n_sub, name):
    s = ca.SX.sym("s", NX); u = ca.SX.sym("u", NU); th = ca.SX.sym("th", 2)
    h = DT / n_sub; x = s
    for _ in range(n_sub):
        k1 = f_ct(x, u, th); k2 = f_ct(x + h / 2 * k1, u, th)
        k3 = f_ct(x + h / 2 * k2, u, th); k4 = f_ct(x + h * k3, u, th)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function(name, [s, u, th], [x])

F_plant = make_step(f_plant_ct, 4, "F_plant")        # simulation truth (4 RK4 sub-steps)
F_true_mpc = make_step(f_plant_ct, 2, "F_true_mpc")  # oracle MPC model (true physics, MPC discretisation)
F_nom = make_step(f_nom_ct, 2, "F_nom")              # nominal MPC model
_map_cache = {}
def fmap(F, n):
    key = (F.name(), n)
    if key not in _map_cache:
        _map_cache[key] = F.map(n)
    return _map_cache[key]

def np_step(F, S, U, TH):
    """batched numpy step. S:(B,8) U:(B,2) TH:(B,2) -> (B,8)"""
    B = S.shape[0]
    return np.array(fmap(F, B)(S.T, U.T, TH.T)).T

TH_NOM_TRAIN = (0.9, 1500.0)
NOISE_PROC = np.array([0, 0, 0, 0, 0.02, 0.002, 0, 0.0])         # added after each plant step
NOISE_MEAS = np.array([0.03, 0.03, 0.002, 0.05, 0.03, 0.003, 0.001, 0.05])
print("models compiled")

# %% [markdown]
# ## 2. Data: excitation trajectories on the **training domain only** (high friction)
# Training domain: `mu ~ U[0.8, 1.0]`, `m ~ U[1450, 1550]`. The test scenarios (wet+heavy, ice, abrupt friction switch) are **out of distribution** by design.

# %%
def gen_traj(rng, mu, m, T_steps=100, noise=True):
    """open-loop excitation trajectory on the plant; returns measured states S (T+1,8), inputs U (T,2), truth Str"""
    th = np.array([mu, m])
    v_t = rng.uniform(8, 16)
    s = np.array([0, 0, 0, v_t, 0, 0, 0, 0.0])
    nsin = 3
    A = rng.uniform(0.0, 0.07, nsin) * (12.0 / v_t); fr = rng.uniform(0.1, 0.7, nsin); ph = rng.uniform(0, 2 * np.pi, nsin)
    step_t = rng.integers(10, 60); step_a = rng.uniform(-0.06, 0.06) * (12.0 / v_t) * (rng.random() < 0.6)
    Str = [s.copy()]; U = []
    for k in range(T_steps):
        t = k * DT
        dcmd = float((A * np.sin(2 * np.pi * fr * t + ph)).sum() + (step_a if k >= step_t else 0.0))
        if rng.random() < 0.15: dcmd *= rng.uniform(1.0, 1.8)
        acmd = float(np.clip(0.8 * (v_t - s[3]) + 0.8 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t), V.ax_min + 2, V.ax_max))
        dcmd = float(np.clip(dcmd, -V.d_max, V.d_max)); u = np.array([dcmd, acmd])
        s = np.array(F_plant(s, u, th)).ravel() + NOISE_PROC * rng.standard_normal(NX)
        U.append(u); Str.append(s.copy())
        if abs(s[4]) > 3.5 or s[3] < 3.0 or abs(s[2]) > 1.0: break
    Str = np.array(Str); U = np.array(U)
    S = Str + (NOISE_MEAS * rng.standard_normal(Str.shape) if noise else 0.0)
    return S, U, Str

def make_dataset(rng, n_traj, mu_rng=(0.8, 1.0), m_rng=(1450, 1550), T_steps=100):
    Z, Y = [], []
    trajs = []
    for _ in range(n_traj):
        mu = rng.uniform(*mu_rng); m = rng.uniform(*m_rng)
        S, U, _ = gen_traj(rng, mu, m, T_steps)
        if len(U) < 5: continue
        trajs.append((S, U, mu, m))
        TH = np.tile([TH_NOM_TRAIN[0], TH_NOM_TRAIN[1]], (len(U), 1))
        pred = np_step(F_nom, S[:-1], U, TH)
        Y.append(S[1:, DYN] - pred[:, DYN])
        Z.append(np.hstack([S[:-1, DYN], U]))
    return np.vstack(Z), np.vstack(Y), trajs

rng_data = np.random.default_rng(SEED0)
t0 = time.time()
Ztr, Ytr, traj_tr = make_dataset(rng_data, CFG["n_train"])
Zva, Yva, traj_va = make_dataset(rng_data, CFG["n_val"])
Zcal, Ycal, traj_cal = make_dataset(rng_data, CFG["n_cal"])                    # calibration (in-distribution)
Zood, Yood, traj_ood = make_dataset(rng_data, 20, mu_rng=(0.3, 0.45), m_rng=(1450, 1550))  # OOD diagnostic only
print(f"data: train {Ztr.shape}, val {Zva.shape}, cal {Zcal.shape}, ood {Zood.shape} in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 3. Deep-ensemble residual (PyTorch) and its CasADi-embeddable form
# `K=5` MLPs `7 -> 32 -> 32 -> 5` (tanh, so the NMPC stays smooth). The **ensemble mean is exactly linear in the concatenated
# last hidden layers**: `Delta_hat(z) = Theta * Phi(z)` with `Phi = [h2_1, ..., h2_K, 1]`. We keep `Phi` frozen and adapt `Theta` online with RLS (C1).

# %%
class MLP(nn.Module):
    def __init__(self, din, h, dout):
        super().__init__()
        self.l1 = nn.Linear(din, h); self.l2 = nn.Linear(h, h); self.l3 = nn.Linear(h, dout)
    def forward(self, x, feats=False):
        h2 = torch.tanh(self.l2(torch.tanh(self.l1(x))))
        return (self.l3(h2), h2) if feats else self.l3(h2)

z_mu, z_sd = Ztr.mean(0), Ztr.std(0) + 1e-6
y_sd = Ytr.std(0) + 1e-9
def zn(Z): return (Z - z_mu) / z_sd

def train_ensemble(Ztr, Ytr, Zva, Yva, epochs, seed=0):
    X = torch.tensor(zn(Ztr), dtype=torch.float32); Y = torch.tensor(Ytr / y_sd, dtype=torch.float32)
    Xv = torch.tensor(zn(Zva), dtype=torch.float32); Yv = torch.tensor(Yva / y_sd, dtype=torch.float32)
    members, curves = [], []
    for k in range(K_ENS):
        torch.manual_seed(seed * 100 + k)
        net = MLP(Ztr.shape[1], HID, ND); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        idx_boot = torch.randint(0, len(X), (len(X),))            # bootstrap
        Xb, Yb = X[idx_boot], Y[idx_boot]
        best, best_state, curve = 1e9, None, []
        for ep in range(epochs):
            perm = torch.randperm(len(Xb))
            for i in range(0, len(Xb), 256):
                b = perm[i:i + 256]
                loss = ((net(Xb[b]) - Yb[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            with torch.no_grad(): vl = ((net(Xv) - Yv) ** 2).mean().item()
            curve.append(vl)
            if vl < best: best, best_state = vl, {k_: v.clone() for k_, v in net.state_dict().items()}
        net.load_state_dict(best_state); members.append(net); curves.append(curve)
    return members, curves

t0 = time.time()
members, curves = train_ensemble(Ztr, Ytr, Zva, Yva, CFG["epochs"])
print(f"ensemble trained in {time.time()-t0:.1f}s | best val MSE (normalised): {[round(min(c),4) for c in curves]}")

def extract(members):
    W1 = [m.l1.weight.detach().numpy().astype(float) for m in members]; b1 = [m.l1.bias.detach().numpy().astype(float) for m in members]
    W2 = [m.l2.weight.detach().numpy().astype(float) for m in members]; b2 = [m.l2.bias.detach().numpy().astype(float) for m in members]
    W3 = [m.l3.weight.detach().numpy().astype(float) for m in members]; b3 = [m.l3.bias.detach().numpy().astype(float) for m in members]
    return W1, b1, W2, b2, W3, b3
W1, b1, W2, b2, W3, b3 = extract(members)
FDIM = K_ENS * HID + 1
THETA0 = np.hstack([np.hstack([w / K_ENS for w in W3]), (sum(b3) / K_ENS)[:, None]])   # (ND, FDIM)

def phi_np(Zn):
    """Zn:(B,7) normalised -> Phi (B,FDIM)"""
    feats = []
    for k in range(K_ENS):
        h1 = np.tanh(Zn @ W1[k].T + b1[k]); h2 = np.tanh(h1 @ W2[k].T + b2[k]); feats.append(h2)
    return np.hstack(feats + [np.ones((Zn.shape[0], 1))])

def member_np(Zn, k):
    h1 = np.tanh(Zn @ W1[k].T + b1[k]); h2 = np.tanh(h1 @ W2[k].T + b2[k]); return h2 @ W3[k].T + b3[k]

def resid_np(Z, Theta):
    return (phi_np(zn(Z)) @ Theta.T) * y_sd

def phi_ca(z):
    """CasADi version of Phi for a (7,1) raw input (SX or MX)"""
    zz = (z - ca.DM(z_mu)) / ca.DM(z_sd)
    feats = []
    for k in range(K_ENS):
        h1 = ca.tanh(ca.mtimes(ca.DM(W1[k]), zz) + ca.DM(b1[k]))
        h2 = ca.tanh(ca.mtimes(ca.DM(W2[k]), h1) + ca.DM(b2[k])); feats.append(h2)
    return ca.vertcat(*feats, 1.0)

def rmse(a): return float(np.sqrt((a ** 2).mean()))
def eval_one_step(Z, Y, Theta=THETA0):
    return rmse(Y / y_sd), rmse((Y - resid_np(Z, Theta)) / y_sd)
for name, (Z, Y) in {"val (in-dist)": (Zva, Yva), "OOD (mu 0.3-0.45)": (Zood, Yood)}.items():
    a, b = eval_one_step(Z, Y)
    print(f"1-step normalised RMSE | {name:20s} nominal {a:.3f} -> +NN residual {b:.3f}")

# %% [markdown]
# ## 4. Unit tests (numpy <-> CasADi consistency) — cheap insurance for reproducibility

# %%
def nn_step_ca_fn():
    s = ca.SX.sym("s", NX); u = ca.SX.sym("u", NU); th = ca.SX.sym("th", 2); Th = ca.SX.sym("Th", ND, FDIM)
    sn = F_nom(s, u, th)
    z = ca.vertcat(s[3:8], u)
    d = ca.mtimes(Th, phi_ca(z)) * ca.DM(y_sd)
    return ca.Function("F_nn", [s, u, th, Th], [sn + ca.vertcat(ca.DM.zeros(3, 1), d)])
F_nn = nn_step_ca_fn()

rng_t = np.random.default_rng(1)
S, U, _ = gen_traj(rng_t, 0.9, 1500.0, 30)
s0, u0 = S[3], U[3]
a = np.array(F_nn(s0, u0, [0.9, 1500.0], THETA0)).ravel()
b = np_step(F_nom, s0[None], u0[None], np.array([[0.9, 1500.0]]))[0] + np.concatenate([np.zeros(3), resid_np(np.hstack([s0[DYN], u0])[None], THETA0)[0]])
assert np.allclose(a, b, atol=1e-8), (a - b)
# plant energy/sanity: straight line, zero input -> only drag decelerates
sx = np.array([0, 0, 0, 15, 0, 0, 0, 0.0]); sx2 = np.array(F_plant(sx, [0, 0], [0.9, 1500.0])).ravel()
assert abs(sx2[1]) < 1e-9 and abs(sx2[4]) < 1e-9 and sx2[3] < 15
# steady-state turn is left-turning for positive steering
sy = np.array([0, 0, 0, 12, 0, 0, 0.05, 0.0])
for _ in range(30): sy = np.array(F_plant(sy, [0.05, 0.0], [0.9, 1500.0])).ravel()
assert sy[5] > 0.05 and sy[4] != 0
print("unit tests passed ✔  (numpy NN == CasADi NN; plant sanity)")
