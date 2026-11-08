# Extracted verbatim from theory_flat.py (home plant only): plant, nominal model, data generator.
import os, sys, time, math, pickle, warnings
from dataclasses import dataclass
import numpy as np
warnings.filterwarnings("ignore")
import casadi as ca
PLANT = "home"
DT = 0.1; N = 25; NX, NU = 8, 2; DYN = [3, 4, 5, 6, 7]; ND = len(DYN); SEED0 = 12345

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
    Cf_nom: float = 80000.0; Cr_nom: float = 72000.0; mu0_nom: float = 0.9
    tau_d_nom: float = 0.20; tau_a_nom: float = 0.35
    cd_nom: float = 0.30; crr_nom: float = 0.010
    # limits
    d_max: float = 0.35; ax_min: float = -6.0; ax_max: float = 3.0
if PLANT == "mb":
    try:
        from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "commonroad-vehicle-models"], check=True)
        from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
    from vehiclemodels.init_mb import init_mb
    from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
    _P0 = parameters_vehicle2()          # BMW-320i-like parameter set shipped with CommonRoad vehicle models
    # the controller knows the vehicle's geometry/mass/inertia but NOT its tyre law: nominal tyre = saturating tanh with a *mismatched* stiffness
    V = Veh(lf=_P0.a, lr=_P0.b, m0=_P0.m, Iz0=_P0.I_z, m_nom=_P0.m, Iz_nom=_P0.I_z, Cf_nom=1.0e5, Cr_nom=9.0e4,
            B=abs(_P0.tire.p_ky1) / (_P0.tire.p_cy1 * _P0.tire.p_dy1), C=_P0.tire.p_cy1, E=_P0.tire.p_ey1)     # (B, C, E) only used by the single-track *oracle*
    ORACLE_MU_GAIN = _P0.tire.p_dy1 / 0.9
else:
    V = Veh(); ORACLE_MU_GAIN = 1.0
L_WB = V.lf + V.lr

def _smooth_vx(vx):
    return 0.5 * (vx + 1.0 + ca.sqrt((vx - 1.0) ** 2 + 0.1))

def _kin(s):
    x, y, psi, vx, vy, r = s[0], s[1], s[2], s[3], s[4], s[5]
    return ca.vertcat(vx * ca.cos(psi) - vy * ca.sin(psi), vx * ca.sin(psi) + vy * ca.cos(psi), r)

def _pac(alpha, D, cs=1.0):
    Ba = V.B * cs * alpha
    return D * ca.sin(V.C * ca.atan(Ba - V.E * (Ba - ca.atan(Ba))))

def f_plant_ct(s, u, th):
    """true continuous-time model. th = [mu, mass, cornering-stiffness scale cs, CoG shift (positive = CoG moves rearwards)]"""
    mu, m, cs, cog = th[0], th[1], th[2], th[3]
    Iz = V.Iz0 * m / V.m0
    lf, lr = V.lf + cog, V.lr - cog
    vx, vy, r, dl, ax = s[3], s[4], s[5], s[6], s[7]
    vxs = _smooth_vx(vx)
    Fzf = m * V.g * lr / L_WB; Fzr = m * V.g * lf / L_WB
    axlim = 0.95 * mu * V.g
    ax_eff = axlim * ca.tanh(ax / axlim)
    rho = ax_eff / (mu * V.g)
    ell = ca.sqrt(1.0 - rho ** 2)                      # friction ellipse factor
    af = dl - ca.atan((vy + lf * r) / vxs)
    ar = -ca.atan((vy - lr * r) / vxs)
    Fyf = ell * _pac(af, mu * Fzf, cs); Fyr = ell * _pac(ar, mu * Fzr, cs)
    dvx = ax_eff + vy * r - V.cd * vx ** 2 / m - V.crr * V.g
    dvy = (Fyf * ca.cos(dl) + Fyr) / m - vx * r
    dr = (lf * Fyf * ca.cos(dl) - lr * Fyr) / Iz
    ddl = V.rate_max * ca.tanh((u[0] - dl) / (V.tau_d * V.rate_max))
    dax = (u[1] - ax) / V.tau_a
    return ca.vertcat(_kin(s), dvx, dvy, dr, ddl, dax)

def f_nom_ct(s, u, th):
    """nominal (controller) model: SATURATING (tanh) tyres assuming dry road mu0=0.9, wrong stiffness/inertia/lag constants,
    no friction ellipse, no steering-rate limit. th[0] = xi is the friction scale (xi=1: dry). The controller never sees the true mu."""
    xi = th[0]
    m, Iz = V.m_nom, V.Iz_nom
    vx, vy, r, dl, ax = s[3], s[4], s[5], s[6], s[7]
    vxs = _smooth_vx(vx)
    Fzf = m * V.g * V.lr / L_WB; Fzr = m * V.g * V.lf / L_WB
    D0f, D0r = V.mu0_nom * Fzf, V.mu0_nom * Fzr
    af = dl - ca.atan((vy + V.lf * r) / vxs); ar = -ca.atan((vy - V.lr * r) / vxs)
    Fyf = xi * D0f * ca.tanh(V.Cf_nom * af / D0f); Fyr = xi * D0r * ca.tanh(V.Cr_nom * ar / D0r)
    axlim = xi * 0.95 * V.mu0_nom * V.g
    ax_eff = axlim * ca.tanh(ax / axlim)
    dvx = ax_eff + vy * r - V.cd_nom * vx ** 2 / m - V.crr_nom * V.g
    dvy = (Fyf * ca.cos(dl) + Fyr) / m - vx * r
    dr = (V.lf * Fyf * ca.cos(dl) - V.lr * Fyr) / Iz
    ddl = (u[0] - dl) / V.tau_d_nom
    dax = (u[1] - ax) / V.tau_a_nom
    return ca.vertcat(_kin(s), dvx, dvy, dr, ddl, dax)

def make_step(f_ct, n_sub, name, n_th=2):
    s = ca.SX.sym("s", NX); u = ca.SX.sym("u", NU); th = ca.SX.sym("th", n_th)
    h = DT / n_sub; x = s
    for _ in range(n_sub):
        k1 = f_ct(x, u, th); k2 = f_ct(x + h / 2 * k1, u, th)
        k3 = f_ct(x + h / 2 * k2, u, th); k4 = f_ct(x + h * k3, u, th)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function(name, [s, u, th], [x])

F_plant = make_step(f_plant_ct, 4, "F_plant", 4)        # simulation truth of the "home" plant (4 RK4 sub-steps)
F_true_mpc = make_step(f_plant_ct, 2, "F_true_mpc", 4)  # oracle MPC model (single-track physics with the true mu/mass/stiffness/CoG, MPC discretisation)
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

TH_NOM_TRAIN = (1.0, 0.0)      # nominal model parameter vector: [xi (friction scale), unused]
OFFLINE_NOISE = 0.2
NOISE_PROC = np.array([0, 0, 0, 0, 0.02, 0.002, 0, 0.0])         # added after each plant step
NOISE_MEAS = np.array([0.03, 0.03, 0.002, 0.05, 0.03, 0.003, 0.001, 0.05])
print("models compiled")

def mu_at(sc, t):
    return sc["mu2"] if ("mu2" in sc and t >= sc["t_switch"]) else sc["mu"]

def apply_fault(sc, u, t):
    """actuator fault (unmodelled): after fault_t the steering actuator loses gain and gets an offset"""
    if "fault_t" in sc and t >= sc["fault_t"]:
        return np.array([sc.get("steer_gain", 1.0) * u[0] + sc.get("steer_off", 0.0), u[1]])
    return u

class HomePlant:
    def __init__(self, sc): self.sc = sc
    def reset(self, v0): self.s = np.array([0, 0, 0, v0, 0, 0, 0, 0.0]); return self.s.copy()
    def step(self, u, t, pn, dvy=0.0):
        sc = self.sc; th = np.array([mu_at(sc, t), sc["m"], sc.get("cs", 1.0), sc.get("cog", 0.0)]); u = apply_fault(sc, u, t)
        self.s = np.array(F_plant(self.s, u, th)).ravel() + pn
        self.s[4] += dvy
        return self.s.copy()
def make_plant(sc): return MBPlant(sc) if PLANT == "mb" else HomePlant(sc)

AMP = float(os.environ.get('TH_AMP', '2.2'))       # excitation aggressiveness (x the benchmark's training excitation) so that a grip loss is actually felt
def gen_traj2(rng, sc1, sc2=None, ts=None, T_steps=120, noise_scale=1.0):
    plant = make_plant(sc1)
    v_t = rng.uniform(8, 16); s = plant.reset(v_t)
    nsin = 3
    A = AMP * rng.uniform(0.0, 0.07, nsin) * (12.0 / v_t); fr = rng.uniform(0.1, 0.7, nsin); ph = rng.uniform(0, 2 * np.pi, nsin)
    step_t = rng.integers(10, 60); step_a = AMP * rng.uniform(-0.06, 0.06) * (12.0 / v_t) * (rng.random() < 0.6)
    Str = [s.copy()]; U = []
    for k in range(T_steps):
        if sc2 is not None and k == ts: plant.sc = sc2
        t = k * DT
        dcmd = float((A * np.sin(2 * np.pi * fr * t + ph)).sum() + (step_a if k >= step_t else 0.0))
        if rng.random() < 0.15: dcmd *= rng.uniform(1.0, 1.8)
        acmd = float(np.clip(0.8 * (v_t - s[3]) + 0.8 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t), V.ax_min + 2, V.ax_max))
        dcmd = float(np.clip(dcmd, -V.d_max, V.d_max)); u = np.array([dcmd, acmd])
        s = plant.step(u, t, NOISE_PROC * rng.standard_normal(NX))
        U.append(u); Str.append(s.copy())
        if abs(s[4]) > 3.5 or s[3] < 3.0 or abs(s[2]) > 1.0 or not np.all(np.isfinite(s)): break
    Str = np.array(Str); U = np.array(U)
    S = Str + noise_scale * NOISE_MEAS * rng.standard_normal(Str.shape)
    return S, U, Str

KINDS = ["id", "mu", "ice", "cs", "mass"]
def make_episode(rng, kind, T_steps, min_after=N + 25):
    for attempt in range(60):
        base = dict(mu=float(rng.uniform(0.8, 1.0)), m=float(rng.uniform(1450, 1550)))
        ts = int(rng.integers(30, 46)); sc2 = None
        if kind == "mu":     sc2 = dict(base, mu=float(rng.uniform(0.35, 0.8)))
        elif kind == "ice":  sc2 = dict(base, mu=float(rng.uniform(0.2, 0.3)))
        elif kind == "cs":   sc2 = dict(base, cs=float(rng.uniform(0.40, 0.55)))
        elif kind == "mass": sc2 = dict(base, m=float(base["m"] * rng.uniform(1.40, 1.60)))
        S, U, _ = gen_traj2(rng, base, sc2, ts, T_steps)
        if len(U) >= (ts + min_after if sc2 else N + 60):
            return dict(S=S, U=U, ts=(ts if sc2 else None), kind=kind, attempts=attempt + 1)
    return None

def rollout_nom(S0, Useq, xi=1.0):
    B, L = Useq.shape[0], Useq.shape[1]
    out = np.zeros((B, L + 1, NX)); out[:, 0] = S0; S = S0.copy(); TH_ = np.tile([xi, 0.0], (B, 1))
    for l in range(L):
        S = np_step(F_nom, S, Useq[:, l], TH_); out[:, l + 1] = S
    return out

