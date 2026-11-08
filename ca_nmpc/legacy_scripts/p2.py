# %% [markdown]
# ## 5. Scenarios (the distribution-shift benchmark)
# A 2-lane road, two static obstacles that force an evasive double lane-change, and four conditions:
#
# | scenario | friction `mu` | mass | what it tests |
# |---|---|---|---|
# | `nominal` | 0.90 | 1500 kg | in-distribution sanity |
# | `wet_heavy` | 0.55 | 1800 kg | parametric shift (OOD) |
# | `ice` | 0.35 | 1500 kg | strong shift: handling limit is ~3.4 m/s² |
# | `mu_switch` | 0.90 → 0.40 at t = 1.5 s | 1500 kg | *abrupt* non-stationary shift while the evasive manoeuvre starts |
#
# Per-seed randomisation: obstacle positions, initial speed, sensor noise and process disturbances. **All controllers share the same random streams per seed (paired design).**

# %%
R_O, R_V = 0.8, 1.0
Y_LO, Y_HI = -1.75, 5.25
V_REF = 12.0
M_OBS = 2
SCEN = {
    "nominal":   dict(mu=0.90, m=1500.0),
    "wet_heavy": dict(mu=0.55, m=1800.0),
    "ice":       dict(mu=0.35, m=1500.0),
    "mu_switch": dict(mu=0.90, m=1500.0, mu2=0.40, t_switch=1.5),
}
def mu_at(sc, t):
    return sc["mu2"] if ("mu2" in sc and t >= sc["t_switch"]) else sc["mu"]

def make_world(seed):
    rng = np.random.default_rng(50_000 + seed)
    obs = np.array([[45.0 + rng.uniform(-2, 2), 0.0 + rng.uniform(-0.3, 0.3)],
                    [82.0 + rng.uniform(-2, 2), 3.0 + rng.uniform(-0.3, 0.3)]])
    v0 = 12.0 + rng.uniform(-0.5, 0.5)
    n_steps = int(round(CFG["T_sim"] / DT))
    meas_noise = rng.standard_normal((n_steps + 1, NX)) * NOISE_MEAS
    proc_noise = rng.standard_normal((n_steps + 1, NX)) * NOISE_PROC
    return obs, v0, meas_noise, proc_noise

# %% [markdown]
# ## 6. The NMPC (CasADi/IPOPT) — one NLP structure for all controllers
# * Decision variables: states `X`, inputs `U`, slacks for obstacle/road constraints.
# * **Constraint form (C4):** for every barrier `h` (obstacle clearance, road edge distance) with conformal margin `q_j`:
#   `h~_j = h(x_j) - q_j`  and  `h~_j >= (1-gamma) h~_{j-1} - slack`.
#   `gamma = 1` is the classical hard constraint `h~_j >= 0`; `gamma < 1` is the discrete-time CBF form (MPC-CBF).
# * Dynamics model is switchable: **nominal**, **nominal + neural residual (Theta is a runtime parameter, so RLS never triggers a rebuild)**, or **oracle (true physics + true mu)**.

# %%
BSEL = np.zeros((NX, ND)); BSEL[DYN, np.arange(ND)] = 1.0

class NMPC:
    def __init__(self, kind, n=N):
        assert kind in ("nom", "nn", "oracle")
        self.kind, self.n = kind, n
        opti = ca.Opti()
        X = opti.variable(NX, n + 1); U = opti.variable(NU, n)
        S_obs = opti.variable(M_OBS, n); S_rd = opti.variable(2, n)
        p_s0 = opti.parameter(NX); p_uprev = opti.parameter(NU); p_obs = opti.parameter(M_OBS, 2)
        p_q = opti.parameter(n); p_gamma = opti.parameter(1)
        p_th = opti.parameter(2) if kind == "oracle" else None
        p_Th = opti.parameter(ND, FDIM) if kind == "nn" else None
        th0 = ca.DM([0.9, 1500.0])
        opti.subject_to(X[:, 0] == p_s0)
        for j in range(n):
            x, u = X[:, j], U[:, j]
            if kind == "nom":
                xn = F_nom(x, u, th0)
            elif kind == "oracle":
                xn = F_true_mpc(x, u, p_th)
            else:
                z = ca.vertcat(x[3:8], u)
                xn = F_nom(x, u, th0) + ca.mtimes(ca.DM(BSEL), ca.mtimes(p_Th, phi_ca(z)) * ca.DM(y_sd))
            opti.subject_to(X[:, j + 1] == xn)
        opti.subject_to(opti.bounded(-V.d_max, U[0, :], V.d_max))
        opti.subject_to(opti.bounded(V.ax_min, U[1, :], V.ax_max))
        opti.subject_to(opti.bounded(-0.5, X[6, :], 0.5))
        opti.subject_to(opti.bounded(1.0, X[3, :], 30.0))
        opti.subject_to(S_obs >= 0); opti.subject_to(S_rd >= 0)

        def barriers(px, py, q):
            hs = []
            for m in range(M_OBS):
                hs.append(ca.sqrt((px - p_obs[m, 0]) ** 2 + (py - p_obs[m, 1]) ** 2 + 1e-4) - (R_O + R_V) - q)
            hs.append(py - (Y_LO + R_V) - q)
            hs.append((Y_HI - R_V) - py - q)
            return hs
        h_prev = barriers(X[0, 0], X[1, 0], 0.0)
        for j in range(1, n + 1):
            hj = barriers(X[0, j], X[1, j], p_q[j - 1])
            for i in range(M_OBS + 2):
                sl = S_obs[i, j - 1] if i < M_OBS else S_rd[i - M_OBS, j - 1]
                opti.subject_to(hj[i] >= (1 - p_gamma) * h_prev[i] - sl)
            h_prev = hj
        # cost
        cost = 0
        for j in range(1, n + 1):
            w = 3.0 if j == n else 1.0
            cost += w * (1.0 * X[1, j] ** 2 + 4.0 * X[2, j] ** 2 + 0.5 * (X[3, j] - V_REF) ** 2 + 1.0 * X[5, j] ** 2)
        for j in range(n):
            du = U[:, j] - (p_uprev if j == 0 else U[:, j - 1])
            cost += 40.0 * U[0, j] ** 2 + 0.2 * U[1, j] ** 2 + 3000.0 * du[0] ** 2 + 1.0 * du[1] ** 2
        cost += 1e4 * (ca.sum1(ca.sum2(S_obs)) + ca.sum1(ca.sum2(S_rd))) + 1e3 * (ca.sumsqr(S_obs) + ca.sumsqr(S_rd))
        opti.minimize(cost)
        opti.solver("ipopt", {"expand": True, "print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
                              "ipopt.max_iter": 60, "ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-2,
                              "ipopt.acceptable_iter": 3, "ipopt.mu_strategy": "adaptive"})
        self.opti, self.X, self.U, self.S_obs, self.S_rd = opti, X, U, S_obs, S_rd
        self.p = dict(s0=p_s0, uprev=p_uprev, obs=p_obs, q=p_q, gamma=p_gamma, th=p_th, Th=p_Th)
        self.reset()

    def reset(self):
        self.X0, self.U0 = None, None

    def solve(self, s0, uprev, obs, q, gamma, th=None, Theta=None):
        o, p = self.opti, self.p
        o.set_value(p["s0"], s0); o.set_value(p["uprev"], uprev); o.set_value(p["obs"], obs)
        o.set_value(p["q"], q); o.set_value(p["gamma"], gamma)
        if p["th"] is not None: o.set_value(p["th"], th)
        if p["Th"] is not None: o.set_value(p["Th"], Theta)
        if self.X0 is None:
            Xg = np.tile(s0[:, None], (1, self.n + 1)); Ug = np.tile(uprev[:, None], (1, self.n))
        else:
            Xg = np.hstack([self.X0[:, 1:], self.X0[:, -1:]]); Xg[:, 0] = s0; Ug = np.hstack([self.U0[:, 1:], self.U0[:, -1:]])
        o.set_initial(self.X, Xg); o.set_initial(self.U, Ug)
        o.set_initial(self.S_obs, 0); o.set_initial(self.S_rd, 0)
        t0 = time.perf_counter(); ok = True
        try:
            sol = o.solve(); Xs, Us = sol.value(self.X), sol.value(self.U)
            slack = float(sol.value(self.S_obs).sum() + sol.value(self.S_rd).sum())
        except RuntimeError:
            ok = False
            try:
                Xs, Us = o.debug.value(self.X), o.debug.value(self.U)
                slack = float(o.debug.value(self.S_obs).sum() + o.debug.value(self.S_rd).sum())
            except Exception:
                Xs, Us, slack = Xg, Ug, np.nan
        dt = time.perf_counter() - t0
        Xs = np.asarray(Xs).reshape(NX, self.n + 1); Us = np.asarray(Us).reshape(NU, self.n)
        if not np.all(np.isfinite(Xs)) or not np.all(np.isfinite(Us)):
            Xs, Us, ok = Xg, Ug, False
        self.X0, self.U0 = Xs, Us
        return dict(u=Us[:, 0].copy(), X=Xs, U=Us, ok=ok, time=dt, slack=slack)

t0 = time.time()
MPCS = {k: NMPC(k) for k in ("nom", "nn", "oracle")}
print(f"NMPC built for 3 model kinds in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 7. Online learning and uncertainty quantification
# ### 7.1 RLS adaptation of the residual's last layer (C1)
# ### 7.2 Model-side multi-step scores
# At time `t` and for every horizon `j = 1..N` we re-roll the *current* model from the measured state at `t-j` along the *executed* inputs and compare with the measured position at `t`. These scores depend only on model error — not on what the controller planned — which keeps them (approximately) exchangeable across controllers.
# ### 7.3 Horizon-grouped, normalised ACI with shift propagation (C2, C3)
# * Normalise scores by the in-distribution median growth curve `g_j`.
# * Three horizon groups `[1-4], [5-10], [11-25]`, each with an ACI-controlled quantile `kappa_g` (`alpha_{t+1} = alpha_t + eta (alpha - err_t)`).
# * **Propagation:** `kappa_g >= kappa_g^cal * infl_{g-1}`: inflation seen at short horizons (observed after 1-4 steps) is imposed on longer horizons (observed after up to 25 steps).
# * Tube: `q_j = min(q_cap, kappa_{g(j)} * g_j)`.

# %%
class RLS:
    def __init__(self, Theta0, lam=0.985, p0=0.05, trace_cap=None):
        self.Theta = Theta0.copy(); self.P = np.eye(Theta0.shape[1]) * p0; self.lam = lam; self.p0 = p0
        self.tcap = trace_cap if trace_cap else Theta0.shape[1] * p0
    def update(self, phi, y):
        e = np.clip(y - self.Theta @ phi, -6, 6)
        Pphi = self.P @ phi; k = Pphi / (self.lam + phi @ Pphi)
        self.Theta = self.Theta + np.outer(e, k)
        self.P = (self.P - np.outer(k, Pphi)) / self.lam
        tr = np.trace(self.P)
        if tr > self.tcap: self.P *= self.tcap / tr          # anti-windup
        return float(np.abs(e).mean())

def rollout_np(S0, Useq, Theta, use_nn=True, member=None):
    """batched open-loop rollout. S0:(B,8) Useq:(B,L,2) -> (B,L+1,8)"""
    B, L = Useq.shape[0], Useq.shape[1]
    out = np.zeros((B, L + 1, NX)); out[:, 0] = S0; S = S0.copy()
    TH = np.tile([0.9, 1500.0], (B, 1))
    for l in range(L):
        Sn = np_step(F_nom, S, Useq[:, l], TH)
        if use_nn:
            Z = np.hstack([S[:, DYN], Useq[:, l]])
            if member is None: d = (phi_np(zn(Z)) @ Theta.T) * y_sd
            else: d = member_np(zn(Z), member) * y_sd
            Sn[:, DYN] += d
        S = Sn; out[:, l + 1] = S
    return out

def multistep_scores(S_hist, U_hist, t, Theta, n=N, use_nn=True):
    """e[j-1] = || model_j-step_pred(from t-j) - measured position at t ||, NaN if not yet available. S_hist has t+1 rows, U_hist t rows."""
    e = np.full(n, np.nan); B = min(n, t)
    if B < 1: return e
    S0 = np.array([S_hist[t - j] for j in range(1, B + 1)])
    Useq = np.zeros((B, B, NU))
    for j in range(1, B + 1): Useq[j - 1, :j] = np.array(U_hist[t - j:t])
    R = rollout_np(S0, Useq, Theta, use_nn)
    for j in range(1, B + 1): e[j - 1] = np.linalg.norm(R[j - 1, j, :2] - S_hist[t][:2])
    return e

# --- offline calibration scores (in-distribution, full online noise) with the OFFLINE model THETA0
def calibration_scores(trajs, Theta, n=N):
    sc = [[] for _ in range(n)]
    for (S, U, mu, m) in trajs:
        T = len(U)
        for i in range(0, T - 1):
            L = min(n, T - i)
            R = rollout_np(S[i][None], U[i:i + L][None], Theta)[0]
            for j in range(1, L + 1): sc[j - 1].append(np.linalg.norm(R[j, :2] - S[i + j, :2]))
    return [np.array(s) for s in sc]
t0 = time.time()
CAL_SCORES = calibration_scores(traj_cal, THETA0)
print(f"calibration scores: {[len(s) for s in CAL_SCORES][:3]}... in {time.time()-t0:.1f}s | median/95% position error at j=1,5,10,25:",
      [(round(float(np.median(CAL_SCORES[j-1])), 3), round(float(np.quantile(CAL_SCORES[j-1], .95)), 3)) for j in (1, 5, 10, 25)])

GROUPS = [(1, 4), (5, 10), (11, 25)]
def group_of(j):
    for gi, (a, b) in enumerate(GROUPS):
        if a <= j <= b: return gi

class ConformalTube:
    """mode: 'static' (split CP, no online update) | 'aci' (online, no propagation) | 'aci_prop' (ours)"""
    def __init__(self, cal_scores, mode="aci_prop", alpha=0.10, eta=0.05, q_cap=3.0, win_per_j=20, seed=0, q_floor=0.0):
        self.mode, self.alpha, self.eta, self.q_cap, self.q_floor = mode, alpha, eta, q_cap, q_floor
        n = len(cal_scores)
        self.g = np.maximum(np.array([np.median(s) for s in cal_scores]), 1e-3)
        rng = np.random.default_rng(seed)
        self.win, self.kappa_cal = [], np.zeros(len(GROUPS))
        for gi, (a, b) in enumerate(GROUPS):
            pool = np.concatenate([cal_scores[j - 1] / self.g[j - 1] for j in range(a, b + 1)])
            self.kappa_cal[gi] = np.quantile(pool, 1 - alpha)
            W = win_per_j * (b - a + 1)
            self.win.append(deque(rng.choice(pool, size=min(W, len(pool)), replace=False).tolist(), maxlen=W))
        self.kappa = self.kappa_cal.copy(); self.alpha_t = np.full(len(GROUPS), alpha)
        self.n = n
    def margins(self):
        q = np.array([min(self.q_cap, self.kappa[group_of(j)] * self.g[j - 1]) for j in range(1, self.n + 1)])
        return np.maximum(q, self.q_floor)
    def update(self, e):
        if self.mode == "static": return
        for gi, (a, b) in enumerate(GROUPS):
            s = e[a - 1:b]; s = s[~np.isnan(s)] / self.g[a - 1:b][~np.isnan(e[a - 1:b])]
            if len(s) == 0: continue
            err = float(np.mean(s > self.kappa[gi]))
            self.alpha_t[gi] = np.clip(self.alpha_t[gi] + self.eta * (self.alpha - err), 0.005, 0.5)
            self.win[gi].extend(s.tolist())
            self.kappa[gi] = float(np.quantile(np.array(self.win[gi]), 1 - self.alpha_t[gi]))
        if self.mode == "aci_prop":
            infl = self.kappa / self.kappa_cal
            for gi in range(1, len(GROUPS)):
                self.kappa[gi] = max(self.kappa[gi], self.kappa_cal[gi] * max(1.0, infl[gi - 1]))
                infl[gi] = self.kappa[gi] / self.kappa_cal[gi]

def ensemble_margins(s_now, U_plan, kappa_e=3.0, q_cap=3.0, n=N):
    """uncalibrated baseline: spread of the K ensemble members' predicted positions along the planned inputs"""
    Useq = np.tile(U_plan.T[None], (K_ENS, 1, 1))
    R = np.stack([rollout_np(s_now[None], Useq[k:k + 1], None, True, member=k)[0] for k in range(K_ENS)])  # (K, n+1, 8)
    P = R[:, 1:, :2]; sig = np.linalg.norm(P - P.mean(0, keepdims=True), axis=2).max(0)
    return np.minimum(q_cap, kappa_e * sig)
print("online-learning components ready")
