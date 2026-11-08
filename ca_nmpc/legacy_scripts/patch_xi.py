import re
# ===================================================== p1
s = open('p1.py').read()
s = s.replace("    Cf_nom: float = 80000.0; Cr_nom: float = 72000.0\n", "    Cf_nom: float = 80000.0; Cr_nom: float = 72000.0; mu0_nom: float = 0.9\n")
a = s.index("def f_nom_ct(s, u, th=None):"); b = s.index("def make_step(")
new_nom = '''def f_nom_ct(s, u, th):
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

'''
s = s[:a] + new_nom + s[b:]
s = s.replace("TH_NOM_TRAIN = (0.9, 1500.0)", "TH_NOM_TRAIN = (1.0, 0.0)      # nominal model parameter vector: [xi (friction scale), unused]")
s = s.replace("TH = np.tile([TH_NOM_TRAIN[0], TH_NOM_TRAIN[1]], (len(U), 1))", "TH = np.tile(TH_NOM_TRAIN, (len(U), 1))")
# features
a = s.index("# ---- physics-informed basis"); b = s.index("def member_np(")
new_feat = '''FDIM = HID_S + 1                                           # Phi = [h2 (48) | 1]
THETA0 = np.hstack([W3, b3[:, None]])                      # (ND, FDIM)  -- frozen after offline training

def phi_np(Z):
    """Z raw (B,7) -> Phi (B,FDIM) = [h2, 1]"""
    Zn = zn(Z)
    h1 = np.tanh(Zn @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2)
    return np.hstack([h2, np.ones((Z.shape[0], 1))])

'''
s = s[:a] + new_feat + s[b:]
a = s.index("def phi_ca(z):"); b = s.index("def rmse(a):")
new_phi_ca = '''def phi_ca(z):
    """CasADi version of Phi for a (7,1) raw input (SX or MX)"""
    zz = (z - ca.DM(z_mu)) / ca.DM(z_sd)
    h1 = ca.tanh(ca.mtimes(ca.DM(W1), zz) + ca.DM(b1))
    h2 = ca.tanh(ca.mtimes(ca.DM(W2), h1) + ca.DM(b2))
    return ca.vertcat(h2, 1.0)

'''
s = s[:a] + new_phi_ca + s[b:]
s = s.replace("a = np.array(F_nn(s0, u0, [0.9, 1500.0], THETA0)).ravel()", "a = np.array(F_nn(s0, u0, [1.0, 0.0], THETA0)).ravel()")
s = s.replace("np_step(F_nom, s0[None], u0[None], np.array([[0.9, 1500.0]]))[0]", "np_step(F_nom, s0[None], u0[None], np.array([[1.0, 0.0]]))[0]")
# markdown edits
s = s.replace("""# *Nominal model (what the controller knows):* **linear tyres**, different mass/inertia/cornering stiffness, different actuator
# time constants, no rate limit, no friction ellipse, no knowledge of `mu`. This is a deliberately realistic structural + parametric mismatch.""",
"""# *Nominal model (what the controller knows):* a **saturating (tanh) tyre model that assumes a dry road (`mu0 = 0.9`)**, with a *friction-scale parameter `xi`* (`xi = 1` dry) that the controller can identify online,
# but with different stiffness/inertia/actuator constants, no friction ellipse and no steering-rate limit. The controller never sees the true `mu`.
# (Using a saturating nominal tyre makes the baselines *strong*: the only first-order unknown is grip, as in practice.)""")
s = s.replace("""# 2. Learns a **deep-ensemble neural residual** `f_true - f_nominal` offline on *high-friction* data only.
# 3. Embeds the residual inside a **CasADi NMPC**, and adapts its **last layer online with RLS**.""","""# 2. Learns a **deep-ensemble neural residual** `f_true - f_nominal` offline on *high-friction* data only (distilled into one student for speed).
# 3. Embeds nominal + residual inside a **CasADi NMPC** and **identifies the friction scale online** with a scalar RLS (physics-structured, 1 parameter).""")
s = s.replace("""# * **C1 – Online last-layer adaptation** of an ensemble-mean residual inside the MPC (RLS, no re-compilation of the NLP).""",
"""# * **C1 – Physics-structured online adaptation**: the one-step tyre-force sensitivity gives a *1-parameter linear-in-parameters* regression for the friction scale `xi`
#   (scalar RLS with forgetting) on top of a frozen neural residual; `xi` is a runtime parameter of the NLP (no re-compilation).""")
s = s.replace("""# The student is **linear in its last layer**""", """# (The student is **linear in its last layer**""",1) if False else s
open('p1.py', 'w').write(s)

# ===================================================== p2
p = open('p2.py').read()
p = p.replace("        p_th = opti.parameter(2) if kind == \"oracle\" else None\n        p_Th = opti.parameter(ND, FDIM) if kind == \"nn\" else None\n        th0 = ca.DM([0.9, 1500.0])",
              "        p_th = opti.parameter(2) if kind == \"oracle\" else None\n        p_Th = opti.parameter(ND, FDIM) if kind == \"nn\" else None\n        p_xi = opti.parameter(1) if kind in (\"nom\", \"nn\") else None\n        th0 = ca.vertcat(p_xi, 0.0) if p_xi is not None else None")
p = p.replace("self.p = dict(s0=p_s0, uprev=p_uprev, obs=p_obs, q=p_q, gamma=p_gamma, th=p_th, Th=p_Th)", "self.p = dict(s0=p_s0, uprev=p_uprev, obs=p_obs, q=p_q, gamma=p_gamma, th=p_th, Th=p_Th, xi=p_xi)")
p = p.replace("    def solve(self, s0, uprev, obs, q, gamma, th=None, Theta=None):", "    def solve(self, s0, uprev, obs, q, gamma, th=None, Theta=None, xi=1.0):")
p = p.replace("        if p[\"Th\"] is not None: o.set_value(p[\"Th\"], Theta)\n", "        if p[\"Th\"] is not None: o.set_value(p[\"Th\"], Theta)\n        if p[\"xi\"] is not None: o.set_value(p[\"xi\"], xi)\n")
a = p.index("RLS_CFG = dict"); b = p.index("def rollout_np")
new_est = '''XI_CFG = dict(lam=0.93, p0=1.0, p_max=3.0, bounds=(0.12, 1.3), w=(1 / 0.05, 1 / 0.005))
class XiEstimator:
    """Scalar RLS (with forgetting) for the friction scale xi.
    Tyre forces are exactly linear in xi, so the one-step residual w.r.t. the xi=1 model is (xi-1)*g(s,u) with g the one-step tyre-force sensitivity
    (computed from the model by finite differences). We regress the lateral (v_y) and yaw-rate (r) channels, weighted by their sensor noise."""
    def __init__(self, Theta0, **kw):
        c = dict(XI_CFG); c.update(kw); self.c = c; self.Theta0 = Theta0
        self.theta = 0.0; self.P = c['p0']; self.xi = 1.0; self.last = 0.0
    def update(self, s_prev, u_prev, s_now):
        c = self.c; S = s_prev[None]; U = u_prev[None]
        th1 = np.array([[1.0, 0.0]]); th2 = np.array([[1.05, 0.0]])
        base = np_step(F_nom, S, U, th1)[0]
        if self.Theta0 is not None: base[DYN] += resid_np(np.hstack([S[:, DYN], U]), self.Theta0)[0]
        g = (np_step(F_nom, S, U, th2)[0] - np_step(F_nom, S, U, th1)[0]) / 0.05
        res = 0.0
        for ch, w in zip((4, 5), c['w']):
            a = g[ch] * w; y = (s_now[ch] - base[ch]) * w
            K = self.P * a / (c['lam'] + a * a * self.P)
            self.theta += K * (y - a * self.theta); self.P = min(c['p_max'], (self.P - K * a * self.P) / c['lam'])
            res += abs(y - a * self.theta)
        self.xi = float(np.clip(1.0 + self.theta, *c['bounds'])); self.theta = self.xi - 1.0
        self.last = res / 2.0
        return self.last

'''
p = p[:a] + new_est + p[b:]
p = p.replace('def rollout_np(S0, Useq, Theta, use_nn=True, member=None):', 'def rollout_np(S0, Useq, Theta, use_nn=True, member=None, xi=1.0):')
p = p.replace("    TH = np.tile([0.9, 1500.0], (B, 1))", "    TH = np.tile([xi, 0.0], (B, 1))")
p = p.replace("def multistep_scores(S_hist, U_hist, t, Theta, n=N, use_nn=True):", "def multistep_scores(S_hist, U_hist, t, Theta, n=N, use_nn=True, xi=1.0):")
p = p.replace("    R = rollout_np(S0, Useq, Theta, use_nn)\n", "    R = rollout_np(S0, Useq, Theta, use_nn, None, xi)\n")
p = p.replace("def ensemble_margins(s_now, U_plan, kappa_e=3.0, q_cap=3.0, n=N):", "def ensemble_margins(s_now, U_plan, kappa_e=3.0, q_cap=3.0, n=N, xi=1.0):")
p = p.replace("rollout_np(s_now[None], Useq[k:k + 1], None, True, member=k)[0]", "rollout_np(s_now[None], Useq[k:k + 1], None, True, member=k, xi=xi)[0]")
p = p.replace("# ### 7.1 RLS adaptation of the physics block of the residual's last layer (C1)", "# ### 7.1 Online friction-scale identification (C1)")
open('p2.py', 'w').write(p)
print("p1/p2 patched")
