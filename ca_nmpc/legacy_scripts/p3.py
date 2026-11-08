# %% [markdown]
# ## 8. Controllers (baselines, proposed method, ablations) and the closed-loop simulator
# All use the same NLP structure; they differ in *model*, *online adaptation* and *tube*:
#
# | name | model | RLS | tube | constraints |
# |---|---|---|---|---|
# | Nominal-MPC | nominal | – | – | hard |
# | Neural-MPC | nominal + NN residual (offline) | – | – | hard |
# | Neural-MPC+RLS | + online last-layer RLS | ✔ | – | hard |
# | NN+RLS+EnsTube | + RLS | ✔ | ensemble-spread tube (uncalibrated) | CBF-form |
# | NN+RLS+StaticCP | + RLS | ✔ | split-conformal tube, fixed after offline calibration | CBF-form |
# | **CA-NMPC (ours)** | + RLS | ✔ | **grouped ACI + shift propagation** | CBF-form |
# | ablation: w/o propagation | | ✔ | grouped ACI only | CBF-form |
# | ablation: w/o RLS | NN offline only | – | grouped ACI + propagation | CBF-form |
# | ablation: hard constraints | | ✔ | grouped ACI + propagation | hard (gamma=1) |
# | Oracle-MPC (reference only) | true physics + true mu | – | – | hard |

# %%
ALPHA, ETA, Q_CAP, GAMMA = 0.10, 0.05, 3.0, 0.5
SPECS = {
    "Nominal-MPC":              dict(model="nom",    adapt=False, tube=None,       gamma=1.0),
    "Neural-MPC":               dict(model="nn",     adapt=False, tube=None,       gamma=1.0),
    "Neural-MPC+RLS":           dict(model="nn",     adapt=True,  tube=None,       gamma=1.0),
    "NN+RLS+EnsTube":           dict(model="nn",     adapt=True,  tube="ens",      gamma=GAMMA),
    "NN+RLS+StaticCP":          dict(model="nn",     adapt=True,  tube="static",   gamma=GAMMA),
    "CA-NMPC (ours)":           dict(model="nn",     adapt=True,  tube="aci_prop", gamma=GAMMA),
    "abl: w/o propagation":     dict(model="nn",     adapt=True,  tube="aci",      gamma=GAMMA),
    "abl: w/o RLS":             dict(model="nn",     adapt=False, tube="aci_prop", gamma=GAMMA),
    "abl: hard constraints":    dict(model="nn",     adapt=True,  tube="aci_prop", gamma=1.0),
    "Oracle-MPC (ref.)":        dict(model="oracle", adapt=False, tube=None,       gamma=1.0),
}
TH_B = np.array([[0.9, 1500.0]])

def clearance_series(P, obs):
    """min clearance over time incl. linear interpolation between samples; P:(T,2)"""
    cl = np.full(len(P), np.inf)
    for t in range(len(P) - 1):
        for a in np.linspace(0, 1, 6):
            p = (1 - a) * P[t] + a * P[t + 1]
            c = np.min(np.linalg.norm(obs - p, axis=1) - R_O - R_V); cl[t] = min(cl[t], c)
    cl[-1] = np.min(np.linalg.norm(obs - P[-1], axis=1) - R_O - R_V)
    return cl

def simulate(name, spec, scen_name, seed, keep=False):
    sc = SCEN[scen_name]; obs, v0, mn, pn = make_world(seed)
    mpc = MPCS[spec["model"]]; mpc.reset()
    use_nn = spec["model"] == "nn"
    rls = RLS(THETA0) if use_nn else None; Theta = THETA0
    tube = None
    if spec["tube"] in ("static", "aci", "aci_prop"):
        tube = ConformalTube(CAL_SCORES, mode=spec["tube"], alpha=ALPHA, eta=ETA, q_cap=Q_CAP, seed=seed)
    need_scores = spec["tube"] is not None
    n_steps = int(round(CFG["T_sim"] / DT))
    s = np.array([0, 0, 0, v0, 0, 0, 0, 0.0]); u_prev = np.zeros(2)
    Sm, Uh, Str = [], [], [s.copy()]
    q_prev = np.zeros(N); q_log, cov_log, ok_log, t_log, slack_log, X_pred_log = [], [], [], [], [], []
    terminated = False
    for k in range(n_steps):
        sm = s + mn[k]; Sm.append(sm)
        if k >= 1:
            if use_nn and spec["adapt"]:
                zp = np.concatenate([Sm[k - 1][DYN], Uh[k - 1]])
                pred = np_step(F_nom, Sm[k - 1][None], Uh[k - 1][None], TH_B)[0]
                rls.update(phi_np(zn(zp[None]))[0], (sm[DYN] - pred[DYN]) / y_sd); Theta = rls.Theta
            if need_scores:
                e = multistep_scores(Sm, Uh, k, Theta if use_nn else None, use_nn=use_nn)
                flags = np.where(np.isnan(e), np.nan, (e <= q_prev + 1e-9).astype(float))
                cov_log.append(flags)
                if tube is not None: tube.update(e)
        # margins
        if spec["tube"] is None: q = np.zeros(N)
        elif spec["tube"] == "ens":
            Uplan = mpc.U0 if mpc.U0 is not None else np.zeros((NU, N))
            q = ensemble_margins(sm, Uplan)
        else: q = tube.margins()
        q_prev = q.copy()
        th = np.array([mu_at(sc, k * DT), sc["m"]])
        res = mpc.solve(sm, u_prev, obs, q, spec["gamma"], th=th, Theta=Theta)
        u = np.array([np.clip(res["u"][0], -V.d_max, V.d_max), np.clip(res["u"][1], V.ax_min, V.ax_max)])
        s = np.array(F_plant(s, u, th)).ravel() + pn[k]
        Uh.append(u); u_prev = u; Str.append(s.copy())
        q_log.append(q); ok_log.append(res["ok"]); t_log.append(res["time"]); slack_log.append(res["slack"])
        if keep: X_pred_log.append(res["X"][:2].copy())
        if abs(s[2]) > 1.2 or s[3] < 1.5 or abs(s[4]) > 6.0 or not np.all(np.isfinite(s)):
            terminated = True; break
    Str = np.array(Str); P = Str[:, :2]
    cl = clearance_series(P, obs)
    min_clear = float(cl.min())
    road_viol = float(max(0.0, np.max((Y_LO + R_V) - P[:, 1]), np.max(P[:, 1] - (Y_HI - R_V))))
    collision = min_clear < 0; road = road_viol > 0.05
    cov = np.array(cov_log) if cov_log else np.zeros((0, N))
    cov_g = [float(np.nanmean(cov[:, a - 1:b])) if cov.size and np.any(~np.isnan(cov[:, a - 1:b])) else np.nan for (a, b) in GROUPS]
    qlog = np.array(q_log)
    m = dict(controller=name, scenario=scen_name, seed=seed,
             success=float((not collision) and (not road) and (not terminated)), collision=float(collision), road_violation=float(road),
             terminated=float(terminated), min_clearance=min_clear, road_viol_m=road_viol,
             rms_y=float(np.sqrt(np.mean(P[:, 1] ** 2))), mean_speed=float(Str[:, 3].mean()), progress=float(P[-1, 0]),
             solve_ms=1e3 * float(np.mean(t_log)), solve_ms_p95=1e3 * float(np.quantile(t_log, 0.95)),
             solver_fail=float(1 - np.mean(ok_log)), slack_mean=float(np.nanmean(slack_log)),
             q_mid=float(qlog[:, 11].mean()), q_end=float(qlog[:, -1].mean()),
             cov_overall=float(np.nanmean(cov)) if cov.size and np.any(~np.isnan(cov)) else np.nan,
             cov_g1=cov_g[0], cov_g2=cov_g[1], cov_g3=cov_g[2])
    ex = dict(Str=Str, Uh=np.array(Uh), obs=obs, qlog=qlog, cov=cov, scen=scen_name) if keep else None
    return m, ex
print("simulator ready")
