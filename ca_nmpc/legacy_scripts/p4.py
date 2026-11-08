# %% [markdown]
# ## 9. Method summary and the safety statement
#
# **Model.** `x_{k+1} = f_nom(x_k, u_k) + B [ Theta_nn Phi_nn(z_k) + Theta_phys Phi_phys(z_k) + b ] ⊙ sigma_y`, `z_k = (v_x, v_y, r, delta, a_x, u_k)`.
# `Theta_nn` is trained offline (ensemble → distilled student) and frozen; `(Theta_phys, b)` start at zero and are updated by RLS with forgetting + leakage.
#
# **Multi-step score.** With `x̂_{t|t-j}` the model rolled from the *measured* state at `t-j` along the *executed* inputs,
# `e_{t,j} = || p̂_{t|t-j} - p_t ||_2`  (position error of the j-step prediction).
#
# **Normalised grouped ACI.** `g_j` = in-distribution median of `e_{·,j}`; `s_{t,j} = e_{t,j}/g_j`. For each horizon group `G`
# (`[1-4]`, `[5-10]`, `[11-25]`): `kappa_G(t) = Quantile_{1-alpha_G(t)}(window of s)`, `alpha_G(t+1) = alpha_G(t) + eta (alpha − err_G(t))`, with `err_G(t)` the fraction of the newly observed scores of the group exceeding the margin in force.
# **Shift propagation:** `kappa_G ← max(kappa_G, kappa_G^cal · max(1, kappa_{G-1}/kappa_{G-1}^cal))`.
# **Tube:** `q_j = min(q_cap, kappa_{G(j)} · g_j)`.
#
# **Proposition (tightened constraints ⇒ true constraints, on the coverage event).** Let `h` be 1-Lipschitz in the position (true for distance-to-obstacle and distance-to-edge). Let `x̂_1..x̂_N` be the MPC prediction for the input sequence `u_0..u_{N-1}`
# and assume `|| p̂_j − p_j^{true}|| ≤ q_j` for `j=1..N` (the *coverage event*) where `p_j^{true}` is the true position under the same inputs.
# If the NLP solution has zero margin-slack, `h(x̂_j) − q_j ≥ 0`, then `h(x_j^{true}) ≥ h(x̂_j) − q_j ≥ 0`.  ∎
# *(Proof: 1-Lipschitz + triangle inequality.)*
#
# **What ACI buys.** For any shift, the group-wise long-run miscoverage obeys `| (1/T) Σ_t err_t − alpha | ≤ (max(alpha_1, 1−alpha_1) + eta)/(eta T)` (Gibbs & Candès, 2021) for the *un-clipped* recursion; our clipping, windowed quantiles and delayed feedback make this an **approximation that we verify empirically** (Section 11, coverage figure) rather than a theorem.
#
# **Honest limitations.** (i) The guarantee is for the open-loop predicted tube; closed-loop recursive feasibility would need a terminal safe set. (ii) Scores are model-side, so they certify *prediction* error, not unmodelled obstacles/perception errors. (iii) Latent shifts (grip that is not excited yet) are invisible to any data-driven calibrator until the manoeuvre excites them.

# %% [markdown]
# ## 10. Run the benchmark
# Evaluation seeds are **disjoint from the development seeds** (scenario design and hyper-parameters were tuned on seeds 0-2 and 100-102; reported seeds start at 1000).
# Episodes run in parallel (fork) when more than one CPU is available.

# %%
import multiprocessing as mp
N_JOBS = int(os.environ.get("CA_JOBS", min(2, os.cpu_count() or 1)))

def _job(a):
    name, sc, seed = a
    m, _ = simulate(name, SPECS[name], sc, seed)
    return m

def run_grid(names, scenarios, seeds, n_jobs=N_JOBS):
    jobs = [(n, s, sd) for sd in seeds for s in scenarios for n in names]
    t0 = time.time(); rows = []
    if n_jobs > 1:
        torch.set_num_threads(1)
        with mp.get_context("fork").Pool(n_jobs) as pool:
            for i, m in enumerate(pool.imap_unordered(_job, jobs, chunksize=1), 1):
                rows.append(m)
                if i % 10 == 0 or i == len(jobs): print(f"  {i}/{len(jobs)} episodes | {time.time()-t0:.0f}s", flush=True)
    else:
        for i, a in enumerate(jobs, 1):
            rows.append(_job(a))
            if i % 5 == 0 or i == len(jobs): print(f"  {i}/{len(jobs)} episodes | {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)

t0 = time.time()
df = run_grid(list(SPECS), CFG["scenarios"], CFG["seeds"])
df["controller"] = pd.Categorical(df["controller"], categories=list(SPECS), ordered=True)
df = df.sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
df.drop(columns=["cov_j", "q_j"]).to_csv(f"{OUT}/episodes.csv", index=False)
print(f"done: {len(df)} episodes in {time.time()-t0:.0f}s")

# %% [markdown]
# ## 11. Results
# ### 11.1 Main table — pooled over the shift scenarios (everything except `nominal`)
# Success = no collision, no road departure (> 5 cm), no spin-out. Wilson 95 % intervals. `Oracle-MPC` knows the true friction and is shown only as a reference.

# %%
SHIFT = [s for s in CFG["scenarios"] if s != "nominal"]
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))
def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    r = np.random.default_rng(seed); bs = r.choice(x, (B, len(x))).mean(1); return tuple(np.quantile(bs, [0.025, 0.975]))
df["violation_m"] = np.maximum(0.0, -df["min_clearance"]) + df["road_viol_m"] + 10.0 * df["terminated"]   # one scalar "how bad was the safety failure"

def pooled_table(d):
    rows = []
    for c in SPECS:
        x = d[d.controller == c]
        if x.empty: continue
        k, n = int(x.success.sum()), len(x); lo, hi = wilson(k, n)
        rows.append(dict(Controller=c, n=n, **{"Success %": 100 * k / n, "CI lo": 100 * lo, "CI hi": 100 * hi,
                    "Collision %": 100 * x.collision.mean(), "Road-dep. %": 100 * x.road_violation.mean(), "Spin-out %": 100 * x.terminated.mean(),
                    "Violation m": x.violation_m.mean(), "Speed m/s": x.mean_speed.mean(), "Path RMS m": x.rms_y.mean(), "Solve ms": x.solve_ms.mean()}))
    return pd.DataFrame(rows).set_index("Controller")
T_main = pooled_table(df[df.scenario.isin(SHIFT)])
T_all = {sc: pooled_table(df[df.scenario == sc]) for sc in CFG["scenarios"]}
display_cols = ["n", "Success %", "CI lo", "CI hi", "Collision %", "Road-dep. %", "Spin-out %", "Violation m", "Speed m/s", "Path RMS m", "Solve ms"]
print("Pooled over:", SHIFT); print(T_main[display_cols].round(2).to_string())
T_main.round(3).to_csv(f"{OUT}/table_main_pooled.csv")
open(f"{OUT}/table_main_pooled.tex", "w").write(T_main[display_cols].round(2).to_latex(caption="Pooled results over the distribution-shift scenarios.", label="tab:main"))

# %% [markdown]
# ### 11.2 Per-scenario success heat-map and paired statistics

# %%
PAL = {  # colour follows the controller (Okabe-Ito based), fixed across all figures
    "Nominal-MPC": "#999999", "Neural-MPC": "#CC79A7", "Neural-MPC+RLS": "#E69F00", "Nominal+ACI (tube only)": "#56B4E9",
    "NN+RLS+EnsTube": "#009E73", "NN+RLS+StaticCP": "#0072B2", "CA-NMPC (ours)": "#D55E00",
    "abl: w/o propagation": "#F0A07A", "abl: w/o RLS": "#B7855A", "abl: hard constraints": "#8C564B", "Oracle-MPC (ref.)": "#222222"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "font.size": 9, "axes.titlesize": 10, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

def fig_heatmap():
    M = np.array([[100 * df[(df.controller == c) & (df.scenario == s)].success.mean() for s in CFG["scenarios"]] for c in SPECS])
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(CFG["scenarios"]), 0.45 * len(SPECS) + 1.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(CFG["scenarios"]))); ax.set_xticklabels(CFG["scenarios"]); ax.set_yticks(range(len(SPECS))); ax.set_yticklabels(list(SPECS))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="white" if M[i, j] > 55 else "#222")
    ax.set_title("Success rate [%] (collision-free, on-road, no spin-out)"); fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(f"{OUT}/fig_success_heatmap.png"); plt.show()
fig_heatmap()

from scipy import stats
def holm(pvals):
    p = np.array(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def paired_tests(ref="CA-NMPC (ours)"):
    d = df[df.scenario.isin(SHIFT)].pivot_table(index=["scenario", "seed"], columns="controller", values=["success", "violation_m"], observed=False)
    rows = []
    for c in SPECS:
        if c == ref: continue
        a, b = d["success"][ref].values, d["success"][c].values
        n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
        p_mc = stats.binomtest(n01, n01 + n10, 0.5, alternative="greater").pvalue if (n01 + n10) > 0 else 1.0
        va, vb = d["violation_m"][ref].values, d["violation_m"][c].values
        try: p_w = stats.wilcoxon(vb - va, alternative="greater").pvalue if np.any(vb != va) else 1.0
        except ValueError: p_w = 1.0
        rows.append(dict(vs=c, ours_only_success=n01, other_only_success=n10, p_McNemar=p_mc, p_Wilcoxon_violation=p_w, mean_violation_diff=float((vb - va).mean())))
    t = pd.DataFrame(rows).set_index("vs")
    t["p_McNemar_holm"] = holm(t.p_McNemar.values); t["p_Wilcoxon_holm"] = holm(t.p_Wilcoxon_violation.values)
    return t
T_stats = paired_tests()
print("Paired one-sided tests: H1 = CA-NMPC (ours) is better (pairs = scenario x seed; Holm-corrected)")
print(T_stats.round(4).to_string()); T_stats.round(5).to_csv(f"{OUT}/table_paired_tests.csv")
print(f"\nNOTE: n = {len(df[(df.controller=='CA-NMPC (ours)') & df.scenario.isin(SHIFT)])} paired episodes. With MODE='fast' the test is under-powered; use MODE='full' for the numbers you report.")

# %% [markdown]
# ### 11.3 Is the tube calibrated? (target coverage = 1 − α = 90 %)
# Empirical coverage of the issued margins, averaged over time, per horizon group and scenario. Under-coverage means the margin was too small; large over-coverage means conservatism.

# %%
TUBE_CTRL = [c for c in SPECS if SPECS[c]["tube"] is not None]
cov_tab = df[df.controller.isin(TUBE_CTRL)].groupby(["controller", "scenario"], observed=True)[["cov_g1", "cov_g2", "cov_g3", "cov_overall", "q_mid", "q_end"]].mean().round(3)
print(cov_tab.to_string()); cov_tab.to_csv(f"{OUT}/table_coverage.csv")

def fig_cov_by_horizon():
    sc_list = SHIFT if SHIFT else CFG["scenarios"]
    fig, axs = plt.subplots(1, len(sc_list), figsize=(3.6 * len(sc_list), 3.1), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, sc in zip(axs, sc_list):
        for c in ["NN+RLS+EnsTube", "NN+RLS+StaticCP", "abl: w/o propagation", "CA-NMPC (ours)"]:
            x = df[(df.controller == c) & (df.scenario == sc)]
            if x.empty: continue
            A = np.vstack(x.cov_j.values); ax.plot(np.arange(1, N + 1), np.nanmean(A, 0), color=PAL[c], lw=2.0 if "ours" in c else 1.3, label=c)
        ax.axhline(1 - ALPHA, color="k", ls="--", lw=0.9); ax.set_title(sc); ax.set_xlabel("horizon step j"); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("empirical coverage"); axs[-1].legend(fontsize=7, loc="lower right"); fig.suptitle("Coverage of the issued margins vs horizon (dashed = target 90 %)", y=1.02)
    fig.savefig(f"{OUT}/fig_coverage_by_horizon.png"); plt.show()
fig_cov_by_horizon()

# %% [markdown]
# ### 11.4 Safety–efficiency trade-off and what the tube costs

# %%
def fig_tradeoff():
    d = df[df.scenario.isin(SHIFT)]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for c in SPECS:
        x = d[d.controller == c]
        if x.empty: continue
        sx = x.mean_speed.mean(); lo, hi = wilson(int(x.success.sum()), len(x))
        axs[0].errorbar(sx, 100 * x.success.mean(), yerr=[[100 * (x.success.mean() - lo)], [100 * (hi - x.success.mean())]], fmt="o", color=PAL[c], ms=6, capsize=2, lw=1)
        axs[0].annotate(c.replace("abl: ", "ab: "), (sx, 100 * x.success.mean()), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        axs[1].scatter(x.q_end.mean(), 100 * x.success.mean(), color=PAL[c], s=36)
    axs[0].set_xlabel("mean speed [m/s]  (efficiency →)"); axs[0].set_ylabel("success [%]"); axs[0].set_title("Safety vs efficiency (shift scenarios)")
    axs[1].set_xlabel("mean tube width at j=25 [m]"); axs[1].set_ylabel("success [%]"); axs[1].set_title("Tube width vs success")
    fig.savefig(f"{OUT}/fig_tradeoff.png"); plt.show()
fig_tradeoff()

# %% [markdown]
# ### 11.5 Closed-loop traces: paths and how the tube reacts to the shift

# %%
def fig_traces(scn, seed, names=("Nominal-MPC", "Neural-MPC+RLS", "NN+RLS+StaticCP", "CA-NMPC (ours)", "Oracle-MPC (ref.)")):
    runs = {n: simulate(n, SPECS[n], scn, seed, keep=True) for n in names}
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.2), gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    obs = next(iter(runs.values()))[1]["obs"]; xs = np.linspace(0, 110, 400)
    ax = axs[0]; ax.axhspan(Y_HI, Y_HI + 1.5, color="#bbb", alpha=0.5); ax.axhspan(Y_LO - 1.5, Y_LO, color="#bbb", alpha=0.5)
    ax.plot(xs, path_y(xs), color="#888", lw=0.8, ls=":", label="lane centre")
    for o in obs: ax.add_patch(plt.Circle(o, R_O, color="#444")); ax.add_patch(plt.Circle(o, R_O + R_V, fill=False, ec="#444", ls="--", lw=0.7))
    for n, (m, ex) in runs.items():
        ax.plot(ex["Str"][:, 0], ex["Str"][:, 1], color=PAL[n], lw=2.0 if "ours" in n else 1.3, label=f"{n} ({'ok' if m['success'] else 'FAIL'})")
    ax.set_xlim(0, 110); ax.set_ylim(Y_LO - 1.5, Y_HI + 1.5); ax.set_aspect("equal"); ax.set_title(f"scenario={scn}, seed={seed}  (dashed ring = obstacle + vehicle radius)"); ax.legend(fontsize=7, ncol=3, loc="upper right")
    tt = np.arange(CFG["T_sim"] / DT) * DT
    for n in ("NN+RLS+StaticCP", "CA-NMPC (ours)"):
        ex = runs[n][1]; axs[1].plot(tt[:len(ex["qlog"])], ex["qlog"][:, -1], color=PAL[n], lw=1.8, label=f"{n}: q at j=25")
        axs[1].plot(tt[:len(ex["qlog"])], ex["qlog"][:, 9], color=PAL[n], lw=1.0, ls="--", label=f"{n}: q at j=10")
    sc_ = SCEN[scn]
    if "t_switch" in sc_:
        for a in axs[1:]: a.axvline(sc_["t_switch"], color="r", lw=0.8, ls=":")
    axs[1].set_ylabel("margin [m]"); axs[1].legend(fontsize=6.5, ncol=2); axs[1].set_title("conformal margins (red dotted = friction switch)")
    ex = runs["CA-NMPC (ours)"][1]
    axs[2].plot(tt[:len(ex["innov"])], ex["innov"], color=PAL["CA-NMPC (ours)"], label="RLS innovation (mean |.|, normalised)")
    axs[2].plot(tt[:len(ex["phys"])], ex["phys"] / max(1e-9, ex["phys"].max()), color="#555", lw=1, label="||Theta_phys|| (scaled)")
    axs[2].set_xlabel("time [s]"); axs[2].legend(fontsize=7); axs[2].set_title("online adaptation (ours)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_traces_{scn}.png"); plt.show()
for scn in SHIFT[:2]: fig_traces(scn, CFG["seeds"][0])

# %% [markdown]
# ### 11.6 Ablation reading guide
# * `Neural-MPC` → `Neural-MPC+RLS`: value of **online adaptation** (C1).
# * `NN+RLS+StaticCP` → `CA-NMPC (ours)`: value of the **online (ACI) calibration** under shift (C2).
# * `abl: w/o propagation` → `CA-NMPC (ours)`: value of **shift propagation** (C3).
# * `abl: hard constraints` → `CA-NMPC (ours)`: value of the **CBF-form margin** (C4).
# * `abl: w/o RLS` → `CA-NMPC (ours)`: does the margin alone suffice without adaptation?
# * `NN+RLS+EnsTube` → `CA-NMPC (ours)`: **calibrated vs uncalibrated** uncertainty.
# * `Nominal+ACI (tube only)` → `CA-NMPC (ours)`: does a **learned model shrink the tube** (cheaper safety)?

# %%
def auto_summary():
    P = T_main
    msg = []
    if "CA-NMPC (ours)" in P.index:
        o = P.loc["CA-NMPC (ours)"]; msg.append(f"CA-NMPC success {o['Success %']:.0f}% [{o['CI lo']:.0f}, {o['CI hi']:.0f}] over {int(o['n'])} shift episodes at {o['Speed m/s']:.1f} m/s.")
        for c in ["Nominal-MPC", "Neural-MPC+RLS", "NN+RLS+StaticCP", "NN+RLS+EnsTube", "Oracle-MPC (ref.)"]:
            if c in P.index: msg.append(f"  vs {c}: {P.loc[c, 'Success %']:.0f}% success, {P.loc[c, 'Speed m/s']:.1f} m/s")
    return "\n".join(msg)
print(auto_summary())
