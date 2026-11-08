# POST-HOC sensitivity of the pilot30 conclusions to the success definition + power analysis (all from stored episodes; no new simulation)
import pandas as pd, numpy as np
from scipy.stats import binomtest, norm
d = pd.read_csv("/home/claude/work/pilotrun/pilot_mb_pilot30/episodes_main.csv"); S = d[d.scenario != "mo_cv"].copy()
S["safe"] = (1 - S.collision) * (1 - S.road_violation) * (1 - S.terminated)
DEF = {"as pre-specified: safe AND x>=100 m": S.safe * (S.progress >= 100 - 1e-9) if False else S.success,
       "safe AND progress >= 95 m": S.safe * (S.progress >= 95), "safe AND progress >= 90 m": S.safe * (S.progress >= 90),
       "safe AND progress >= 85 m": S.safe * (S.progress >= 85), "safe only (no finish requirement)": S.safe}
ORDER = ["Adaptive-MPC (FricID)", "Adaptive+TunedInflation", "Adaptive-MPC + CBF filter", "Adaptive-Track + CBF filter", "Adaptive+ObsStaticCP", "Adaptive+ObsACI", "CA-NMPC (ego tube)", "CA-NMPC-Obs (ours)"]
B4 = ORDER[:4]; PRIM, SEC = "CA-NMPC-Obs (ours)", "Adaptive+ObsACI"
def holm(p):
    p = np.array(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
def paired(x, ref, oth, col):
    p = x.pivot_table(index=["scenario", "seed"], columns="controller", values=col, observed=False); a, b = p[ref].values, p[oth].values
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return 100 * (a - b).mean(), n10, n01, (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)
rows = []; dec = []
for name, col in DEF.items():
    T = S.assign(v=col.values)
    pct = T.groupby("controller").v.mean() * 100
    r = {"definition": name, **{c: round(pct[c], 1) for c in ORDER}}; rows.append(r)
    for ref in (PRIM, SEC):
        res = [paired(T, ref, b, "v") for b in B4]; ph = holm([x[3] for x in res])
        ok = all((x[0] >= 10) and (p < 0.05) for x, p in zip(res, ph))
        dec.append({"definition": name, "ref": ref, **{f"diff vs {b.split(' (')[0]} [pp]": round(x[0], 1) for b, x in zip(B4, res)}, "p_holm_min": round(float(ph.min()), 3), "HA rule met": ok})
    res = paired(T, SEC, "Adaptive+ObsStaticCP", "v"); dec.append({"definition": name, "ref": "ACI minus StaticCP (HC)", "diff vs Adaptive+ObsStaticCP [pp]": round(res[0], 1), "p_holm_min": round(res[3], 3), "HA rule met": ""})
R = pd.DataFrame(rows); D = pd.DataFrame(dec)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("SUCCESS % (pooled shift, 150 episodes per controller) under alternative definitions"); print(R.to_string(index=False))
print(); print(D.to_string(index=False))
R.to_csv("/home/claude/work/pilotrun/sens/sensitivity_success_definition.csv", index=False); D.to_csv("/home/claude/work/pilotrun/sens/sensitivity_decision.csv", index=False)

# ---------------- power analysis (exact McNemar) -----------------
def power(n, pd_, delta, alpha=0.05, sims=1500, rng=np.random.default_rng(0)):
    p10, p01 = (pd_ + delta) / 2, (pd_ - delta) / 2; hits = 0
    for _ in range(sims):
        k10, k01, _r = rng.multinomial(n, [p10, p01, 1 - pd_]); m = k10 + k01
        if m and binomtest(k10, m, 0.5).pvalue < alpha: hits += 1
    return hits / sims
def need(pd_, delta, target=0.8):
    lo, hi = 20, 4000
    while lo < hi:
        mid = (lo + hi) // 2
        if power(mid, pd_, delta) >= target: hi = mid
        else: lo = mid + 1
    return lo
# discordance rates seen in the data (success, pooled shift)
Sx = S.assign(v=S.success)
pv = Sx.pivot_table(index=["scenario", "seed"], columns="controller", values="v", observed=False)
disc = {b.split(" (")[0]: float((pv[PRIM] != pv[b]).mean()) for b in ORDER if b != PRIM}
print("\nobserved discordance (share of paired episodes with different success), ours vs:", {k: round(v, 2) for k, v in disc.items()})
# design effect from seed clustering: variance of seed-level mean paired difference vs iid binomial variance
de = {}
for b in ORDER:
    if b == PRIM: continue
    diff = (pv[PRIM] - pv[b]); per_seed = diff.groupby(level="seed").mean(); k = diff.groupby(level="seed").size().iloc[0]
    de[b] = float(per_seed.var(ddof=1) * k / diff.var(ddof=1))
print("design effect (cluster by seed, 5 scenarios/seed):", {k.split(" (")[0]: round(v, 2) for k, v in de.items()})
DE = max(1.0, float(np.median(list(de.values()))))
rows = []
for pd_ in (0.2, 0.3):
    for delta in (0.05, 0.10):
        n = need(pd_, delta); rows.append({"discordance": pd_, "true diff [pp]": 100 * delta, "paired episodes (iid)": n, "design effect used": round(DE, 2), "paired episodes (clustered)": int(np.ceil(n * DE)), "seeds needed (5 shift scen./seed)": int(np.ceil(n * DE / 5))})
P = pd.DataFrame(rows); print("\nPOWER: exact McNemar, alpha=0.05 two-sided, 80% power"); print(P.to_string(index=False)); P.to_csv("/home/claude/work/pilotrun/sens/power_table.csv", index=False)
