import pandas as pd, numpy as np
from scipy.stats import binomtest
m = pd.read_csv("./pilotrun/matched/out/pilot_mb_pilot30/matched_episodes.csv"); mn = pd.read_csv("./pilotrun/pilot_mb_pilot30/episodes_main.csv")
keep = ["Adaptive-MPC (FricID)", "Adaptive+ObsStaticCP", "Adaptive+ObsACI", "CA-NMPC-Obs (ours)"]
d = pd.concat([mn[mn.controller.isin(keep)], m]); d = d[d.scenario != "mo_cv"]
d["safe"] = (1 - d.collision) * (1 - d.road_violation) * (1 - d.terminated)
ORD = ["Adaptive-MPC (FricID)", "Adaptive+ObsStaticCP", "Adaptive+ObsStaticCP x1.5", "Adaptive+ObsStaticCP x2.0", "Adaptive+ObsACI", "CA-NMPC-Obs (ours)"]
g = d.groupby("controller").agg(n=("success", "size"), success=("success", "mean"), safe_only=("safe", "mean"), finished=("finished", "mean"), collision=("collision", "mean"), road=("road_violation", "mean"), spin=("terminated", "mean"),
    speed=("mean_speed", "mean"), obs_coverage=("covo_overall", "mean"), qo_mid=("qo_mid", "mean"), qo_end=("qo_end", "mean")).reindex(ORD)
for c in ["success", "safe_only", "finished", "collision", "road", "spin"]: g[c] = 100 * g[c]
print(g.round(2).to_string()); g.round(3).to_csv("./pilotrun/matched/matched_summary.csv")
p = d.pivot_table(index=["scenario", "seed"], columns="controller", values="success", observed=False)
def mc(a, b):
    x, y = p[a].values, p[b].values; n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum()); return round(100 * (x - y).mean(), 1), n10, n01, round(binomtest(n10, n10 + n01, .5).pvalue, 4) if n10 + n01 else 1.0
for a, b in [("Adaptive+ObsStaticCP x1.5", "Adaptive+ObsStaticCP"), ("Adaptive+ObsStaticCP x2.0", "Adaptive+ObsStaticCP"), ("Adaptive+ObsACI", "Adaptive+ObsStaticCP x1.5"), ("Adaptive+ObsACI", "Adaptive+ObsStaticCP x2.0")]:
    print(f"{a} minus {b}: diff_pp, only-a, only-b, p =", mc(a, b))
print(d.pivot_table(index="controller", columns="scenario", values="success", aggfunc="mean", observed=False).reindex(ORD).mul(100).round(1).to_string())
print(d.pivot_table(index="controller", columns="scenario", values="covo_overall", aggfunc="mean", observed=False).reindex(ORD).round(2).to_string())
# correlation of coverage and success across the four margin variants (pooled)
print(g.loc[["Adaptive+ObsStaticCP", "Adaptive+ObsStaticCP x1.5", "Adaptive+ObsStaticCP x2.0", "Adaptive+ObsACI"], ["obs_coverage", "success", "collision", "finished"]].round(2).to_string())
