exec(open("./pilotrun/prefix_flat.py").read())
_base = dict(model="nom", adapt="rls", tube=None, gamma=1.0, obs="cv")
_st = dict(_base, obs_tube="static", obs_level="phys")
specs = {"Adaptive+ObsStaticCP x1.5": dict(_st, obs_scale=1.5), "Adaptive+ObsStaticCP x2.0": dict(_st, obs_scale=2.0)}
# --- reproducibility check against the main run (same code path, scale = 1): a few episodes must match episodes_main.csv
main = pd.read_csv("./pilotrun/pilot_mb_pilot30/episodes_main.csv")
rep_specs = {"Adaptive+ObsStaticCP": _st, "Adaptive-MPC (FricID)": _base}
rep = run_jobs2([(n, sp, sc, sd, None, "repro") for sd in (1000, 1001, 1002) for sc in ("mo_swerve", "mo_noisy") for n, sp in rep_specs.items()], ckpt=CKPT2)
cmp_ = rep.merge(main, on=["controller", "scenario", "seed"], suffixes=("_new", "_main"))
cmp_["d_prog"] = (cmp_.progress_new - cmp_.progress_main).abs(); cmp_["d_clear"] = (cmp_.min_clearance_new - cmp_.min_clearance_main).abs()
print("REPRO CHECK (new vs main run):", len(cmp_), "episodes | max |d progress| =", float(cmp_.d_prog.max()), "| max |d min_clearance| =", float(cmp_.d_clear.max()), "| success equal:", bool((cmp_.success_new == cmp_.success_main).all()))
cmp_[["controller", "scenario", "seed", "success_new", "success_main", "progress_new", "progress_main", "d_prog", "d_clear"]].to_csv(f"{PO}/repro_check.csv", index=False)
# --- the experiment
d = run_jobs2(make_jobs(specs, list(specs), _ALL_MO, PC["seeds"], "matched"), ckpt=CKPT2)
d.to_csv(f"{PO}/matched_episodes.csv", index=False)
S = d[d.scenario != "mo_cv"]
print(S.groupby("controller")[["success", "finished", "collision", "road_violation", "terminated", "mean_speed", "covo_overall", "qo_mid", "qo_end"]].mean().round(3).to_string())
print("MATCHED RUN DONE")
