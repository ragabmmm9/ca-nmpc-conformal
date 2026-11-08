q = open('p3.py').read()
a = q.index("SPECS = {"); b = q.index("def clearance_series")
new_specs = '''SPECS = {
    "Nominal-MPC":              dict(model="nom",    adapt=False, tube=None,       gamma=1.0),
    "Adaptive-MPC (FricID)":    dict(model="nom",    adapt=True,  tube=None,       gamma=1.0),
    "Neural-MPC":               dict(model="nn",     adapt=False, tube=None,       gamma=1.0),
    "Neural-MPC+FricID":        dict(model="nn",     adapt=True,  tube=None,       gamma=1.0),
    "Nominal+ACI (tube only)":  dict(model="nom",    adapt=False, tube="aci_prop", gamma=GAMMA),
    "NN+FricID+EnsTube":        dict(model="nn",     adapt=True,  tube="ens",      gamma=GAMMA),
    "NN+FricID+StaticCP":       dict(model="nn",     adapt=True,  tube="static",   gamma=GAMMA),
    "CA-NMPC (ours)":           dict(model="nn",     adapt=True,  tube="aci_prop", gamma=GAMMA),
    "abl: w/o propagation":     dict(model="nn",     adapt=True,  tube="aci",      gamma=GAMMA),
    "abl: w/o FricID":          dict(model="nn",     adapt=False, tube="aci_prop", gamma=GAMMA),
    "abl: hard constraints":    dict(model="nn",     adapt=True,  tube="aci_prop", gamma=1.0),
    "Oracle-MPC (ref.)":        dict(model="oracle", adapt=False, tube=None,       gamma=1.0),
}

'''
q = q[:a] + new_specs + q[b:]
q = q.replace("    rls = RLS(THETA0) if use_nn else None; Theta = THETA0\n", "    est = XiEstimator(THETA0 if use_nn else None) if spec[\"adapt\"] else None; Theta = THETA0; xi = 1.0\n")
q = q.replace("innov_log, phys_log = [], [], [], [], [], [], [], [], []", "innov_log, xi_log = [], [], [], [], [], [], [], [], []")
q = q.replace("q_log, cov_log, ok_log, t_log, slack_log, slackm_log, X_pred_log, innov_log, phys_log = [], [], [], [], [], [], [], [], []", "q_log, cov_log, ok_log, t_log, slack_log, slackm_log, X_pred_log, innov_log, xi_log = [], [], [], [], [], [], [], [], []")
a = q.index("            if use_nn and spec[\"adapt\"]:"); b = q.index("            if need_scores:")
q = q[:a] + "            if est is not None:\n                innov_log.append(est.update(Sm[k - 1], Uh[k - 1], sm)); xi = est.xi\n" + q[b:]
q = q.replace("e = multistep_scores(Sm, Uh, k, Theta if use_nn else None, use_nn=use_nn)", "e = multistep_scores(Sm, Uh, k, Theta if use_nn else None, use_nn=use_nn, xi=xi)")
q = q.replace("q = ensemble_margins(sm, Uplan)", "q = ensemble_margins(sm, Uplan, xi=xi)")
q = q.replace("res = mpc.solve(sm, u_prev, obs_vis, q, spec[\"gamma\"], th=th, Theta=Theta)", "res = mpc.solve(sm, u_prev, obs_vis, q, spec[\"gamma\"], th=th, Theta=Theta, xi=xi); xi_log.append(xi)")
q = q.replace("innov=np.array(innov_log), phys=np.array(phys_log), slack=np.array(slack_log)) if keep else None", "innov=np.array(innov_log), xi=np.array(xi_log), xi_true=np.array([mu_at(sc, k * DT) / 0.9 for k in range(len(xi_log))]), slack=np.array(slack_log)) if keep else None")
q = q.replace("TH_B = np.array([[0.9, 1500.0]])\n", "")
q = q.replace("| Neural-MPC+RLS | + online last-layer RLS (neural + physics-feature block) | ✔ | – | hard |", "| Adaptive-MPC (FricID) | nominal + online friction-scale identification (classical adaptive MPC) | ✔ | – | hard |\n# | Neural-MPC+FricID | + frozen neural residual + online friction identification | ✔ | – | hard |")
q = q.replace("# | Nominal-MPC | nominal | – | – | hard |\n# | Neural-MPC | nominal + NN residual (offline) | – | – | hard |\n", "# | Nominal-MPC | nominal (assumes dry road) | – | – | hard |\n# | Neural-MPC | nominal + NN residual (offline) | – | – | hard |\n")
q = q.replace("NN+RLS+EnsTube", "NN+FricID+EnsTube").replace("NN+RLS+StaticCP", "NN+FricID+StaticCP").replace("abl: w/o RLS", "abl: w/o FricID")
open('p3.py', 'w').write(q)

p = open('p4.py').read()
p = p.replace("Neural-MPC+RLS", "Neural-MPC+FricID").replace("NN+RLS+EnsTube", "NN+FricID+EnsTube").replace("NN+RLS+StaticCP", "NN+FricID+StaticCP").replace("abl: w/o RLS", "abl: w/o FricID")
p = p.replace('"Nominal-MPC": "#999999", "Neural-MPC": "#CC79A7"', '"Nominal-MPC": "#999999", "Adaptive-MPC (FricID)": "#6B6B6B", "Neural-MPC": "#CC79A7"')
open('p4.py', 'w').write(p)
print("p3/p4 patched")
