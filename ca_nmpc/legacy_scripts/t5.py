import numpy as np
rng = np.random.default_rng(5)
def gen_ice(n, mu=0.35):
    Z, Y, Tr = [], [], []
    for _ in range(n):
        S, U, Str = gen_traj(rng, mu, 1500.0, 80, noise_scale=1.0)
        if len(U) < 5: continue
        TH = np.tile([0.9, 1500.], (len(U), 1)); pred = np_step(F_nom, S[:-1], U, TH)
        Y.append(S[1:, DYN] - pred[:, DYN]); Z.append(np.hstack([S[:-1, DYN], U])); Tr.append((S, U))
    return np.vstack(Z), np.vstack(Y), Tr
Zi, Yi, Tr = gen_ice(40)
print("ice target std (normalised by y_sd):", (Yi / y_sd).std(0).round(2), " mean abs:", np.abs(Yi / y_sd).mean(0).round(2))
Phi = phi_np(Zi); lam = 1e-3
Th_batch = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ (Yi / y_sd)).T
print("1-step normalised RMSE on ice: nominal", round(rmse(Yi / y_sd), 3), "| offline NN", round(rmse(Yi / y_sd - Phi @ THETA0.T), 3),
      "| batch-fit(all feats)", round(rmse(Yi / y_sd - Phi @ Th_batch.T), 3))
idx = list(range(HID_S, FDIM)); Pp = Phi[:, idx]
Th_p = np.linalg.solve(Pp.T @ Pp + lam * np.eye(len(idx)), Pp.T @ (Yi / y_sd)).T
print("batch-fit physics block only:", round(rmse(Yi / y_sd - Pp @ Th_p.T), 3))
errs, errs0 = [], []
for (S, U) in Tr[:10]:
    r = RLS(THETA0); e_traj, e0 = [], []
    for k in range(1, len(U)):
        zp = np.concatenate([S[k - 1][DYN], U[k - 1]]); pred = np_step(F_nom, S[k - 1][None], U[k - 1][None], TH_B)[0]
        y = (S[k][DYN] - pred[DYN]) / y_sd; ph = phi_np(zp[None])[0]
        e_traj.append(np.abs(y - r.Theta @ ph).mean()); e0.append(np.abs(y).mean()); r.update(ph, y)
    errs.append(e_traj); errs0.append(e0)
L = min(len(e) for e in errs); E = np.array([e[:L] for e in errs]).mean(0); E0 = np.array([e[:L] for e in errs0]).mean(0)
print("online RLS mean|innov| by 10-step bins:", [round(float(E[i:i + 10].mean()), 2) for i in range(0, L, 10)])
print("nominal  mean|resid| by 10-step bins:  ", [round(float(E0[i:i + 10].mean()), 2) for i in range(0, L, 10)])
