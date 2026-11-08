"""Differentiable PyTorch port of the nominal single-track model F_nom (RK4, 2 sub-steps) used in the benchmark.
Verified against the CasADi implementation in uq_sim.py (see check_nom())."""
import numpy as np, torch
import uq_sim as S
torch.set_default_dtype(torch.float64)
V = S.V; L_WB = S.L_WB; DT = S.DT

def _smooth_vx(vx):
    return 0.5 * (vx + 1.0 + torch.sqrt((vx - 1.0) ** 2 + 0.1))

def f_nom_ct(s, u, xi=1.0):
    m, Iz = V.m_nom, V.Iz_nom
    psi, vx, vy, r, dl, ax = s[..., 2], s[..., 3], s[..., 4], s[..., 5], s[..., 6], s[..., 7]
    vxs = _smooth_vx(vx)
    Fzf = m * V.g * V.lr / L_WB; Fzr = m * V.g * V.lf / L_WB
    D0f, D0r = V.mu0_nom * Fzf, V.mu0_nom * Fzr
    af = dl - torch.atan((vy + V.lf * r) / vxs); ar = -torch.atan((vy - V.lr * r) / vxs)
    Fyf = xi * D0f * torch.tanh(V.Cf_nom * af / D0f); Fyr = xi * D0r * torch.tanh(V.Cr_nom * ar / D0r)
    axlim = xi * 0.95 * V.mu0_nom * V.g
    ax_eff = axlim * torch.tanh(ax / axlim)
    dvx = ax_eff + vy * r - V.cd_nom * vx ** 2 / m - V.crr_nom * V.g
    dvy = (Fyf * torch.cos(dl) + Fyr) / m - vx * r
    dr = (V.lf * Fyf * torch.cos(dl) - V.lr * Fyr) / Iz
    ddl = (u[..., 0] - dl) / V.tau_d_nom
    dax = (u[..., 1] - ax) / V.tau_a_nom
    dX = vx * torch.cos(psi) - vy * torch.sin(psi); dY = vx * torch.sin(psi) + vy * torch.cos(psi)
    return torch.stack([dX, dY, r, dvx, dvy, dr, ddl, dax], dim=-1)

def F_nom_t(s, u, xi=1.0, n_sub=2):
    h = DT / n_sub; x = s
    for _ in range(n_sub):
        k1 = f_nom_ct(x, u, xi); k2 = f_nom_ct(x + h / 2 * k1, u, xi)
        k3 = f_nom_ct(x + h / 2 * k2, u, xi); k4 = f_nom_ct(x + h * k3, u, xi)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

def check_nom(n=2000, seed=0):
    import pickle
    d = pickle.load(open('./theory/theory_results_home_full/episodes_full.pkl', 'rb'))
    Ss = np.vstack([e['S'][:-1] for e in d['TEST_mu']]); Us = np.vstack([e['U'] for e in d['TEST_mu']])
    idx = np.random.default_rng(seed).choice(len(Ss), n, replace=False)
    ref = S.np_step(S.F_nom, Ss[idx], Us[idx], np.tile([1.0, 0.0], (n, 1)))
    out = F_nom_t(torch.tensor(Ss[idx]), torch.tensor(Us[idx])).numpy()
    return float(np.abs(ref - out).max()), float(np.abs(ref).max())
if __name__ == '__main__':
    print('max abs diff torch vs casadi:', check_nom())
