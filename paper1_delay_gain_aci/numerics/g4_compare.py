"""G4: delayed-integrator ACI (this paper, a_{t+1}=a_t+eta(alpha-err_{t-D})) versus the phase-decoupled
tau-DACI of El Halabi & Brandt (arXiv:2609.07251, Eq. 6: alpha_{s+tau}=alpha_s+gamma(alpha-err_s)), same streams.
Delay mapping: tau = D+1 (tau=1 is standard ACI, D=0).  i.i.d. Exp(1) scores, frozen oracle quantile map q(a)=-ln a,
same protocol as G3 (shift at t0, scale c).  Outputs CSV only; no hand-typed numbers."""
import numpy as np, csv, os, math
OUT = os.path.dirname(os.path.abspath(__file__))
alpha = 0.10
def kstar(D): return 2 * np.sin(np.pi / (2 * (2 * D + 1)))

def issue(a):
    with np.errstate(divide="ignore"):
        return np.where(a <= 0, np.inf, np.where(a >= 1, -np.inf, -np.log(np.clip(a, 1e-300, 1))))

def sim_integrator(eta, D, S):
    R, T = S.shape
    a = np.full(R, alpha); A = np.zeros((R, T)); E = np.zeros((R, T))
    for t in range(T):
        A[:, t] = a
        E[:, t] = (S[:, t] > issue(a)).astype(float)
        if t >= D: a = a + eta * (alpha - E[:, t - D])
    return A, E

def sim_phase(gamma, tau, S):
    """level for target s: A[s] = A[s-tau] + gamma*(alpha - err[s-tau]) (s>=tau), A[s]=alpha for s<tau."""
    R, T = S.shape
    A = np.zeros((R, T)); E = np.zeros((R, T))
    for t in range(T):
        A[:, t] = alpha if t < tau else A[:, t - tau] + gamma * (alpha - E[:, t - tau])
        E[:, t] = (S[:, t] > issue(A[:, t])).astype(float)
    return A, E

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    D = 25; tau = D + 1
    t0, Tt, R = 1000, 1800, 1000
    rows = []
    for eta in [0.002, 0.005, 0.01, 0.02, 0.05]:
        for scale in [1.0, 2.0, 3.0]:
            S = rng.exponential(1.0, (R, Tt)); S[:, t0:] *= scale
            kap = (1.0 / scale) * (alpha ** scale) ** (1.0 / scale - 1.0)
            for name, fn, kw in [("integrator", sim_integrator, dict(eta=eta, D=D)), ("phase", sim_phase, dict(gamma=eta, tau=tau))]:
                A, E = fn(S=S, **kw)
                w1 = slice(t0, t0 + 200); w2 = slice(t0 + 200, Tt)
                rows.append(dict(method=name, eta=eta, scale=scale, kappa=kap,
                                 gain=eta * kap, ratio_to_kstar=(eta * kap / kstar(D)) if name == "integrator" else (eta * kap / 2.0),
                                 pinf_pre=float((A[:, 200:t0] <= 0).mean()), pinf_after=float((A[:, w2] <= 0).mean()),
                                 pone_after=float((A[:, w2] >= 1).mean()),
                                 miss_first200=float(E[:, w1].mean()), miss_after=float(E[:, w2].mean()),
                                 sd_pre=float(A[:, 200:t0].std())))
    with open(os.path.join(OUT, "G4_compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
