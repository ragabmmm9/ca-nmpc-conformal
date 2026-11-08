"""Evaluate all uncertainty sources for one training seed. Output: results/eval_seed{s}.npz-like pickle with per-origin arrays."""
import sys, os, time, pickle, math
import numpy as np, torch, torch.nn as nn
import data, models as M, uq_sim as S
from nom_torch import F_nom_t
torch.set_num_threads(int(os.environ.get('NT', '1')))
N = S.N; DYN = M.DYN; ND = M.ND
seed = int(sys.argv[1]); RES = '/home/claude/work/paper_uq/results'
sets = data.load_all(); ck = torch.load(f'{RES}/models_seed{seed}.pt')
assert ck.get('done') or os.environ.get('DEBUG'), 'training not finished'
nm = M.Norm(sets['TRAIN']); 
for k in ('mu', 'sd', 'sy', 'my'): assert torch.allclose(getattr(nm, k), ck['norm'][k])
def mk(sd, **kw):
    n = M.Net(64, **kw); n.load_state_dict(sd); n.eval(); return n
ens = [mk(s) for s in ck['ens']]; mcd = mk(ck['mcd'], p=0.1); het = [mk(s, hetero=True) for s in ck['het']]
det = ens[0]
g = torch.Generator().manual_seed(9000 + seed); torch.manual_seed(9000 + seed)

# ------------------------------------------------------------------ origins
def origins(eps):
    S0, US, Y, E, O, TS = [], [], [], [], [], []
    for i, e in enumerate(eps):
        Sx, U = e['S'], e['U']; T = len(U); o = np.arange(1, T - N + 1)
        idx = o[:, None] + np.arange(N)[None, :]
        S0.append(Sx[o]); US.append(U[idx]); Y.append(Sx[idx + 1][:, :, :2]); E.append(np.full(len(o), i)); O.append(o)
        TS.append(np.full(len(o), -1 if e['ts'] is None else e['ts']))
    return dict(S0=torch.tensor(np.vstack(S0)), U=torch.tensor(np.vstack(US).reshape(-1, N, 2)), Y=torch.tensor(np.vstack(Y).reshape(-1, N, 2)),
                ep=np.concatenate(E), o=np.concatenate(O), ts=np.concatenate(TS))

# ------------------------------------------------------------------ last-layer Laplace posterior (fitted on TRAINQ, hyper-parameter on VAL)
def one_step(eps):
    Sx = np.vstack([e['S'][:-1] for e in eps]); U = np.vstack([e['U'] for e in eps]); Sn = np.vstack([e['S'][1:] for e in eps])
    return torch.tensor(Sx), torch.tensor(U), torch.tensor(Sn)
def llb_fit(net, eps_fit, eps_val):
    s, u, sn = one_step(eps_fit)
    with torch.no_grad():
        phi = net.features(nm.z(s, u)); Phi = torch.cat([phi, torch.ones_like(phi[:, :1])], 1)
        R = ((sn - F_nom_t(s, u))[:, DYN] - nm.my) / nm.sy
        Th = torch.cat([net.out.weight.T, net.out.bias[None]], 0)                     # (W+1, ND)  trained head = posterior mean
        sig2 = ((R - Phi @ Th) ** 2).mean(0)                                          # one-step noise variance per output (held-out data)
    sv, uv, snv = one_step(eps_val)
    with torch.no_grad():
        phv = net.features(nm.z(sv, uv)); Pv = torch.cat([phv, torch.ones_like(phv[:, :1])], 1)
        Rv = ((snv - F_nom_t(sv, uv))[:, DYN] - nm.my) / nm.sy; Mv = Pv @ Th
    best = None; G = Phi.T @ Phi
    for alpha in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4):
        L = []
        for d in range(ND):
            A = G / sig2[d] + alpha * torch.eye(G.shape[0]); Sig = torch.linalg.inv(A); L.append(Sig)
        Sigs = torch.stack(L)                                                          # (ND,W+1,W+1)
        var = sig2[None] + torch.einsum('nw,dwv,nv->nd', Pv, Sigs, Pv)
        nll = (0.5 * (Rv - Mv) ** 2 / var + 0.5 * torch.log(var)).mean().item()
        if best is None or nll < best[0]: best = (nll, alpha, Sigs)
    return Th, sig2, best[2], best[1], best[0]
Th, sig2, Sigs, alpha_llb, nll_llb = llb_fit(det, sets['TRAINQ'], sets['VAL'])
chol = torch.stack([torch.linalg.cholesky(Sigs[d] + 1e-12 * torch.eye(Sigs.shape[1])) for d in range(ND)])
P_LLB = 30
THS = Th[None] + torch.einsum('dwv,pvd->pwd', chol, torch.randn(P_LLB, Th.shape[0], ND, generator=g))   # (P,W+1,ND) fixed posterior samples
print(f'[seed {seed}] LL-Laplace prior precision {alpha_llb}, val NLL {nll_llb:.4f}', flush=True)

# ------------------------------------------------------------------ rollouts
def roll_nom(S0, U):
    s = S0; out = []
    for j in range(N): s = F_nom_t(s, U[:, j]); out.append(s[:, :2])
    return torch.stack(out, 1)[:, None]                                               # (n,1,N,2)
def roll_net(net, S0, U, P=1, masks=None, noise=False, theta=None, sig=None):
    n = S0.shape[0]; s = S0[:, None, :].expand(n, P, 8).clone(); out = []
    for j in range(N):
        u = U[:, j][:, None, :].expand(n, P, 2)
        if theta is not None:
            base = F_nom_t(s, u); z = nm.z(s, u); h = net.features(z); h1 = torch.cat([h, torch.ones_like(h[..., :1])], -1)
            gg = torch.einsum('npw,pwd->npd', h1, theta)
            if sig is not None: gg = gg + sig * torch.randn_like(gg)
            corr = torch.zeros_like(base); corr[..., DYN] = nm.my + nm.sy * gg; s = base + corr
        else:
            s = M.step_learned(net, nm, s, u, masks, noise=noise)
        out.append(s[..., :2])
    return torch.stack(out, 2)                                                        # (n,P,N,2)

def stats(Xs, Y):
    """Xs (n,P,N,2) samples, Y (n,N,2) truth -> dict(err, s, crps, es) using the sample mean as the point prediction"""
    P = Xs.shape[1]; mu = Xs.mean(1); err = torch.linalg.norm(Y - mu, dim=-1)
    d = dict(err=err.numpy().astype(np.float32))
    if P > 1:
        var = Xs.var(1, unbiased=True).sum(-1)                                        # tr(Sigma)
        d['s'] = torch.sqrt(var / 2).numpy().astype(np.float32)                       # per-axis std (isotropic equivalent)
        # fair CRPS (mean over the two coordinates) and fair energy score
        cr = 0.0
        for c in range(2):
            x = Xs[..., c]; y = Y[..., c]
            t1 = (x - y[:, None]).abs().mean(1); t2 = (x[:, :, None] - x[:, None, :]).abs().sum((1, 2)) / (2 * P * (P - 1))
            cr = cr + (t1 - t2) / 2
        d['crps'] = cr.numpy().astype(np.float32)
        t1 = torch.linalg.norm(Xs - Y[:, None], dim=-1).mean(1)
        dd = torch.linalg.norm(Xs[:, :, None] - Xs[:, None, :], dim=-1).sum((1, 2)) / (2 * P * (P - 1))
        d['es'] = (t1 - dd).numpy().astype(np.float32)
    return d

def monitor_scale(o):
    """RMS of the last K=4 one-step residuals of the Det model (delay 1), scaled by sy, per origin; computed per episode"""
    pass

def run_set(name, eps, with_all=True):
    t0 = time.time(); od = origins(eps); n = len(od['o']); res = {k: {} for k in ('Nom', 'Det', 'Ens', 'MCD', 'Het', 'LLB', 'LLB-N')}
    chunks = [(a, min(a + 1200, n)) for a in range(0, n, 1200)]
    acc = {k: [] for k in res}
    for a, b in chunks:
        S0, U, Y = od['S0'][a:b], od['U'][a:b], od['Y'][a:b]
        with torch.no_grad():
            acc['Nom'].append(stats(roll_nom(S0, U), Y))
            acc['Det'].append(stats(roll_net(det, S0, U, 1), Y))
            if with_all:
                acc['Ens'].append(stats(torch.cat([roll_net(m, S0, U, 1) for m in ens], 1), Y))
                P = 30; masks = mcd.sample_masks((b - a, P)); acc['MCD'].append(stats(roll_net(mcd, S0, U, P, masks=masks), Y))
                if het: acc['Het'].append(stats(torch.cat([roll_net(m, S0, U, 6, noise=True) for m in het], 1), Y))
                acc['LLB'].append(stats(roll_net(det, S0, U, P_LLB, theta=THS), Y))
                acc['LLB-N'].append(stats(roll_net(det, S0, U, P_LLB, theta=THS, sig=torch.sqrt(sig2)), Y))
    for k in res:
        if acc[k]:
            res[k] = {f: np.concatenate([c[f] for c in acc[k]]) for f in acc[k][0]}
    # Det one-step residual monitor (delay 1): z_k = (F^(s_{k-1},u_{k-1}) - s_k)[DYN]/sy ; eps_hat(o) = RMS of the last K=4 (available) residuals
    mon = np.zeros(n); K = 4
    for i, e in enumerate(eps):
        Sx, U = torch.tensor(e['S']), torch.tensor(e['U']); T = len(U)
        with torch.no_grad(): pred = M.step_learned(det, nm, Sx[:-1], U)
        z2 = (((pred - Sx[1:])[:, DYN]) / nm.sy).pow(2).mean(1).numpy()               # z2[k-1] for k=1..T
        cs = np.concatenate([[0.0], np.cumsum(z2)]); sel = od['ep'] == i
        oo = od['o'][sel]; lo = np.maximum(0, oo - K)
        mon[sel] = np.sqrt((cs[oo] - cs[lo]) / (oo - lo))
    res['Mon'] = dict(err=res['Det']['err'], s=np.repeat(mon[:, None], N, 1).astype(np.float32))
    print(f'  {name}: {n} origins, {time.time()-t0:.0f}s', flush=True)
    return dict(meta=dict(ep=od['ep'], o=od['o'], ts=od['ts']), S0=od['S0'].numpy().astype(np.float32), U=od['U'].numpy().astype(np.float32), res=res)

# ------------------------------------------------------------------ direct quantile network (fitted on TRAINQ errors of Det)
class QNet(nn.Module):
    def __init__(s, nin, W=128):
        super().__init__(); s.f = nn.Sequential(nn.Linear(nin, W), nn.SiLU(), nn.Linear(W, W), nn.SiLU(), nn.Linear(W, N))
    def forward(s, x): return nn.functional.softplus(s.f(x))
def qin(S0, U): return torch.cat([S0[:, DYN], U.reshape(len(U), -1)], 1)
def fit_qnn(rq, tau=0.9, epochs=200):
    X = qin(torch.tensor(rq['S0']).double(), torch.tensor(rq['U']).double()); E = torch.tensor(rq['res']['Det']['err']).double()
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xn = (X - mu) / sd
    eps_id = rq['meta']['ep']; ids = np.unique(eps_id); rng = np.random.default_rng(seed); va_ids = set(rng.choice(ids, len(ids) // 5, replace=False))
    va = torch.tensor([i in va_ids for i in eps_id]); tr = ~va
    net = QNet(X.shape[1]).double(); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4); best = (1e9, None)
    def pin(q, e): d = e - q; return torch.maximum(tau * d, (tau - 1) * d).mean()
    for ep in range(epochs):
        perm = torch.randperm(int(tr.sum()), generator=g); Xt, Et = Xn[tr][perm], E[tr][perm]
        for i in range(0, len(Xt), 512):
            loss = pin(net(Xt[i:i + 512]), Et[i:i + 512]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): vl = pin(net(Xn[va]), E[va]).item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval(); print(f'[seed {seed}] QNN val pinball {best[0]:.4f}', flush=True)
    return net, mu, sd

if __name__ == '__main__':
    t0 = time.time(); out = {}
    rq = run_set('TRAINQ', sets['TRAINQ'], with_all=False)
    qnet, qmu, qsd = fit_qnn(rq)
    for name in ['CAL_ID', 'CAL_SH', 'TEST_id', 'TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']:
        r = run_set(name, sets[name])
        with torch.no_grad(): q = qnet((qin(torch.tensor(r['S0']).double(), torch.tensor(r['U']).double()) - qmu) / qsd).numpy().astype(np.float32)
        r['res']['QNN'] = dict(err=r['res']['Det']['err'], s=q, q=q)
        out[name] = r
    out['info'] = dict(seed=seed, alpha_llb=alpha_llb, nll_llb=nll_llb, ens_val=ck['ens_val'], mcd_val=ck['mcd_val'], het_val=ck['het_val'])
    pickle.dump(out, open(f'{RES}/eval_seed{seed}.pkl', 'wb')); print('done', time.time() - t0)
