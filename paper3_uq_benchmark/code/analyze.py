"""Tidy metrics from eval_seed*.pkl. Usage: python3 analyze.py seed1 seed2 ..."""
import sys, pickle, math, numpy as np, pandas as pd
from scipy import stats as st
RES = './paper_uq/results'; N = 25; ALPHA = 0.1
C90 = math.sqrt(-2 * math.log(ALPHA))                       # Rayleigh multiplier: 90% radius of an isotropic 2-D Gaussian with per-axis std s
TAUS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
METHODS = ['Nom', 'Det', 'Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN', 'Mon']
NATIVE = ['Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN']
SETS = ['TEST_id', 'TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']
FLOOR = 1e-4
import os
RHO = float(os.environ.get('RHO', 0.01))      # relative scale floor: s~_j = max(s_j, FLOOR, RHO * median_CAL(e_j)); position spread is identically 0 at j=1 by structure
SUF = os.environ.get('SUF', '')

def qlev(n, a=ALPHA): return min(1.0, math.ceil((n + 1) * (1 - a)) / n)
def calib(err, s):
    """split-conformal (per horizon) and max-score joint calibration on calibration scores; returns kappa (N,), lam"""
    r = err / np.maximum(s, FLOOR); n = len(r)
    kap = np.quantile(r, qlev(n), axis=0, method='higher')
    lam = np.quantile((r / kap).max(1), qlev(n), method='higher')
    return kap, lam
def windows(meta, kind):
    o, ts = meta['o'], meta['ts']
    if kind in ('TEST_id', 'TEST_cov'): return {'ID': np.ones(len(o), bool)}
    return {'POST': o >= ts, 'LATE': o >= ts + N, 'PRE': o + N < ts}
def cluster_ci(vals_by_ep, B=300, seed=0):
    r = np.random.default_rng(seed); v = np.array([x[0] for x in vals_by_ep]); w = np.array([x[1] for x in vals_by_ep], float); n = len(v)
    bs = [np.sum(v[i] * w[i]) / np.sum(w[i]) for i in (r.integers(0, n, n) for _ in range(B))]
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
def pinball(err, rad, tau=0.9):
    d = err - rad; return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))
def auroc(pos, neg):
    """P(score_pos > score_neg) via rank-sum"""
    x = np.concatenate([pos, neg]); rk = st.rankdata(x); n1, n0 = len(pos), len(neg)
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

SEV = pickle.load(open(RES + '/severity.pkl', 'rb'))
def analyse_seed(path, seed_label):
    D = pickle.load(open(path, 'rb')); rows, aux = [], []
    import os
    cp = path.replace('.pkl', '_cov.pkl')
    if os.path.exists(cp): D.update(pickle.load(open(cp, 'rb')))
    # origins with a non-finite prediction (diverged rollouts) are removed from ALL estimators for comparability; the rates are recorded
    for sname in [k for k in D if k.startswith(('CAL', 'TEST'))]:
        R = D[sname]; n = len(R['meta']['o']); keep = np.ones(n, bool)
        for m, r in R['res'].items():
            bad = ~np.all([np.isfinite(a).all(1) for a in r.values()], axis=0)
            aux.append(dict(seed=seed_label, method=m, set=sname, kind='diverge', frac=float(bad.mean()), n_bad=int(bad.sum()), n=n)); keep &= ~bad
        for m, r in R['res'].items():
            for f_ in list(r.keys()): r[f_] = r[f_][keep]
        for k_ in ('ep', 'o', 'ts'): R['meta'][k_] = R['meta'][k_][keep]
        aux.append(dict(seed=seed_label, set=sname, kind='dropped', frac=float(1 - keep.mean()), n=n))
    for m in METHODS:
        cal = D['CAL_ID']['res'][m]
        fl = np.maximum(FLOOR, RHO * np.median(cal['err'], 0)) if m not in ('Nom', 'Det') else np.zeros(N)
        S = lambda x, fl=fl: np.maximum(x, fl[None])
        kap, lam = calib(cal['err'], S(cal['s']) if m not in ('Nom', 'Det') else np.ones_like(cal['err']))
        cal_sh = D['CAL_SH']['res'][m]
        s_cal = np.concatenate([S(cal['s']), S(cal_sh['s'])]) if m not in ('Nom', 'Det') else np.ones((len(cal['err']) + len(cal_sh['err']), N), np.float32)
        kap_mix, lam_mix = calib(np.concatenate([cal['err'], cal_sh['err']]), s_cal)
        id_res = D['TEST_id']['res'][m]
        for sname in SETS + (['TEST_cov'] if 'TEST_cov' in D else []):
            R = D[sname]; r = R['res'][m]; meta = R['meta']; err = r['err']
            s = r['s'] if m not in ('Nom', 'Det') else np.ones_like(err)
            for wname, mask in windows(meta, sname).items():
                if mask.sum() < 50: continue
                e, ss, ep = err[mask], s[mask], meta['ep'][mask]
                layers = {}
                sf = S(ss)
                layers['cp'] = kap[None] * sf
                layers['cpj'] = lam * kap[None] * sf
                layers['cpj-mix'] = lam_mix * kap_mix[None] * sf
                if m in NATIVE: layers['native'] = (C90 * ss) if m != 'QNN' else r['q'][mask]
                for lname, rad in layers.items():
                    ok = e <= rad; joint_ep = []; 
                    for i in np.unique(ep):
                        sel = ep == i; joint_ep.append((ok[sel].all(1).mean(), sel.sum()))
                    ci = cluster_ci(joint_ep)
                    # shape quality independent of level: scalar c90 giving 90% joint coverage on this window (oracle, diagnostic)
                    shape = (kap[None] * sf) if lname != 'native' else (rad if m == 'QNN' else C90 * sf)
                    c90 = float(np.quantile((e / np.maximum(shape, FLOOR)).max(1), 0.9))
                    if lname in ('native', 'cp', 'cpj'):
                        aux.append(dict(seed=seed_label, method=m, layer=lname, set=sname, win=wname, kind='covj', covj=ok.mean(0).tolist(), rad_med=np.median(rad, 0).tolist()))
                    rows.append(dict(seed=seed_label, method=m, layer=lname, set=sname, win=wname, n=int(mask.sum()), n_eps=len(np.unique(ep)),
                                     cov=float(ok.mean()), cov25=float(ok[:, 24].mean()), cov_g1=float(ok[:, :4].mean()), cov_g2=float(ok[:, 4:10].mean()), cov_g3=float(ok[:, 10:].mean()),
                                     joint=float(ok.all(1).mean()), joint_lo=ci[0], joint_hi=ci[1],
                                     w10=float(np.median(rad[:, 9])), w25=float(np.median(rad[:, 24])), wavg=float(np.median(rad, 0).mean()), pin=pinball(e, rad),
                                     c90=c90, w25_oracle=float(c90 * np.median(shape[:, 24])), wavg_oracle=float(c90 * np.median(shape, 0).mean())))
                # reliability / ECE (per-step coverage across nominal levels)
                if m in ('Ens', 'MCD', 'Het', 'LLB', 'LLB-N'):
                    for lname in ('native', 'cp'):
                        cov = []
                        for tau in TAUS:
                            if lname == 'native': rad = math.sqrt(-2 * math.log(1 - tau)) * ss
                            else:
                                cr = cal['err'] / S(cal['s']); kt = np.quantile(cr, qlev(len(cr), 1 - tau), axis=0, method='higher'); rad = kt[None] * sf
                            cov.append(float((e <= rad).mean()))
                        aux.append(dict(seed=seed_label, method=m, layer=lname, set=sname, win=wname, kind='rel', ece=float(np.mean(np.abs(np.array(cov) - TAUS))), cov=cov))
                # dose-response: per-episode post-shift medians against the true shift severity
                if sname not in ('TEST_id', 'TEST_cov') and wname == 'POST' and m not in ('Nom', 'Det'):
                    sev = SEV[sname.replace('TEST_', '')]['severity']; eids = np.unique(ep)
                    ms = np.array([np.median(ss[ep == i][:, 24]) for i in eids]); me = np.array([np.median(e[ep == i][:, 24]) for i in eids]); sv = sev[eids]
                    aux.append(dict(seed=seed_label, method=m, set=sname, win=wname, kind='dose', rho_sev_s=float(st.spearmanr(sv, ms)[0]), rho_sev_e=float(st.spearmanr(sv, me)[0]), n_eps=len(eids)))
                    aux.append(dict(seed=seed_label, method=m, set=sname, win=wname, kind='doseep', sev=sv.tolist(), ms=ms.tolist(), me=me.tolist()))
                # proper scores, accuracy, rank quality, scale response, shift-detection AUROC
                acc = dict(seed=seed_label, method=m, set=sname, win=wname, kind='score', med_e10=float(np.median(e[:, 9])), med_e25=float(np.median(e[:, 24])),
                           q90_e25=float(np.quantile(e[:, 24], .9)), mean_e=float(e.mean()))
                if 'crps' in r:
                    acc.update(crps=float(r['crps'][mask].mean()), crps25=float(r['crps'][mask][:, 24].mean()), es=float(r['es'][mask].mean()), es25=float(r['es'][mask][:, 24].mean()))
                if m not in ('Nom', 'Det'):
                    acc.update(rho10=float(st.spearmanr(ss[:, 9], e[:, 9])[0]), rho25=float(st.spearmanr(ss[:, 24], e[:, 24])[0]),
                               med_s25=float(np.median(ss[:, 24])), med_s10=float(np.median(ss[:, 9])),
                               s_ratio25=float(np.median(ss[:, 24]) / np.median(id_res['s'][:, 24] if m != 'Mon' else id_res['s'][:, 24])),
                               e_ratio25=float(np.median(e[:, 24]) / np.median(id_res['err'][:, 24])))
                    if sname != 'TEST_id':
                        sid = (id_res['s'] if m != 'Mon' else id_res['s']).mean(1); acc.update(auroc_shift=auroc(ss.mean(1), sid))
                aux.append(acc)
    return pd.DataFrame(rows), pd.DataFrame(aux)
if __name__ == '__main__':
    M_, A_ = [], []
    for sd in sys.argv[1:]:
        m, a = analyse_seed(f'{RES}/eval_seed{sd}.pkl', int(sd)); M_.append(m); A_.append(a); print('seed', sd, 'analysed', flush=True)
    pd.concat(M_).to_csv(f'{RES}/metrics{SUF}.csv', index=False); pd.concat(A_).to_pickle(f'{RES}/aux{SUF}.pkl')
    print('saved')
