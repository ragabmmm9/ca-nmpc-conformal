"""Positive control: a COVARIATE shift (higher speed than any training episode, same physics)."""
import pickle, os, numpy as np
import uq_sim as S
OUT = './paper_uq/results/episodes_cov.pkl'
def gen_traj_cov(rng, sc1, T_steps=120, vrange=(20.0, 26.0), noise_scale=1.0):
    plant = S.make_plant(sc1); v_t = rng.uniform(*vrange); s = plant.reset(v_t); nsin = 3
    A = S.AMP * rng.uniform(0.0, 0.07, nsin) * (12.0 / v_t); fr = rng.uniform(0.1, 0.7, nsin); ph = rng.uniform(0, 2 * np.pi, nsin)
    step_t = rng.integers(10, 60); step_a = S.AMP * rng.uniform(-0.06, 0.06) * (12.0 / v_t) * (rng.random() < 0.6)
    Str = [s.copy()]; U = []
    for k in range(T_steps):
        t = k * S.DT
        dcmd = float((A * np.sin(2 * np.pi * fr * t + ph)).sum() + (step_a if k >= step_t else 0.0))
        if rng.random() < 0.15: dcmd *= rng.uniform(1.0, 1.8)
        acmd = float(np.clip(0.8 * (v_t - s[3]) + 0.8 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t), S.V.ax_min + 2, S.V.ax_max))
        dcmd = float(np.clip(dcmd, -S.V.d_max, S.V.d_max)); u = np.array([dcmd, acmd])
        s = plant.step(u, t, S.NOISE_PROC * rng.standard_normal(S.NX)); U.append(u); Str.append(s.copy())
        if abs(s[4]) > 3.5 or s[3] < 3.0 or abs(s[2]) > 1.0 or not np.all(np.isfinite(s)): break
    Str = np.array(Str); U = np.array(U); Sx = Str + noise_scale * S.NOISE_MEAS * rng.standard_normal(Str.shape)
    return Sx, U
def build(n=80):
    if os.path.exists(OUT): return pickle.load(open(OUT, 'rb'))
    rng = np.random.default_rng(S.SEED0 + 1000 * 41); eps = []; rej = 0
    while len(eps) < n:
        base = dict(mu=float(rng.uniform(0.8, 1.0)), m=float(rng.uniform(1450, 1550)))
        Sx, U = gen_traj_cov(rng, base)
        if len(U) >= S.N + 60: eps.append(dict(S=Sx, U=U, ts=None, kind='cov', attempts=1, v0=float(Sx[0, 3])))
        else: rej += 1
    print('covariate-shift episodes', len(eps), 'rejected', rej, 'mean length', np.mean([len(e['U']) for e in eps]), 'mean v0', np.mean([e['v0'] for e in eps]))
    pickle.dump(eps, open(OUT, 'wb')); return eps
if __name__ == '__main__':
    eps = build()
    allv = np.concatenate([e['S'][:, 3] for e in eps]); print('vx range in covariate set: %.1f-%.1f' % (allv.min(), allv.max()))
    import data
    tr = data.load_all()['TRAIN']; trv = np.concatenate([e['S'][:, 3] for e in tr]); print('vx range in training set: %.1f-%.1f' % (trv.min(), trv.max()), '| 99th pct', np.percentile(trv, 99))
