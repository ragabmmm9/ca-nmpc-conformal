"""Re-generate the stored TEST episodes with the SAME random streams, capturing the true shift parameters (verified identical to the stored states)."""
import pickle, numpy as np
import uq_sim as S
OLD = './theory/theory_results_home_full/episodes_full.pkl'
def make_episode_p(rng, kind, T_steps, min_after=S.N + 25):
    for attempt in range(60):
        base = dict(mu=float(rng.uniform(0.8, 1.0)), m=float(rng.uniform(1450, 1550)))
        ts = int(rng.integers(30, 46)); sc2 = None
        if kind == "mu":     sc2 = dict(base, mu=float(rng.uniform(0.35, 0.8)))
        elif kind == "ice":  sc2 = dict(base, mu=float(rng.uniform(0.2, 0.3)))
        elif kind == "cs":   sc2 = dict(base, cs=float(rng.uniform(0.40, 0.55)))
        elif kind == "mass": sc2 = dict(base, m=float(base["m"] * rng.uniform(1.40, 1.60)))
        Sx, U, _ = S.gen_traj2(rng, base, sc2, ts, T_steps)
        if len(U) >= (ts + min_after if sc2 else S.N + 60):
            return dict(S=Sx, U=U, ts=(ts if sc2 else None), kind=kind, attempts=attempt + 1, base=base, sc2=sc2)
    return None
def regen(sets_ref):
    out = {}
    for i, k in enumerate(S.KINDS):
        rng = np.random.default_rng(S.SEED0 + 1000 * (20 + i)); eps = []
        while len(eps) < 80:
            e = make_episode_p(rng, k, 120)
            if e is not None: eps.append(e)
        ref = sets_ref[f'TEST_{k}']
        ok = all(np.allclose(a['S'], b['S']) and np.allclose(a['U'], b['U']) for a, b in zip(eps, ref))
        sev = []
        for e in eps:
            if e['sc2'] is None: sev.append(0.0)
            elif k in ('mu', 'ice'): sev.append(e['base']['mu'] - e['sc2']['mu'])
            elif k == 'cs': sev.append(1.0 - e['sc2']['cs'])
            else: sev.append(e['sc2']['m'] / e['base']['m'] - 1.0)
        out[k] = dict(severity=np.array(sev), identical=ok, base_mu=np.array([e['base']['mu'] for e in eps]))
        print(k, 'identical to stored:', ok, 'severity range', np.round([min(sev), max(sev)], 3))
    return out
if __name__ == '__main__':
    ref = pickle.load(open(OLD, 'rb')); res = regen(ref)
    pickle.dump(res, open('./paper_uq/results/severity.pkl', 'wb'))
