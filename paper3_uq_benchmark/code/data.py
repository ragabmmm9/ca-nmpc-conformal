"""Datasets. TEST/CAL sets are the stored episodes of the earlier study (identical generator, verified); TRAIN and TRAINQ are NEW
in-distribution episodes (disjoint seeds) used to fit the learned predictors and the post-hoc uncertainty heads."""
import os, pickle, time
import numpy as np
import uq_sim as S
OLD = '/home/claude/work/theory/theory_results_home_full/episodes_full.pkl'
CACHE = '/home/claude/work/paper_uq/results/episodes_new.pkl'
N = S.N
def build_new(n_train=300, n_q=150, n_val=60, T_steps=120):
    if os.path.exists(CACHE): return pickle.load(open(CACHE, 'rb'))
    out = {}
    for name, n, sd in (('TRAIN', n_train, 31), ('TRAINQ', n_q, 32), ('VAL', n_val, 33)):
        rng = np.random.default_rng(S.SEED0 + 1000 * sd); eps = []; t0 = time.time()
        while len(eps) < n:
            e = S.make_episode(rng, 'id', T_steps)
            if e is not None: eps.append(e)
        out[name] = eps; print(name, len(eps), 'episodes, mean length', np.mean([len(e['U']) for e in eps]), f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(out, open(CACHE, 'wb')); return out
def load_all():
    old = pickle.load(open(OLD, 'rb')); new = build_new()
    sets = dict(old); sets.update(new); return sets
if __name__ == '__main__':
    sets = load_all()
    for k, v in sets.items(): print(k, len(v), np.mean([len(e['U']) for e in v]).round(1))
