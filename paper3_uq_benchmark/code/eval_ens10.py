"""Sensitivity: deep ensemble with K=10 members (seed 0): re-uses the evaluation machinery."""
import os, sys, pickle, numpy as np, torch
os.environ['DEBUG'] = '1'; sys.argv = ['x', '0']
import eval_uq as E
ex = torch.load(f'{E.RES}/models_seed0_extra.pt'); assert ex.get('done')
ens10 = E.ens + [E.mk(s) for s in ex['ens']]; assert len(ens10) == 10
out = {}
for name in ['CAL_ID', 'TEST_id', 'TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']:
    od = E.origins(E.sets[name]); n = len(od['o']); acc = []
    for a in range(0, n, 1200):
        b = min(a + 1200, n)
        with torch.no_grad(): acc.append(E.stats(torch.cat([E.roll_net(m, od['S0'][a:b], od['U'][a:b], 1) for m in ens10], 1), od['Y'][a:b]))
    out[name] = {f: np.concatenate([c[f] for c in acc]) for f in acc[0]}; print(name, n, flush=True)
pickle.dump(out, open(f'{E.RES}/eval_seed0_ens10.pkl', 'wb')); print('done')
