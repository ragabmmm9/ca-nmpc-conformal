import os, sys, time
os.environ['DEBUG'] = '1'; sys.argv = ['x', '0']
import numpy as np
import eval_uq as E
eps = E.sets['TEST_mu'][:6]
t0 = time.time(); r = E.run_set('dbg', eps); print('time', time.time() - t0)
for k, v in r['res'].items():
    if v: print(k, {f: (a.shape, float(np.nanmedian(a[:, 24])) if a.ndim == 2 else None) for f, a in v.items()})
