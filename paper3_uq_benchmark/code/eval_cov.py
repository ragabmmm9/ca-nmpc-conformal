"""Positive control: evaluate the same estimators on a covariate shift (higher speed than any training episode)."""
import os, sys, pickle, numpy as np, torch
seed = int(sys.argv[1]); sys.argv = ['x', str(seed)]
import eval_uq as E
import cov_data
eps = cov_data.build()
rq = E.run_set('TRAINQ', E.sets['TRAINQ'], with_all=False); qnet, qmu, qsd = E.fit_qnn(rq)
r = E.run_set('TEST_cov', eps)
with torch.no_grad(): q = qnet((E.qin(torch.tensor(r['S0']).double(), torch.tensor(r['U']).double()) - qmu) / qsd).numpy().astype(np.float32)
r['res']['QNN'] = dict(err=r['res']['Det']['err'], s=q, q=q)
pickle.dump({'TEST_cov': r}, open(f'{E.RES}/eval_seed{seed}_cov.pkl', 'wb')); print('done')
