"""Train all learned predictors for one training seed (resumable): 5 deterministic ensemble members (multi-step loss),
1 MC-dropout net, 5 heteroscedastic NLL members."""
import sys, os, time, torch, numpy as np
import data, models as M
torch.set_num_threads(1)
seed = int(sys.argv[1]); K = 5; H = 8; EP = 40
PATH = f'./paper_uq/results/models_seed{seed}.pt'
sets = data.load_all(); nm = M.Norm(sets['TRAIN'])
tr = M.windows(sets['TRAIN'], H, stride=2); va = M.windows(sets['VAL'], H, stride=4)
out = torch.load(PATH) if os.path.exists(PATH) else {'norm': dict(mu=nm.mu, sd=nm.sd, sy=nm.sy, my=nm.my), 'ens': [], 'ens_val': [], 'mcd': None, 'het': [], 'het_val': []}
sd = lambda n: {kk: v.clone() for kk, v in n.state_dict().items()}
t0 = time.time()
while len(out['ens']) < K:
    k = len(out['ens']); net = M.Net(64)
    vl = M.train(net, nm, tr, va, kind='mse', epochs=EP, seed=1000 * seed + k, bs=1024, lr=5e-3, H=H)
    out['ens'].append(sd(net)); out['ens_val'].append(vl); torch.save(out, PATH)
    print(f'[seed {seed}] ens member {k} val {vl:.4f}  t={time.time()-t0:.0f}s', flush=True)
if out['mcd'] is None:
    net = M.Net(64, p=0.1); vl = M.train(net, nm, tr, va, kind='mse', epochs=EP, seed=1000 * seed + 77, bs=1024, lr=5e-3, H=H)
    out['mcd'] = sd(net); out['mcd_val'] = vl; torch.save(out, PATH)
    print(f'[seed {seed}] mc-dropout val {vl:.4f}  t={time.time()-t0:.0f}s', flush=True)
while len(out['het']) < K:
    k = len(out['het']); net = M.Net(64, hetero=True)
    vl = M.train(net, nm, tr, va, kind='nll', epochs=EP, seed=1000 * seed + 500 + k, bs=2048, lr=3e-3, H=H)
    out['het'].append(sd(net)); out['het_val'].append(vl); torch.save(out, PATH)
    print(f'[seed {seed}] hetero member {k} val NLL {vl:.4f}  t={time.time()-t0:.0f}s', flush=True)
out['done'] = True; torch.save(out, PATH)
print('done', time.time() - t0)
