"""Sensitivity: five additional deterministic ensemble members (K=10 in total) for one training seed."""
import sys, os, time, torch
import data, models as M
torch.set_num_threads(1)
seed = int(sys.argv[1]); H = 8; EP = 40
PATH = f'/home/claude/work/paper_uq/results/models_seed{seed}_extra.pt'
sets = data.load_all(); nm = M.Norm(sets['TRAIN'])
tr = M.windows(sets['TRAIN'], H, stride=2); va = M.windows(sets['VAL'], H, stride=4)
out = torch.load(PATH) if os.path.exists(PATH) else {'ens': [], 'ens_val': []}
t0 = time.time()
while len(out['ens']) < 5:
    k = 5 + len(out['ens']); net = M.Net(64)
    vl = M.train(net, nm, tr, va, kind='mse', epochs=EP, seed=1000 * seed + k, bs=1024, lr=5e-3, H=H)
    out['ens'].append({kk: v.clone() for kk, v in net.state_dict().items()}); out['ens_val'].append(vl); torch.save(out, PATH)
    print(f'[seed {seed}] extra ens member {k} val {vl:.4f} t={time.time()-t0:.0f}s', flush=True)
out['done'] = True; torch.save(out, PATH); print('done')
