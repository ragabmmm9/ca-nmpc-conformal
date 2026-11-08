import time, numpy as np, torch, sys
import data, models as M, uq_sim as S
from nom_torch import F_nom_t
torch.set_num_threads(1)
N = S.N
sets = data.load_all()
nm = M.Norm(sets['TRAIN'])
H = int(sys.argv[1]); ep_n = int(sys.argv[2]); bs = int(sys.argv[3]); lr = float(sys.argv[4])
tr = M.windows(sets['TRAIN'], H, stride=2); va = M.windows(sets['VAL'], H, stride=4)
net = M.Net(64)
t0 = time.time(); vl = M.train(net, nm, tr, va, epochs=ep_n, seed=0, bs=bs, lr=lr, H=H, verbose=True); print('train time', time.time()-t0, 'best val', vl, flush=True)
torch.save(net.state_dict(), f'/home/claude/work/paper_uq/results/t2_H{H}.pt')
def roll_err(net, eps):
    E=[]
    for e in eps:
        Sx, U = torch.tensor(e['S']), torch.tensor(e['U']); T=len(U)
        o = torch.arange(1, T-N+1); s = Sx[o]; errs=[]
        for j in range(N):
            if net is None: s = F_nom_t(s, U[o+j])
            else:
                with torch.no_grad(): s = M.step_learned(net, nm, s, U[o+j])
            errs.append(torch.linalg.norm(s[:, :2]-Sx[o+j+1, :2], dim=1))
        E.append(torch.stack(errs,1))
    return torch.cat(E).numpy()
for k in ('id','mu','ice','cs','mass'):
    for name, n_ in (('nominal', None), ('learned', net)):
        E = roll_err(n_, sets['TEST_'+k]); print(k, name, 'median e_j j=1,5,10,25', [round(float(np.median(E[:,j-1])),3) for j in (1,5,10,25)], 'q90 j=25', round(float(np.quantile(E[:,24],.9)),3), flush=True)
