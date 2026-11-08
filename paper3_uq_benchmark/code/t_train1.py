import time, numpy as np, torch
import data, models as M, uq_sim as S
from nom_torch import F_nom_t
sets = data.load_all()
nm = M.Norm(sets['TRAIN'])
print('residual scale sy', nm.sy.numpy().round(4), 'my', nm.my.numpy().round(4))
tr = M.windows(sets['TRAIN'], 10, stride=2); va = M.windows(sets['VAL'], 10, stride=4)
print('train windows', tr[0].shape, 'val', va[0].shape)
net = M.Net(64)
t0 = time.time(); vl = M.train(net, nm, tr, va, epochs=40, seed=0, verbose=True); print('train time', time.time()-t0, 'best val', vl)
# evaluate 25-step position error on VAL origins: nominal vs learned
def roll_err(net, eps):
    E=[]
    for e in eps:
        Sx, U = torch.tensor(e['S']), torch.tensor(e['U']); T=len(U)
        o = torch.arange(1, T-N+1)
        s = Sx[o]; errs=[]
        for j in range(N):
            if net is None: s = F_nom_t(s, U[o+j])
            else:
                with torch.no_grad(): s = M.step_learned(net, nm, s, U[o+j])
            errs.append(torch.linalg.norm(s[:, :2]-Sx[o+j+1, :2], dim=1))
        E.append(torch.stack(errs,1))
    return torch.cat(E).numpy()
for name, n_ in (('nominal', None), ('learned', net)):
    E = roll_err(n_, sets['TEST_id'])
    print(name, 'TEST_id median e_j j=1,5,10,25:', [round(float(np.median(E[:,j-1])),3) for j in (1,5,10,25)], ' q90:', [round(float(np.quantile(E[:,j-1],.9)),3) for j in (1,5,10,25)])
for k in ('mu','ice','cs','mass'):
    for name, n_ in (('nominal', None), ('learned', net)):
        E = roll_err(n_, sets['TEST_'+k]); print(k, name, 'median e_25', round(float(np.median(E[:,24])),3), 'q90', round(float(np.quantile(E[:,24],.9)),3))
