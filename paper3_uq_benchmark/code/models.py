import numpy as np, torch, torch.nn as nn, time, math
import uq_sim as S
from nom_torch import F_nom_t
torch.set_default_dtype(torch.float64)
N = S.N; DYN = S.DYN; ND = len(DYN)
torch.set_num_threads(2)

# ---------------- data tensors ----------------
def windows(eps, H, stride=2):
    """training windows of H steps from measured states: states (n,H+1,8), inputs (n,H,2)"""
    A, B = [], []
    for e in eps:
        Sx, U = e['S'], e['U']; T = len(U)
        for k in range(0, T - H + 1, stride):
            A.append(Sx[k:k + H + 1]); B.append(U[k:k + H])
    return torch.tensor(np.array(A)), torch.tensor(np.array(B))

def feats(s, u):
    """network input z = (vx, vy, r, delta, ax, u_delta, u_ax) : (...,7)"""
    return torch.cat([s[..., 3:8], u], dim=-1)

class Norm:
    def __init__(self, eps):
        Sx = np.vstack([e['S'][:-1] for e in eps]); U = np.vstack([e['U'] for e in eps]); Sn = np.vstack([e['S'][1:] for e in eps])
        Z = np.hstack([Sx[:, 3:8], U]); self.mu = torch.tensor(Z.mean(0)); self.sd = torch.tensor(Z.std(0) + 1e-9)
        res = Sn - S.np_step(S.F_nom, Sx, U, np.tile([1.0, 0.0], (len(Sx), 1)))
        self.sy = torch.tensor(res[:, DYN].std(0) + 1e-12); self.my = torch.tensor(res[:, DYN].mean(0))
    def z(self, s, u): return (feats(s, u) - self.mu) / self.sd

# ---------------- network ----------------
class Net(nn.Module):
    """MLP 7 -> W -> W -> 5 (SiLU); optional dropout (mask shared along a rollout = a sampled 'sub-network'); optional log-variance head."""
    def __init__(self, W=64, p=0.0, hetero=False):
        super().__init__()
        self.W = W; self.p = p; self.hetero = hetero
        self.l1 = nn.Linear(7, W); self.l2 = nn.Linear(W, W); self.out = nn.Linear(W, ND * (2 if hetero else 1))
        self.act = nn.SiLU()
    def features(self, z, masks=None):
        h = self.act(self.l1(z))
        if masks is not None: h = h * masks[0]
        h = self.act(self.l2(h))
        if masks is not None: h = h * masks[1]
        return h
    def forward(self, z, masks=None):
        o = self.out(self.features(z, masks))
        if self.hetero: return o[..., :ND], o[..., ND:]
        return o
    def sample_masks(self, shape):
        if self.p <= 0: return None
        keep = 1 - self.p
        return [(torch.rand(*shape, self.W) < keep).to(torch.get_default_dtype()) / keep for _ in range(2)]

def step_learned(net, nm, s, u, masks=None, theta=None, xi=1.0, noise=False):
    """one predictive step  s' = F_nom(s,u) + my + sy * g(z) (+ sy*exp(0.5 logvar)*eps if hetero & noise)."""
    base = F_nom_t(s, u, xi)
    z = nm.z(s, u)
    if theta is not None:                      # last-layer Bayesian sample: theta (...,W+1,ND)
        h = net.features(z, masks); h1 = torch.cat([h, torch.ones_like(h[..., :1])], -1)
        g = torch.einsum('...w,...wd->...d', h1, theta)
    elif net.hetero:
        g, lv = net(z, masks)
        if noise: g = g + torch.exp(0.5 * lv) * torch.randn_like(g)
    else:
        g = net(z, masks)
    corr = torch.zeros_like(base); corr[..., DYN] = nm.my + nm.sy * g
    return base + corr

def unroll_loss(net, nm, Wst, Wu, w_pos=0.3, masks=None):
    """multi-step (teacher forcing only at the first state) loss on the DYN states (scaled) and on the position"""
    H = Wu.shape[1]; s = Wst[:, 0]; loss = 0.0
    for i in range(H):
        s = step_learned(net, nm, s, Wu[:, i], masks)
        tgt = Wst[:, i + 1]
        loss = loss + ((((s - tgt)[:, DYN]) / nm.sy) ** 2).mean() + w_pos * (((s - tgt)[:, :2] / 1.0) ** 2).mean() / (i + 1) ** 0.5
    return loss / H

def nll_loss(net, nm, Wst, Wu):
    """one-step Gaussian NLL of the scaled residual (heteroscedastic head)"""
    s, u, sn = Wst[:, :-1].reshape(-1, 8), Wu.reshape(-1, 2), Wst[:, 1:].reshape(-1, 8)
    base = F_nom_t(s, u); r = ((sn - base)[:, DYN] - nm.my) / nm.sy
    g, lv = net(nm.z(s, u))
    lv = lv.clamp(-8, 4)
    return (0.5 * (r - g) ** 2 / torch.exp(lv) + 0.5 * lv).mean()

def train(net, nm, tr, va, kind='mse', epochs=60, lr=3e-3, bs=256, seed=0, H=10, verbose=False):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Wst, Wu = tr; n = len(Wst); best = (1e18, None)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; a, b = Wst[idx], Wu[idx]
            if kind == 'mse':
                masks = net.sample_masks((len(idx),)) if net.p > 0 else None
                loss = unroll_loss(net, nm, a, b, masks=masks)
            else:
                loss = nll_loss(net, nm, a, b)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = (unroll_loss(net, nm, *va) if kind == 'mse' else nll_loss(net, nm, *va)).item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
        if verbose and (ep % 10 == 0 or ep == epochs - 1): print(f'   ep {ep:3d} val {vl:.4f}', flush=True)
    net.load_state_dict(best[1]); net.eval(); return best[0]
