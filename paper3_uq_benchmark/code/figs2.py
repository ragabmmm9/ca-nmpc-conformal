"""Explanatory and summary figures (design schematic, shift schematic, case-study episode, reaction map, conformal gap, trade-off, scorecard)."""
import pickle, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from make_outputs import *
import analyze as AN
SEEDC = '#52514e'; INK = '#0b0b0b'; RED = '#c8372d'
def box(ax, x, y, w, h, text, fc='#f3f2ee', ec='#9a9a9a', fs=7.2, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.018', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold' if bold else 'normal', linespacing=1.25)
def arrow(ax, p, q, color=SEEDC, ls='-', lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, color=color, lw=lw, ls=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))

def fig_design():
    fig, ax = plt.subplots(figsize=(6.7, 2.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.grid(False)
    box(ax, 0.00, 0.60, 0.19, 0.28, 'Vehicle plant\nhidden parameters\n$\\vartheta$ (friction,\nstiffness, mass)', fc='#fdecea', ec=RED)
    box(ax, 0.00, 0.10, 0.19, 0.28, 'Inputs\nmeasured state,\nfuture commands')
    box(ax, 0.28, 0.10, 0.17, 0.28, 'Hybrid predictor\n$F_{\\rm nom}$ + neural\nresidual: 25-step\npositions')
    box(ax, 0.28, 0.64, 0.17, 0.28, 'Error $e_j(o)$\n$\\|$observed $-$\npredicted$\\|$')
    box(ax, 0.56, 0.52, 0.20, 0.38, 'Input-based scales\nEns, MCD, Het, LLB,\nLLB-N, QNN', fc='#e8f1fb', ec=COL['Ens'])
    box(ax, 0.56, 0.05, 0.20, 0.28, 'Feedback scale\nResidual monitor\n(past one-step errors)', fc='#efecf8', ec=COL['Mon'])
    box(ax, 0.84, 0.28, 0.16, 0.40, 'Layers\nnative radius\nCP (per\nhorizon)\nCP-joint\n(whole path)')
    arrow(ax, (0.095, 0.60), (0.095, 0.38)); ax.text(0.105, 0.49, 'measured', fontsize=6.5, va='center', color=SEEDC)
    arrow(ax, (0.19, 0.76), (0.28, 0.76), color=RED, ls='--'); ax.text(0.235, 0.81, 'observed', fontsize=6.5, ha='center', color=RED)
    ax.text(0.02, 0.97, '$\\vartheta$ is never an input and shifts abruptly', fontsize=7, color=RED, va='top', ha='left') if False else None
    ax.text(0.0, 0.99, '$\\vartheta$ shifts and is never an input', fontsize=7, color=RED, ha='left', va='top')
    arrow(ax, (0.19, 0.24), (0.28, 0.24)); arrow(ax, (0.365, 0.38), (0.365, 0.64))
    arrow(ax, (0.45, 0.30), (0.56, 0.60)); arrow(ax, (0.45, 0.16), (0.56, 0.14), color=COL['Mon'])
    arrow(ax, (0.45, 0.78), (0.60, 0.33), color=COL['Mon'], ls=':', rad=-0.2); ax.text(0.55, 0.43, 'past errors\nfed back', fontsize=6.3, color=COL['Mon'], ha='right', style='italic')
    arrow(ax, (0.76, 0.70), (0.84, 0.58)); arrow(ax, (0.76, 0.19), (0.84, 0.38))
    ax.text(0.92, 0.16, 'scored by coverage,\nradius, CRPS,\nresponse to shift', ha='center', va='top', fontsize=6.8, color=INK)
    fig.savefig(f'{OUT}/figs/fig_design.pdf', bbox_inches='tight'); plt.close(fig)

def fig_shifts():
    rng = np.random.default_rng(3); fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    x = np.linspace(-0.2, 1.8, 200); f = lambda x: 0.6 * np.sin(2.2 * x) + 0.3 * x
    xt = rng.uniform(0, 1, 60); yt = f(xt) + 0.05 * rng.standard_normal(60)
    members = [lambda x, a=a, b=b: f(x) + a * np.clip(x - 1, 0, None) ** 1.5 * 2.2 + b * (np.clip(x - 1, 0, None)) for a, b in [(0.9, 0.2), (-0.7, -0.3), (0.3, 0.6), (-0.2, -0.5), (0.1, 0.0)]]
    titles = ['(a) in distribution', '(b) covariate shift\n(inputs leave the range)', '(c) concept shift\n(hidden parameter changes)']
    for k, a in enumerate(ax):
        a.scatter(xt, yt, s=5, color='#bdbdb9', lw=0, label='training data'); a.axvspan(0, 1, color='#f3f2ee', zorder=0)
        for m in members: a.plot(x, m(x), color=COL['Ens'], lw=0.8, alpha=0.75)
        a.plot(x, f(x), color=INK, lw=1.4)
        if k == 1:
            xn = np.array([1.3, 1.45, 1.6]); a.scatter(xn, f(xn), s=16, color=RED, zorder=5)
            a.annotate('members disagree\nspread $\\uparrow$, error $\\uparrow$', (0.5, 1.12), fontsize=7, color=INK, ha='center', va='center')
        if k == 2:
            fs = lambda x: f(x) - 0.55; xn = np.array([0.3, 0.55, 0.8]); a.plot(x[(x > -0.2) & (x < 1.0)], fs(x[(x > -0.2) & (x < 1.0)]), color=RED, lw=1.4, ls='--'); a.scatter(xn, fs(xn), s=16, color=RED, zorder=5)
            a.annotate('members agree\nspread $\\rightarrow$, error $\\uparrow$', (0.5, 1.12), fontsize=7, color=INK, ha='center', va='center')
        if k == 0: a.annotate('members agree\nspread small, error small', (0.5, 1.12), fontsize=7, color=INK, ha='center', va='center')
        a.set_title(titles[k], fontsize=7.6); a.set_xlabel('input (e.g. speed)'); a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.2, 1.8); a.set_ylim(-1.15, 1.45)
        a.text(0.5, -1.05, 'training range', ha='center', fontsize=6.3, color=SEEDC)
    ax[0].set_ylabel('next-state residual')
    fig.savefig(f'{OUT}/figs/fig_shifts.pdf', bbox_inches='tight'); plt.close(fig)

def fig_episode(seed=0, sname='TEST_ice'):
    D = pickle.load(open(f'{RES}/eval_seed{seed}.pkl', 'rb'))
    def keep(R):
        k = np.ones(len(R['meta']['o']), bool)
        for m, r in R['res'].items(): k &= np.all([np.isfinite(a).all(1) for a in r.values()], axis=0)
        return k
    kc = keep(D['CAL_ID']); R = D[sname]; k = keep(R); meta = {q: R['meta'][q][k] for q in ('o', 'ep', 'ts')}
    rad = {}; scl = {}
    for m in ['Det', 'Ens', 'Het', 'QNN', 'Mon']:
        cal = D['CAL_ID']['res'][m]; ce = cal['err'][kc]; cs = cal['s'][kc] if 's' in cal else None; r = R['res'][m]
        if m == 'Det': fl = np.zeros(25); S = lambda x: np.ones_like(x); s_te = np.ones_like(r['err'][k])
        else: fl = np.maximum(AN.FLOOR, AN.RHO * np.median(ce, 0)); S = lambda x, fl=fl: np.maximum(x, fl[None]); s_te = r['s'][k]
        kap, lam = AN.calib(ce, S(cs) if m != 'Det' else np.ones_like(ce)); rad[m] = lam * kap[None] * S(s_te); scl[m] = (r['s'][k][:, 24] / np.median(D['TEST_id']['res'][m]['s'][keep(D['TEST_id'])][:, 24])) if m != 'Det' else None
    err = R['res']['Det']['err'][k]
    # episode with median post-shift error (not hand-picked)
    eps = np.unique(meta['ep']); med = np.array([np.median(err[(meta['ep'] == i) & (meta['o'] >= meta['ts'])][:, 24]) for i in eps]); ep = eps[np.argsort(med)[len(eps) // 2]]
    sel = meta['ep'] == ep; o = meta['o'][sel]; ts = meta['ts'][sel][0]
    e25 = err[sel][:, 24]; e_id = np.median(D['TEST_id']['res']['Det']['err'][keep(D['TEST_id'])][:, 24])
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 4.1), sharex=True, gridspec_kw=dict(hspace=0.12, height_ratios=[1.15, 1]))
    a = ax[0]; a.scatter(o, e25, s=9, color=INK, lw=0, zorder=4, label='error $e_{25}$')
    for m, lab in [('Det', 'CP-joint, constant scale'), ('Ens', 'CP-joint, ensemble scale'), ('Mon', 'CP-joint, residual monitor')]:
        a.plot(o, rad[m][sel][:, 24], color=COL[m], lw=1.8 if m != 'Det' else 1.4, ls='--' if m == 'Det' else '-', label=lab)
    a.set_ylabel('radius / error at $j=25$ (m)')
    b = ax[1]
    for m in ['Ens', 'Het', 'QNN', 'Mon']: b.plot(o, scl[m][sel], color=COL[m], lw=1.6, label=NAMES[m])
    b.plot(o, e25 / e_id, color=INK, lw=0.9, alpha=0.8, label='error (single net)')
    b.set_yscale('log'); b.set_ylabel('scale / in-distribution median'); b.set_xlabel('origin $o$ (time step)')
    for a_ in ax: a_.axvline(ts, color=RED, lw=1.0, ls=':'); a_.axvspan(ts, o.max() + 1, color='#fdecea', alpha=0.45, zorder=0)
    ax[0].text(ts + 0.8, ax[0].get_ylim()[1] * 0.96, 'friction drops to ice', color=RED, fontsize=7.3, va='top')
    ax[0].legend(loc='lower left', fontsize=7, ncol=4, bbox_to_anchor=(-0.01, 1.01), handlelength=1.6, columnspacing=1.0, frameon=False)
    b.legend(loc='upper center', fontsize=7, ncol=5, bbox_to_anchor=(0.5, -0.32), handlelength=1.6, columnspacing=1.0, frameon=False)
    b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    fig.savefig(f'{OUT}/figs/fig_episode.pdf', bbox_inches='tight'); plt.close(fig)

def fig_react():
    fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.55), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 0.9]))
    mk = {'TEST_mu': 'o', 'TEST_ice': 's', 'TEST_cs': '^'}
    for a, conds, ttl in [(ax[0], ['TEST_mu', 'TEST_ice', 'TEST_cs'], '(a) concept shift'), (ax[1], ['TEST_cov'], '(b) covariate shift')]:
        lim = np.array([0.7, 40]); a.plot(lim, lim, color='#9a9a9a', lw=0.9, ls='--'); a.fill_between(lim, lim, 0.1, color='#f3f2ee', zorder=0)
        for s in conds:
            for m in SCALED:
                x = aa('score', m, s, 'e_ratio25') if s == 'TEST_cov' else aa('score', 'Mon', s, 'e_ratio25'); y = aa('score', m, s, 's_ratio25')
                a.plot([x], [y], marker=(mk.get(s, 'D')), color=COL[m], ms=5.8, ls='none', mec='white', mew=0.5)
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(*lim); a.set_ylim(0.7, 40); a.set_xlabel('error ratio (post-shift / id)'); a.set_title(ttl, fontsize=8.5)
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}')); a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax[0].set_ylabel('scale ratio at $j=25$')
    for s, lab in mk.items(): ax[0].plot([], [], marker=lab, color='#6b6b6b', ls='none', ms=5, label={'TEST_mu': 'friction drop', 'TEST_ice': 'ice', 'TEST_cs': 'stiffness'}[s])
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.86), fontsize=6.6, handletextpad=0.1)
    c = ax[2]; ys = np.arange(len(SCALED))[::-1]
    for dy, s, mkr in [(0.17, 'TEST_mu', 'o'), (-0.17, 'TEST_cs', '^')]:
        for y, m in zip(ys, SCALED): c.plot([aa('dose', m, s, 'rho_sev_s')], [y + dy], marker=mkr, color=COL[m], ms=5.5, ls='none', mec='white', mew=0.5)
    c.axvline(0, color='#9a9a9a', lw=0.8); c.set_yticks(ys); c.set_yticklabels([m for m in SCALED], fontsize=7); c.set_xlim(-0.2, 0.9); c.set_xlabel('Spearman $\\rho$ with true severity'); c.set_title('(c) dose--response', fontsize=8.5)
    c.plot([], [], 'o', color='#6b6b6b', ms=5, label='friction drop'); c.plot([], [], '^', color='#6b6b6b', ms=5, label='stiffness'); c.legend(loc='center right', fontsize=6.6, handletextpad=0.1)
    hs = [plt.Line2D([], [], marker=MRK[m], color=COL[m], ls='none', ms=5.5, label=NAMES[m]) for m in SCALED]
    fig.legend(handles=hs, loc='lower center', ncol=7, fontsize=6.6, bbox_to_anchor=(0.5, -0.19), handletextpad=0.1, columnspacing=0.7)
    fig.savefig(f'{OUT}/figs/fig_react.pdf', bbox_inches='tight'); plt.close(fig)

def fig_conformal():
    fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.5), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.05]))
    order = ['Nom', 'Det'] + SCALED; a = ax[0]; ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        c1 = mm(m, 'cp', 'TEST_id', 'joint'); c2 = mm(m, 'cpj', 'TEST_id', 'joint'); a.plot([c1, c2], [y, y], color='#bdbdb9', lw=1.2, zorder=1)
        a.plot([c1], [y], marker='o', mfc='white', mec=COL[m], mew=1.5, ms=6, ls='none', zorder=3); a.plot([c2], [y], marker='o', color=COL[m], ms=6, ls='none', zorder=3)
    a.axvline(0.9, color=INK, lw=0.8, ls='--'); a.set_yticks(ys); a.set_yticklabels(order, fontsize=7.2); a.set_xlim(0.05, 1.0); a.set_xlabel('joint coverage in distribution')
    a.plot([], [], 'o', mfc='white', mec='#6b6b6b', mew=1.5, ls='none', label='per-horizon CP'); a.plot([], [], 'o', color='#6b6b6b', ls='none', label='CP-joint'); a.legend(loc='lower left', fontsize=6.8, handletextpad=0.1, bbox_to_anchor=(0.0, 0.0)); a.set_title('(a) marginal calibration is not joint', fontsize=8.3)
    b = ax[1]
    for m in order:
        x = mm(m, 'cpj', 'TEST_id', 'wavg'); y = np.mean([mm(m, 'cpj', s, 'joint') for s in ('TEST_mu', 'TEST_ice', 'TEST_cs')])
        b.plot([x], [y], marker=MRK[m], color=COL[m], ms=7, ls='none', mec='white', mew=0.5)
        if m in ('Nom', 'Mon', 'QNN'): b.annotate({'Nom': 'Nominal', 'Mon': 'Residual monitor', 'QNN': 'QNN'}[m], (x, y), xytext=((5, 4) if m != 'QNN' else (-3, -11)), textcoords='offset points', fontsize=6.8, color=INK, ha='left' if m != 'QNN' else 'center')
    b.annotate('Single net, Ens, MCD,\nHet, LLB, LLB-N', (0.40, 0.2), xytext=(0.47, 0.42), fontsize=6.8, color=INK, arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.7))
    b.axhline(0.9, color=INK, lw=0.8, ls='--'); b.set_xlabel('mean radius in distribution (m), at 90% joint coverage'); b.set_ylabel('joint coverage after the shifts\n(mean of friction, ice, stiffness)'); b.set_ylim(-0.03, 1.0); b.set_title('(b) narrow in distribution $\\neq$ robust', fontsize=8.3)
    b.text(0.03, 0.9, 'target $0.90$', transform=b.get_yaxis_transform(), fontsize=6.8, color=SEEDC, ha='left', va='bottom') if False else None
    fig.savefig(f'{OUT}/figs/fig_conformal.pdf', bbox_inches='tight'); plt.close(fig)

def fig_scorecard():
    order = ['Nom', 'Det'] + SCALED
    cols = [('Native\nper-step cov.\n(id)', lambda m: mm(m, 'native', 'TEST_id', 'cov') if m in NATIVE else np.nan, 'unit'),
            ('Native\njoint cov.\n(id)', lambda m: mm(m, 'native', 'TEST_id', 'joint') if m in NATIVE else np.nan, 'unit'),
            ('AUROC\nice', lambda m: aa('score', m, 'TEST_ice', 'auroc_shift') if m not in ('Nom', 'Det') else np.nan, 'unit'),
            ('Dose $\\rho$\nfriction', lambda m: aa('dose', m, 'TEST_mu', 'rho_sev_s') if m not in ('Nom', 'Det') else np.nan, 'unit'),
            ('AUROC\nfast (cov.)', lambda m: aa('score', m, 'TEST_cov', 'auroc_shift') if m not in ('Nom', 'Det') else np.nan, 'unit'),
            ('CP-joint\ncov. (id)', lambda m: mm(m, 'cpj', 'TEST_id', 'joint'), 'unit'),
            ('CP-joint cov.\nafter shifts', lambda m: np.mean([mm(m, 'cpj', s, 'joint') for s in ('TEST_mu', 'TEST_ice', 'TEST_cs')]), 'unit'),
            ('Mean radius\n(id, m)', lambda m: mm(m, 'cpj', 'TEST_id', 'wavg'), 'inv')]
    V = np.array([[fn(m) for _, fn, _ in cols] for m in order]); cm = LinearSegmentedColormap.from_list('b', ['#eef4fb', '#9cc2ee', '#1f5fa8'])
    fig, ax = plt.subplots(figsize=(6.7, 3.2)); ax.grid(False)
    for j, (_, _, kind) in enumerate(cols):
        for i in range(len(order)):
            v = V[i, j]
            if np.isnan(v): ax.add_patch(Rectangle((j, i), 1, 1, fc='white', ec='white')); ax.text(j + 0.5, i + 0.5, '--', ha='center', va='center', fontsize=7, color='#9a9a9a'); continue
            u = v if kind == 'unit' else 1 - (v - np.nanmin(V[:, j])) / (np.nanmax(V[:, j]) - np.nanmin(V[:, j]) + 1e-9)
            u = float(np.clip(u, 0, 1)); ax.add_patch(Rectangle((j, i), 1, 1, fc=cm(u), ec='white', lw=1.5))
            ax.text(j + 0.5, i + 0.5, f'{abs(v) if abs(v) < 0.005 else v:.2f}', ha='center', va='center', fontsize=7.3, color='white' if u > 0.55 else INK)
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(order), 0); ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels([c[0] for c in cols], fontsize=6.8); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(order)) + 0.5); ax.set_yticklabels([NAMES[m] for m in order], fontsize=7.3)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.axvline(2, color=INK, lw=1.0); ax.axvline(5, color=INK, lw=1.0)
    for x0, x1, t in [(0, 2, 'native intervals'), (2, 5, 'reaction to shifts'), (5, 8, 'conformal layer')]: ax.text((x0 + x1) / 2, len(order) + 0.45, t, ha='center', va='top', fontsize=7.6, style='italic')
    fig.savefig(f'{OUT}/figs/fig_scorecard.pdf', bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    for fn in (fig_design, fig_shifts, fig_episode, fig_react, fig_conformal, fig_scorecard): fn(); print(fn.__name__, 'ok')
