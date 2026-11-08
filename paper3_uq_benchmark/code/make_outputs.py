"""Tables (LaTeX), figures and in-text numbers from metrics.csv / aux.pkl. All numbers in the paper are produced here."""
import sys, math, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RES = '/home/claude/work/paper_uq/results'; OUT = '/home/claude/work/paper_uq/paper'
M = pd.read_csv(f'{RES}/metrics.csv'); A = pd.read_pickle(f'{RES}/aux.pkl')
SEEDS = sorted(M.seed.unique()); NS = len(SEEDS)
SETS = ['TEST_id', 'TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']; SL = {'TEST_id': 'id', 'TEST_mu': '$\\mu$-drop', 'TEST_ice': 'ice', 'TEST_cs': 'stiffness', 'TEST_mass': 'mass'}
WIN = lambda s, late=False: 'ID' if s in ('TEST_id', 'TEST_cov') else ('LATE' if late else 'POST')
NAMES = {'Nom': 'Nominal', 'Det': 'Single net', 'Ens': 'Deep ensemble', 'MCD': 'MC dropout', 'Het': 'Heteroscedastic ens.', 'LLB': 'Last-layer Laplace', 'LLB-N': 'Last-layer Laplace + noise', 'QNN': 'Quantile net', 'Mon': 'Residual monitor'}
NUM = {}   # in-text macros
def mm(method, layer, sname, col, late=False):
    d = M[(M.method == method) & (M.layer == layer) & (M.set == sname) & (M.win == WIN(sname, late))]
    return d[col].mean() if len(d) else np.nan
def mm_rng(method, layer, sname, col, late=False):
    d = M[(M.method == method) & (M.layer == layer) & (M.set == sname) & (M.win == WIN(sname, late))][col]
    return (d.min(), d.max()) if len(d) else (np.nan, np.nan)
def aa(kind, method, sname, col, layer=None, late=False):
    d = A[(A.kind == kind) & (A.method == method) & (A.set == sname) & (A.win == WIN(sname, late))]
    if layer is not None: d = d[d.layer == layer]
    return d[col].mean() if len(d) and col in d else np.nan
def f(x, n=2): return '--' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.{n}f}'
def tab(path, header, rows, align):
    s = '\\begin{tabular}{' + align + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' + '\n'.join(r + ' \\\\' for r in rows) + '\n\\bottomrule\n\\end{tabular}\n'
    open(f'{OUT}/tables/{path}.tex', 'w').write(s)
import os; os.makedirs(f'{OUT}/tables', exist_ok=True); os.makedirs(f'{OUT}/figs', exist_ok=True)

COL = {'Ens': '#2a78d6', 'MCD': '#eb6834', 'Het': '#1baf7a', 'LLB': '#eda100', 'LLB-N': '#e87ba4', 'QNN': '#008300', 'Mon': '#4a3aa7', 'Det': '#6b6b6b', 'Nom': '#9a9a9a'}
MRK = {'Ens': 'o', 'MCD': 's', 'Het': '^', 'LLB': 'D', 'LLB-N': 'v', 'QNN': 'P', 'Mon': 'X', 'Det': '*', 'Nom': 'h'}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'legend.frameon': False,
                     'figure.dpi': 150, 'savefig.dpi': 300, 'axes.titlesize': 9, 'axes.titleweight': 'normal'})
NATIVE = ['Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN']; SAMPLE = ['Ens', 'MCD', 'Het', 'LLB', 'LLB-N']; ALL = ['Nom', 'Det', 'Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN', 'Mon']
SCALED = ['Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN', 'Mon']
SH = ['TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']
def b(x, n=2, bold=False): s = f(x, n); return f'\\textbf{{{s}}}' if bold and s != '--' else s

# ------------------------------------------------------------------ tables
def table_acc():
    rows = []
    for m in ['Nom', 'Det', 'Ens', 'MCD', 'Het', 'LLB', 'LLB-N']:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(aa('score', m, s, 'med_e25'), 2) for s in SETS))
    tab('acc', 'Estimator (own point prediction) & ' + ' & '.join(SL[s] for s in SETS), rows, 'l' + 'r' * 5)
def table_native():
    rows = []
    for m in NATIVE:
        c = [mm(m, 'native', s, 'cov') for s in SETS]; j = [mm(m, 'native', s, 'joint') for s in SETS]
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(x) for x in c) + ' & ' + ' & '.join(f(x) for x in j) + ' & ' + f(aa('rel', m, 'TEST_id', 'ece', layer='native') if m != 'QNN' else np.nan, 3))
    hdr = 'Estimator & ' + ' & '.join(SL[s] for s in SETS) + ' & ' + ' & '.join(SL[s] for s in SETS) + ' & ECE (id)'
    tab('native', hdr, rows, 'l' + 'r' * 5 + 'r' * 5 + 'r')
def table_scale():
    rows = []
    for m in SCALED:
        rows.append(NAMES[m] + ' & ' + f(aa('score', m, 'TEST_id', 'rho25')) + ' & ' + ' & '.join(f(aa('score', m, s, 's_ratio25'), 2) for s in SH) + ' & ' + ' & '.join(f(aa('score', m, s, 'auroc_shift'), 2) for s in SH))
    er = [aa('score', 'Mon', s, 'e_ratio25') for s in SH]
    rows.append('\\midrule\nMedian error ratio (single net) & -- & ' + ' & '.join(f(x, 2) for x in er) + ' & \\multicolumn{4}{c}{--}')
    hdr = 'Estimator & $\\rho_{25}$ (id) & ' + ' & '.join(SL[s] for s in SH) + ' & ' + ' & '.join(SL[s] for s in SH)
    tab('scale', hdr, rows, 'lr' + 'r' * 4 + 'r' * 4)
def table_dose():
    rows = []
    for m in SCALED:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(aa('dose', m, s, 'rho_sev_s')) for s in SH))
    rows.append('\\midrule\nSingle-net error & ' + ' & '.join(f(aa('dose', 'Mon', s, 'rho_sev_e')) for s in SH))
    tab('dose', 'Estimator & ' + ' & '.join(SL[s] for s in SH), rows, 'l' + 'r' * 4)
def table_cp(layer, name, col='joint', widths=True):
    rows = []
    for m in ALL:
        v = [mm(m, layer, s, col) for s in SETS]
        r = NAMES[m] + ' & ' + ' & '.join(f(x) for x in v)
        if widths: r += ' & ' + f(mm(m, layer, 'TEST_id', 'wavg')) + ' & ' + f(mm(m, layer, 'TEST_ice', 'wavg'))
        rows.append(r)
    hdr = 'Estimator (scale) & ' + ' & '.join(SL[s] for s in SETS) + (' & $\\bar r$ id & $\\bar r$ ice' if widths else '')
    tab(name, hdr, rows, 'l' + 'r' * 5 + ('rr' if widths else ''))
def table_oracle():
    rows = []
    for m in ALL:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(mm(m, 'cp', s, 'wavg_oracle')) for s in SETS) + ' & ' + ' & '.join(f(mm(m, 'cp', s, 'c90')) for s in SETS))
    tab('oracle', 'Estimator (scale) & ' + ' & '.join(SL[s] for s in SETS) + ' & ' + ' & '.join(SL[s] for s in SETS), rows, 'l' + 'r' * 10)
def table_scores():
    rows = []
    for m in SAMPLE:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(aa('score', m, s, 'crps'), 3) for s in SETS) + ' & ' + ' & '.join(f(aa('score', m, s, 'es'), 3) for s in SETS))
    tab('scores', 'Estimator & ' + ' & '.join(SL[s] for s in SETS) + ' & ' + ' & '.join(SL[s] for s in SETS), rows, 'l' + 'r' * 10)

# ------------------------------------------------------------------ hypotheses
LEARNED = ['Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN']
def hyp_one():
    """Evaluate the pre-registered predictions H1-H8 on the current (M, A); returns {H: (holds, evidence-string)}."""
    R = {}
    v = {m: (mm(m, 'native', 'TEST_id', 'cov'), mm(m, 'native', 'TEST_id', 'joint')) for m in ['Ens', 'MCD', 'LLB']}
    R['H1'] = (all(c < 0.85 and j < 0.5 for c, j in v.values()), 'per-step ' + '/'.join(f(v[m][0]) for m in v) + '; joint $\\le$' + f(max(x[1] for x in v.values())))
    dev = {m: abs(mm(m, 'native', 'TEST_id', 'cov') - 0.9) for m in NATIVE}; best = min(dev, key=dev.get); ice_cov = mm('Het', 'native', 'TEST_ice', 'cov')
    R['H2'] = (best == 'Het' and ice_cov < 0.85, f'closest to 0.90: {NAMES[best]} ($|\\Delta|$={dev[best]:.2f}); Het: {dev["Het"]:.2f}; Het post-ice {ice_cov:.2f}')
    d3 = {m: {s: (aa('score', m, s, 's_ratio25'), aa('score', m, s, 'e_ratio25')) for s in ('TEST_ice', 'TEST_mu')} for m in ['Ens', 'MCD', 'LLB']}
    smax = max(d3[m][s][0] for m in d3 for s in d3[m])
    R['H3'] = (smax < 1.5 and all(d3['Ens'][s][1] > 1.5 for s in ('TEST_ice', 'TEST_mu')), f'spread ratio $\\le${smax:.2f}; error ratio {d3["Ens"]["TEST_ice"][1]:.1f} (ice), {d3["Ens"]["TEST_mu"][1]:.1f} ($\\mu$)')
    d4 = {m: aa('score', m, 'TEST_ice', 'auroc_shift') for m in ['Ens', 'MCD', 'LLB', 'Mon']}
    R['H4'] = (all(d4[m] < 0.7 for m in ['Ens', 'MCD', 'LLB']) and d4['Mon'] > 0.9, 'AUROC (ice) ' + '/'.join(f(d4[m]) for m in ['Ens', 'MCD', 'LLB']) + f'; monitor {d4["Mon"]:.2f}')
    d5 = {m: {s: mm(m, 'cpj', s, 'joint') - mm('Det', 'cpj', s, 'joint') for s in SH} for m in LEARNED + ['Mon']}
    g = max((x, m, s) for m in LEARNED for s, x in d5[m].items())
    R['H5'] = (g[0] <= 0.05, f'largest gain over single-net CP: {g[0]:+.2f} ({g[1]}, {SL[g[2]].replace("$", "")}); monitor {max(d5["Mon"].values()):+.2f}')
    d6 = {m: {s: mm(m, 'cpj', s, 'joint') for s in ('TEST_mu', 'TEST_ice', 'TEST_cs')} for m in ALL}; g6 = max(x for v_ in d6.values() for x in v_.values())
    R['H6'] = (g6 < 0.9, f'largest joint coverage: {g6:.2f}')
    d7 = {m: aa('dose', m, 'TEST_mu', 'rho_sev_s') for m in ['Ens', 'MCD', 'LLB', 'Mon']}
    R['H7'] = (all(d7[m] < 0.3 for m in ['Ens', 'MCD', 'LLB']) and d7['Mon'] > 0.5, '$\\rho$ = ' + '/'.join(f(d7[m]) for m in ['Ens', 'MCD', 'LLB']) + f'; monitor {d7["Mon"]:.2f}')
    r_id = aa('score', 'Nom', 'TEST_id', 'med_e25') / aa('score', 'Det', 'TEST_id', 'med_e25'); r_ice = aa('score', 'Det', 'TEST_ice', 'med_e25') / aa('score', 'Nom', 'TEST_ice', 'med_e25')
    R['H8'] = (r_id >= 2 and r_ice > 0.8, f'nominal/learned error {r_id:.1f}$\\times$ (id); learned/nominal {r_ice:.2f} (ice)')
    return R
def hyp_report():
    global M, A
    M0, A0 = M, A; pooled = hyp_one(); per = {}
    for sd in SEEDS:
        M = M0[M0.seed == sd]; A = A0[A0.seed == sd]; per[sd] = hyp_one()
    M, A = M0, A0
    L = []
    for h in pooled: L.append(f'{h}: pooled={pooled[h][0]} seeds={[per[s][h][0] for s in SEEDS]} | {pooled[h][1]}')
    open(f'{RES}/hyp_report.txt', 'w').write('\n'.join(L)); print('\n'.join(L)); return pooled, per

CLAIM = {'H1': 'Epistemic spreads (Ens, MCD, LLB) under-cover natively, even in distribution',
         'H2': 'Heteroscedastic ensemble is the best-calibrated native estimator in distribution, but under-covers after shift',
         'H3': 'Epistemic spread does not react to dynamics shifts (ratio $<1.5$) while the error does ($>1.5$)',
         'H4': 'Shift-detection AUROC of the spread $<0.7$ (Ens, MCD, LLB); $>0.9$ for the residual monitor (ice)',
         'H5': 'Normalising by a learned spread does not raise post-shift joint coverage by more than 0.05',
         'H6': 'Every statically calibrated layer has joint coverage $<0.9$ after $\\mu$-drop, ice, stiffness',
         'H7': 'Severity--spread correlation $<0.3$ (Ens, MCD, LLB) and $>0.5$ (monitor), $\\mu$-drop',
         'H8': 'Learned models $\\ge2\\times$ more accurate in distribution, but not after ice (ratio $>0.8$)'}
def table_hyp(R):
    pooled, per = R; rows = []
    for h in pooled:
        ok, ev = pooled[h]; k = sum(per[s][h][0] for s in SEEDS)
        rows.append(f'{h} & \\parbox[t]{{5.6cm}}{{{CLAIM[h]}}} & \\parbox[t]{{5.2cm}}{{{ev}}} & {"supported" if ok else "not supported"} & {k}/{len(SEEDS)} \\\\')
    s = '\\begin{tabular}{@{}llp{5.2cm}lc@{}}\n\\toprule\nNo. & Pre-registered prediction & Observed & Verdict & Seeds \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
    open(f'{OUT}/tables/hyp.tex', 'w').write(s)

# ------------------------------------------------------------------ figures
def lst(kind, method, sname, key, layer=None, win=None):
    d = A[(A.kind == kind) & (A.method == method) & (A.set == sname) & (A.win == (win or WIN(sname)))]
    if layer is not None: d = d[d.layer == layer]
    return np.mean([np.array(x) for x in d[key]], axis=0) if len(d) else None
def fig_native():
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.35), gridspec_kw=dict(wspace=0.38))
    hh = np.arange(1, 26)
    for m in NATIVE:
        v = lst('covj', m, 'TEST_id', 'covj', layer='native')
        if v is not None: ax[0].plot(hh, v, color=COL[m], lw=1.6, marker=MRK[m], ms=0, label=NAMES[m])
    ax[0].axhline(0.9, color='#0b0b0b', lw=0.8, ls='--'); ax[0].set_xlabel('horizon $j$ (steps)'); ax[0].set_ylabel('per-step coverage (id)'); ax[0].set_ylim(-0.02, 1.02); ax[0].set_title('(a) native 90% radius')
    taus = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    for k, (sname, ttl) in enumerate([('TEST_id', '(b) reliability, id'), ('TEST_ice', '(c) reliability, ice (post)')]):
        a = ax[k + 1]; a.plot([0, 1], [0, 1], color='#0b0b0b', lw=0.8, ls='--')
        for m in SAMPLE:
            v = lst('rel', m, sname, 'cov', layer='native')
            if v is not None: a.plot(taus, v, color=COL[m], lw=1.6, marker=MRK[m], ms=3.5)
        a.set_xlabel('nominal level'); a.set_ylabel('empirical coverage'); a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.set_title(ttl)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.17), handlelength=1.4, columnspacing=1.2)
    fig.savefig(f'{OUT}/figs/fig_native.pdf', bbox_inches='tight'); plt.close(fig)
def fig_response():
    fig, ax = plt.subplots(1, 4, figsize=(6.6, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    for a, s in zip(ax, SH):
        er = aa('score', 'Mon', s, 'e_ratio25')
        for i, m in enumerate(SCALED):
            a.plot([i], [aa('score', m, s, 's_ratio25')], marker=MRK[m], color=COL[m], ms=6.5, ls='none')
        a.axhline(er, color='#0b0b0b', lw=1.0, ls='--'); a.axhline(1, color='#9a9a9a', lw=0.8)
        a.text(len(SCALED) - 0.5, er * 1.07, f'error ×{er:.1f}', ha='right', va='bottom', fontsize=7)
        a.set_yscale('log'); a.set_xticks([]); a.set_xlim(-0.6, len(SCALED) - 0.4); a.set_title(SL[s].replace('$', '').replace('\\mu', 'μ') + (' (null)' if s == 'TEST_mass' else ''))
    ax[0].set_ylabel('post-shift / id median at $j=25$'); ax[0].set_ylim(0.6, 20)
    hs = [plt.Line2D([], [], marker=MRK[m], color=COL[m], ls='none', ms=6, label=NAMES[m]) for m in SCALED]
    fig.legend(handles=hs, loc='lower center', ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.1), handletextpad=0.2, columnspacing=0.9)
    fig.savefig(f'{OUT}/figs/fig_response.pdf', bbox_inches='tight'); plt.close(fig)
def fig_dose(sname='TEST_mu'):
    ms_ = ['Det', 'Ens', 'Het', 'Mon']; fig, ax = plt.subplots(1, 4, figsize=(6.6, 2.1), gridspec_kw=dict(wspace=0.62))
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for a, m in zip(ax, ms_):
        d = A[(A.kind == 'doseep') & (A.method == ('Mon' if m == 'Det' else m)) & (A.set == sname)]
        sv = np.concatenate([np.array(x) for x in d.sev]); y = np.concatenate([np.array(x) for x in (d.me if m == 'Det' else d.ms)])
        from scipy import stats as st
        rho = st.spearmanr(sv, y)[0]
        a.scatter(sv, y, s=7, color=COL[m], alpha=0.55, lw=0)
        a.set_title(('single-net error' if m == 'Det' else NAMES[m] + ' scale').replace('Heteroscedastic ens.', 'Het. ensemble') + f'\n$\\rho$={rho:.2f}', fontsize=8)
        a.set_xlabel('true severity'); a.set_yscale('log'); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a.yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel('post-shift median at $j=25$ (m)')
    fig.savefig(f'{OUT}/figs/fig_dose.pdf', bbox_inches='tight'); plt.close(fig)
def fig_joint(layer='cpj'):
    fig, ax = plt.subplots(1, 5, figsize=(6.6, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    order = ['Nom', 'Det'] + SCALED
    for a, s in zip(ax, SETS):
        for i, m in enumerate(order):
            d = M[(M.method == m) & (M.layer == layer) & (M.set == s) & (M.win == WIN(s))]
            if not len(d): continue
            y = d.joint.mean(); a.plot([i, i], [d.joint_lo.mean(), d.joint_hi.mean()], color=COL[m], lw=1.1); a.plot([i], [y], marker=MRK[m], color=COL[m], ms=5.5, ls='none')
        a.axhline(0.9, color='#0b0b0b', lw=0.8, ls='--'); a.set_xticks([]); a.set_title(SL[s].replace('$', '').replace('\\mu', 'μ') + (' (null)' if s == 'TEST_mass' else ''), fontsize=8.5); a.set_ylim(0, 1.02)
    ax[0].set_ylabel('joint coverage of 25 steps')
    hs = [plt.Line2D([], [], marker=MRK[m], color=COL[m], ls='none', ms=5.5, label=NAMES[m]) for m in order]
    fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.17), handletextpad=0.2, columnspacing=0.9)
    fig.savefig(f'{OUT}/figs/fig_joint.pdf', bbox_inches='tight'); plt.close(fig)


def table_cov():
    if 'TEST_cov' not in set(M.set): return
    rows = []
    for m in SCALED:
        rows.append(NAMES[m] + ' & ' + f(aa('score', m, 'TEST_cov', 's_ratio25')) + ' & ' + f(aa('score', m, 'TEST_cov', 'e_ratio25')) + ' & ' + f(aa('score', m, 'TEST_cov', 'auroc_shift')) + ' & '
                    + f(mm(m, 'native', 'TEST_cov', 'cov')) + ' & ' + f(mm(m, 'cpj', 'TEST_id', 'joint')) + ' & ' + f(mm(m, 'cpj', 'TEST_cov', 'joint')))
    tab('cov', 'Estimator & Spread ratio & Error ratio & AUROC & Native cov. & CPJ joint (id) & CPJ joint (fast)', rows, 'lrrrrrr')
def table_late():
    rows = []
    for m in ALL:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(mm(m, 'cpj', s, 'joint', late=True)) for s in SH) + ' & ' + ' & '.join(f(mm(m, 'cpj', s, 'joint')) for s in SH))
    tab('late', 'Estimator & ' + ' & '.join(SL[s] for s in SH) + ' & ' + ' & '.join(SL[s] for s in SH), rows, 'l' + 'r' * 8)
def table_div():
    d = A[(A.kind == 'diverge') & (A.set.isin(SETS + ['CAL_ID', 'CAL_SH']))]; rows = []
    for m in ['Het']:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(100 * d[(d.method == m) & (d.set == s)].frac.mean(), 2) for s in ['CAL_ID', 'CAL_SH'] + SETS))
    dr = A[(A.kind == 'dropped') & (A.set.isin(SETS + ['CAL_ID', 'CAL_SH']))]
    rows.append('Origins dropped (all estimators) & ' + ' & '.join(f(100 * dr[dr.set == s].frac.mean(), 2) for s in ['CAL_ID', 'CAL_SH'] + SETS))
    tab('div', 'Diverged rollouts (\\%) & CAL-id & CAL-shift & ' + ' & '.join(SL[s] for s in SETS), rows, 'l' + 'r' * 7)
def table_rho():
    import os
    labs = [('0', '_rho0'), ('0.01', ''), ('0.1', '_rho01'), ('1', '_rho1')]
    if not all(os.path.exists(f'{RES}/metrics{sf}.csv') for _, sf in labs): return
    Ms = {lab: pd.read_csv(f'{RES}/metrics{sf}.csv') for lab, sf in labs}; rows = []
    def g(Mx, m, layer, s, col):
        d = Mx[(Mx.method == m) & (Mx.layer == layer) & (Mx.set == s) & (Mx.win == WIN(s))]; return d[col].mean()
    for m in ['Ens', 'MCD', 'Het', 'LLB', 'LLB-N']:
        rows.append(NAMES[m] + ' & ' + ' & '.join(f(g(Ms[l], m, 'cp', 'TEST_id', 'joint')) for l, _ in labs) + ' & ' + ' & '.join(f(g(Ms[l], m, 'cpj', 'TEST_id', 'wavg')) for l, _ in labs)
                    + ' & ' + ' & '.join(f(g(Ms[l], m, 'cpj', 'TEST_ice', 'joint')) for l, _ in labs))
    sub = ' & '.join(f'{l}' for l, _ in labs)
    hdr = ('& \\multicolumn{4}{c}{CP joint coverage, id} & \\multicolumn{4}{c}{CP-joint mean radius, id (m)} & \\multicolumn{4}{c}{CP-joint coverage, ice} \\\\\n'
           '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\nEstimator $\\backslash$ floor factor & ' + sub + ' & ' + sub + ' & ' + sub)
    tab('rho', hdr, rows, 'l' + 'r' * 12)
def table_ens10():
    import os, pickle
    path = f'{RES}/eval_seed0_ens10.pkl'
    if not os.path.exists(path): return
    E10 = pickle.load(open(path, 'rb')); E5 = pickle.load(open(f'{RES}/eval_seed0.pkl', 'rb')); from analyze import C90, windows, auroc
    rows = []; N_ = 25
    def one(res, sname):
        r = res[sname]; return r
    out = {}
    for lab, src in (('5', {k: E5[k]['res']['Ens'] for k in E10}), ('10', {k: E10[k] for k in E10})):
        o = {}
        sid = src['TEST_id']['s']; eid = src['TEST_id']['err']
        o['cov_id'] = float((eid <= C90 * sid).mean()); o['joint_id'] = float((eid <= C90 * sid).all(1).mean())
        for s_ in ('TEST_mu', 'TEST_ice', 'TEST_cs'):
            meta = E5[s_]['meta']; mask = windows(meta, s_)['POST']; ss = src[s_]['s'][mask]; ee = src[s_]['err'][mask]
            o[s_] = (float(np.median(ss[:, 24]) / np.median(sid[:, 24])), float(auroc(ss.mean(1), sid.mean(1))))
        out[lab] = o
    for lab in ('5', '10'):
        o = out[lab]; rows.append(f'$K={lab}$ & ' + f(o['cov_id']) + ' & ' + f(o['joint_id']) + ' & ' + ' & '.join(f(o[s][0]) + ' / ' + f(o[s][1]) for s in ('TEST_mu', 'TEST_ice', 'TEST_cs')))
    tab('ens10', 'Ensemble size & native cov. (id) & native joint (id) & spread ratio / AUROC ($\\mu$) & ice & stiffness', rows, 'lrrrrr')
    json.dump(out, open(f'{RES}/ens10_summary.json', 'w'))

if __name__ == '__main__':
    R = hyp_report()
    table_hyp(R)
    for fn in (table_acc, table_native, table_scale, table_dose, table_oracle, table_scores):
        fn()
    table_cp('cp', 'cp_marg'); table_cp('cpj', 'cpj'); table_cp('cpj-mix', 'cpj_mix')
    for fn in (table_cov, table_late, table_div, table_rho, table_ens10): fn()
    fig_native(); fig_response(); fig_dose(); fig_joint()
    print('outputs written')
