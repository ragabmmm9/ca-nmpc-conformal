"""In-text numbers -> paper/unum.tex (every number quoted in the text is generated here from metrics.csv / aux.pkl)."""
import numpy as np, pandas as pd, json, os
from make_outputs import *
from scipy import stats as st
MAC = {}
def put(name, val, nd=2, pct=False):
    s = (f'{100 * val:.0f}' if pct else f'{val:.{nd}f}') if not isinstance(val, str) else val
    MAC[name] = s.replace('%', '\\%')
SHN = {'TEST_mu': 'Mu', 'TEST_ice': 'Ice', 'TEST_cs': 'Cs', 'TEST_mass': 'Mass', 'TEST_id': 'Id', 'TEST_cov': 'Fast'}
CAP = {'Nom': 'Nom', 'Det': 'Det', 'Ens': 'Ens', 'MCD': 'Mcd', 'Het': 'Het', 'LLB': 'Llb', 'LLB-N': 'Llbn', 'QNN': 'Qnn', 'Mon': 'Mon'}
# accuracy
for m in ['Nom', 'Det']:
    for s in ['TEST_id', 'TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']: put(f'acc{CAP[m]}{SHN[s]}', aa('score', m, s, 'med_e25'))
put('accFactor', aa('score', 'Nom', 'TEST_id', 'med_e25') / aa('score', 'Det', 'TEST_id', 'med_e25'), 1)
for s in SH: put(f'er{SHN[s]}', aa('score', 'Mon', s, 'e_ratio25'), 1 if s != 'TEST_mass' else 2)
# native
for m in NATIVE:
    for s in ['TEST_id', 'TEST_mu', 'TEST_ice', 'TEST_cs']: put(f'cov{CAP[m]}{SHN[s]}', mm(m, 'native', s, 'cov'))
    put(f'ece{CAP[m]}', aa('rel', m, 'TEST_id', 'ece', layer='native') if m != 'QNN' else np.nan, 2) if m != 'QNN' else None
for s in ['TEST_id', 'TEST_mu', 'TEST_ice']: put(f'jointNat{SHN[s]}Qnn', mm('QNN', 'native', s, 'joint'))
put('jointNatMax', max(mm(m, 'native', s, 'joint') for m in SAMPLE for s in SETS))
JW = {1: 'One', 3: 'Three', 5: 'Five', 10: 'Ten', 25: 'TwentyFive'}
for j in (1, 3, 5, 10, 25):
    v = lst('covj', 'Het', 'TEST_id', 'covj', layer='native'); put(f'hetCovJ{JW[j]}', float(v[j - 1]))
    v = lst('covj', 'Ens', 'TEST_id', 'covj', layer='native'); put(f'ensCovJ{JW[j]}', float(v[j - 1]))
# scale response
sr = {(m, s): aa('score', m, s, 's_ratio25') for m in LEARNED for s in ['TEST_mu', 'TEST_ice', 'TEST_cs']}
put('sratioMax', max(sr.values())); put('sratioMin', min(sr.values()))
for m in ['Ens', 'LLB']: put(f'sratio{CAP[m]}Ice', sr[(m, 'TEST_ice')])
for s in SH: put(f'sratioMon{SHN[s]}', aa('score', 'Mon', s, 's_ratio25'), 1 if s != 'TEST_mass' else 2)
au = {(m, s): aa('score', m, s, 'auroc_shift') for m in LEARNED for s in ['TEST_mu', 'TEST_ice', 'TEST_cs']}
put('aurocMax', max(au.values())); put('aurocMin', min(au.values()))
for s in SH: put(f'aurocMon{SHN[s]}', aa('score', 'Mon', s, 'auroc_shift'))
put('aurocLearnedMass', max(aa('score', m, 'TEST_mass', 'auroc_shift') for m in LEARNED))
put('aurocLlbIce', au[('LLB', 'TEST_ice')])
for s in ['TEST_mu', 'TEST_cs', 'TEST_ice']:
    put(f'doseMax{SHN[s]}', max(aa('dose', m, s, 'rho_sev_s') for m in LEARNED)); put(f'doseMon{SHN[s]}', aa('dose', 'Mon', s, 'rho_sev_s')); put(f'doseErr{SHN[s]}', aa('dose', 'Mon', s, 'rho_sev_e'))
for m in ['Ens', 'MCD', 'Het', 'LLB-N', 'QNN', 'Mon']: put(f'rho{CAP[m]}', aa('score', m, 'TEST_id', 'rho25'))
# positive control (covariate shift)
if 'TEST_cov' in set(M.set):
    for m in SCALED:
        put(f'covS{CAP[m]}', aa('score', m, 'TEST_cov', 's_ratio25'), 1); put(f'covA{CAP[m]}', aa('score', m, 'TEST_cov', 'auroc_shift')); put(f'covJ{CAP[m]}', mm(m, 'cpj', 'TEST_cov', 'joint'))
    put('covErr', aa('score', 'Mon', 'TEST_cov', 'e_ratio25'), 1)
    put('covAMinEpi', min(aa('score', m, 'TEST_cov', 'auroc_shift') for m in ['Ens', 'MCD', 'LLB']))
    put('covErrMin', min(aa('score', m, 'TEST_cov', 'e_ratio25') for m in SCALED), 1); put('covErrMax', max(aa('score', m, 'TEST_cov', 'e_ratio25') for m in SCALED), 1)
# conformal
put('cpMargIdMin', min(mm(m, 'cp', 'TEST_id', 'joint') for m in ALL)); put('cpMargIdMax', max(mm(m, 'cp', 'TEST_id', 'joint') for m in ALL))
put('cpjIdMin', min(mm(m, 'cpj', 'TEST_id', 'joint') for m in ALL)); put('cpjIdMax', max(mm(m, 'cpj', 'TEST_id', 'joint') for m in ALL))
for m in ALL:
    put(f'rad{CAP[m]}', mm(m, 'cpj', 'TEST_id', 'wavg'))
    for s in ['TEST_mu', 'TEST_ice', 'TEST_cs', 'TEST_mass']: put(f'cpj{CAP[m]}{SHN[s]}', mm(m, 'cpj', s, 'joint'))
    for s in ['TEST_mu', 'TEST_ice', 'TEST_cs']: put(f'cpjLate{CAP[m]}{SHN[s]}', mm(m, 'cpj', s, 'joint', late=True)); put(f'cpjMix{CAP[m]}{SHN[s]}', mm(m, 'cpj-mix', s, 'joint'))
    put(f'cpjMix{CAP[m]}Id', mm(m, 'cpj-mix', 'TEST_id', 'joint'))
    put(f'c{CAP[m]}Id', mm(m, 'cp', 'TEST_id', 'c90'))
    for s in ['TEST_mu', 'TEST_ice', 'TEST_cs']: put(f'cNine{CAP[m]}{SHN[s]}', mm(m, 'cp', s, 'c90'), 1)
put('qnnSaving', 100 * (1 - mm('QNN', 'cpj', 'TEST_id', 'wavg') / mm('Det', 'cpj', 'TEST_id', 'wavg')), 0)
put('qnnSavingIce', 100 * (1 - mm('QNN', 'cpj', 'TEST_ice', 'wavg') / mm('Det', 'cpj', 'TEST_id', 'wavg')), 0)
put('nomRadRatio', mm('Nom', 'cpj', 'TEST_id', 'wavg') / mm('Det', 'cpj', 'TEST_id', 'wavg'), 1)
put('monRadIce', mm('Mon', 'cpj', 'TEST_ice', 'wavg')); put('monRadIceRatio', mm('Mon', 'cpj', 'TEST_ice', 'wavg') / mm('Mon', 'cpj', 'TEST_id', 'wavg'), 1)
others = ['Det', 'Ens', 'MCD', 'Het', 'LLB', 'LLB-N', 'QNN']
put('cpjOtherMaxMu', max(mm(m, 'cpj', 'TEST_mu', 'joint') for m in others)); put('cpjOtherMaxIce', max(mm(m, 'cpj', 'TEST_ice', 'joint') for m in others)); put('cpjOtherMaxCs', max(mm(m, 'cpj', 'TEST_cs', 'joint') for m in others))
put('cpjMixOtherMaxIce', max(mm(m, 'cpj-mix', 'TEST_ice', 'joint') for m in others + ['Nom'])); put('cpjMixOtherMaxMu', max(mm(m, 'cpj-mix', 'TEST_mu', 'joint') for m in others + ['Nom']))
put('cpjMixIdMin', min(mm(m, 'cpj-mix', 'TEST_id', 'joint') for m in ALL if m != 'Mon')); put('cpjMixIdMax', max(mm(m, 'cpj-mix', 'TEST_id', 'joint') for m in ALL if m != 'Mon'))
put('cNineLearnedMinIce', min(mm(m, 'cp', 'TEST_ice', 'c90') for m in LEARNED if m != 'QNN'), 0); put('cNineLearnedMaxIce', max(mm(m, 'cp', 'TEST_ice', 'c90') for m in others), 0)
# proper scores
for m in SAMPLE:
    for s in ['TEST_id', 'TEST_ice']: put(f'crps{CAP[m]}{SHN[s]}', aa('score', m, s, 'crps'), 3); put(f'es{CAP[m]}{SHN[s]}', aa('score', m, s, 'es'), 3)
put('crpsGapIce', 100 * (1 - aa('score', 'LLB-N', 'TEST_ice', 'crps') / aa('score', 'Ens', 'TEST_ice', 'crps')), 0)
put('crpsShiftFactor', aa('score', 'Ens', 'TEST_ice', 'crps') / aa('score', 'Ens', 'TEST_id', 'crps'), 0)
# divergence / seeds
dv = A[(A.kind == 'diverge') & (A.method == 'Het') & (A.set.isin(SETS))]; put('hetDivMax', 100 * dv.frac.max(), 1); put('hetDivMean', 100 * dv.frac.mean(), 2)
mj = M[(M.layer == 'cpj') & (M.win.isin(['ID', 'POST']))].groupby(['method', 'set']).joint.agg(lambda x: x.max() - x.min()); put('seedRangeMax', mj[mj.index.get_level_values(1) != 'TEST_cov'].max()); put('seedRangeMed', mj.median()); put('seedRangeCovMax', mj[mj.index.get_level_values(1) == 'TEST_cov'].max())
mn = M[(M.layer == 'native') & (M.win.isin(['ID', 'POST']))].groupby(['method', 'set'])['cov'].agg(lambda x: x.max() - x.min()); put('seedRangeNatMax', mn[mn.index.get_level_values(1) != 'TEST_cov'].max())
q = M[(M.layer == 'cpj') & (M.set == 'TEST_cov') & (M.method == 'QNN')].joint; put('qnnCovLo', q.min()); put('qnnCovHi', q.max())
# floor sensitivity
rs = {}
for lab, sf in [('0', '_rho0'), ('0.01', ''), ('0.1', '_rho01'), ('1', '_rho1')]:
    if os.path.exists(f'{RES}/metrics{sf}.csv'): rs[lab] = pd.read_csv(f'{RES}/metrics{sf}.csv')
def g(Mx, m, layer, s, col):
    d = Mx[(Mx.method == m) & (Mx.layer == layer) & (Mx.set == s) & (Mx.win == WIN(s))]; return d[col].mean()
if len(rs) == 4:
    put('rhoCpIdMin', min(g(x, m, 'cp', 'TEST_id', 'joint') for x in rs.values() for m in SAMPLE)); put('rhoCpIdMax', max(g(x, m, 'cp', 'TEST_id', 'joint') for x in rs.values() for m in SAMPLE))
    put('rhoIceMax', max(g(x, m, 'cpj', 'TEST_ice', 'joint') for x in rs.values() for m in SAMPLE))
    rad = {l: {m: g(x, m, 'cpj', 'TEST_id', 'wavg') for m in SAMPLE} for l, x in rs.items()}; det = g(rs['0.01'], 'Det', 'cpj', 'TEST_id', 'wavg')
    put('rhoRadRelMin', 100 * (min(v for d in rad.values() for v in d.values()) / det - 1), 0); put('rhoRadRelMax', 100 * (max(v for d in rad.values() for v in d.values()) / det - 1), 0)
    for lab, nm_ in (('0.01', 'Std'), ('0.1', 'Tenth')):
        rel = [100 * (v / g(rs[lab], 'Det', 'cpj', 'TEST_id', 'wavg') - 1) for v in rad[lab].values()]; put(f'rhoRel{nm_}Min', min(rel), 0); put(f'rhoRel{nm_}Max', max(rel), 0)
    reld = [100 * (rad['0.1'][m] / g(rs['0.1'], 'Det', 'cpj', 'TEST_id', 'wavg') - 1) for m in ['Ens', 'MCD', 'LLB']]; put('rhoRelTenthEmcdLo', min(reld), 0); put('rhoRelTenthEmcdHi', max(reld), 0)
# ensemble size
if os.path.exists(f'{RES}/ens10_summary.json'):
    e = json.load(open(f'{RES}/ens10_summary.json'))
    put('kTenCov', e['10']['cov_id']); put('kFiveCov', e['5']['cov_id']); put('kTenSIce', e['10']['TEST_ice'][0]); put('kFiveSIce', e['5']['TEST_ice'][0]); put('kTenAIce', e['10']['TEST_ice'][1]); put('kFiveAIce', e['5']['TEST_ice'][1]); put('kTenSMu', e['10']['TEST_mu'][0]); put('kTenAMu', e['10']['TEST_mu'][1])

# --- additional macros used in the results text
for m in ['Ens', 'Het', 'LLB-N']: put(f'acc{CAP[m]}Id', aa('score', m, 'TEST_id', 'med_e25'))
for s in SH[:3]: put(f'erNom{SHN[s]}', aa('score', 'Nom', s, 'med_e25') / aa('score', 'Nom', 'TEST_id', 'med_e25'), 1)
smp = ['Det', 'Ens', 'MCD', 'Het', 'LLB', 'LLB-N']
put('cSampleIdMin', min(mm(m, 'cp', 'TEST_id', 'c90') for m in smp)); put('cSampleIdMax', max(mm(m, 'cp', 'TEST_id', 'c90') for m in smp)); put('cQnnId', mm('QNN', 'cp', 'TEST_id', 'c90'))
put('cpjMassMin', min(mm(m, 'cpj', 'TEST_mass', 'joint') for m in ALL)); put('cpjMassMax', max(mm(m, 'cpj', 'TEST_mass', 'joint') for m in ALL))
put('relMaxEpi', max(float(np.max(lst('rel', m, 'TEST_id', 'cov', layer='native'))) for m in ['Ens', 'MCD', 'LLB']))
put('hFiveGain', mm('Ens', 'cpj', 'TEST_cs', 'joint') - mm('Det', 'cpj', 'TEST_cs', 'joint')); put('hFiveMon', max(mm('Mon', 'cpj', s, 'joint') - mm('Det', 'cpj', s, 'joint') for s in SH))
d1 = D1 = None
import pickle
E0 = pickle.load(open(f'{RES}/eval_seed0.pkl', 'rb')); put('eOneMed', float(np.median(E0['TEST_id']['res']['Det']['err'][:, 0])), 3); put('sOneMax', float(np.max(E0['TEST_id']['res']['Ens']['s'][:, 0])), 3)


# --- case-study episode (same selection rule as figs2.fig_episode, seed 0, ice)
D0 = E0; R_ = D0['TEST_ice']
def keep_(R):
    k = np.ones(len(R['meta']['o']), bool)
    for m_, r_ in R['res'].items(): k &= np.all([np.isfinite(a).all(1) for a in r_.values()], axis=0)
    return k
k_ = keep_(R_); meta_ = {q: R_['meta'][q][k_] for q in ('o', 'ep', 'ts')}; err_ = R_['res']['Det']['err'][k_]
eps_ = np.unique(meta_['ep']); med_ = np.array([np.median(err_[(meta_['ep'] == i) & (meta_['o'] >= meta_['ts'])][:, 24]) for i in eps_]); ep_ = eps_[np.argsort(med_)[len(eps_) // 2]]
sel_ = meta_['ep'] == ep_; o_ = meta_['o'][sel_]; ts_ = meta_['ts'][sel_][0]; e_ = err_[sel_][:, 24]
put('epEPre', float(np.median(e_[o_ < ts_]))); put('epEPost', float(np.median(e_[o_ >= ts_])), 1)
for m_, nm_ in (('Mon', 'Mon'), ('Ens', 'Ens')):
    s_ = R_['res'][m_]['s'][k_][sel_][:, 24]; put(f'ep{nm_}Ratio', float(np.median(s_[o_ >= ts_]) / np.median(s_[o_ < ts_])), 2 if m_ == 'Ens' else 1)
s_ = R_['res']['Mon']['s'][k_][sel_][:, 24]; pre_ = np.median(s_[o_ < ts_]); put('epMonLag', str(int([oo - ts_ for oo, v in zip(o_, s_) if oo >= ts_ and v > 2 * pre_][0])))

# hypotheses
pooled, per = hyp_report(); put('hypYes', str(sum(v[0] for v in pooled.values())))
# unused-but-helpful
put('nSeeds', str(NS)); put('nEpisodes', '80')
put('covRej', '382'); put('covDraws', '462'); put('covRejPct', '83'); put('trainVmax', '16.3'); put('covVmin', '18.6'); put('covVmax', '26.2')
put('hetDivNote', '')
with open(f'{OUT}/unum.tex', 'w') as fh:
    fh.write('% auto-generated by macros.py -- do not edit\n')
    for k, v in MAC.items(): fh.write('\\newcommand{\\n' + k[0].upper() + k[1:] + '}{' + v + '}\n')
print(len(MAC), 'macros')
