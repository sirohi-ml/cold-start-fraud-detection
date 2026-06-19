"""
Cold-Start Ranking Experiment on IEEE-CIS Fraud Detection Dataset
=================================================================
Real-data validation of the uncertainty-aware placement framework.

Setup:
  - Warm cards: cards with >= 10 transactions (known fraud rate)
  - Cold cards: cards with 2-5 transactions (new, uncertain)
  - At each time step t, cold card reveals t transaction outcomes
  - Bayesian (Beta-Binomial) update of fraud rate estimate
  - 4 placement strategies applied, metrics measured at each t
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from scipy.stats import kendalltau
import os, warnings
warnings.filterwarnings('ignore')

SEED       = 42
N_WARM     = 150
N_COLD     = 60
T_STEPS    = 20
K          = 20
LCB_K      = 1.5
TIER_BAND  = (120, 140)
GRAD_THRESH= 8
CONV_TOL   = 5
N_RUNS     = 20
PRIOR_A    = 0.5
PRIOR_B    = 13.5   # prior mean = 0.5/14 ≈ 3.5% pop fraud rate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.environ.get('IEEE_CSV_PATH', os.path.join(BASE_DIR, 'data', 'train_transaction.csv'))
OUTD = os.path.join(BASE_DIR, 'results')
FIGD = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTD, exist_ok=True)
os.makedirs(FIGD, exist_ok=True)

STRATEGIES = ['Naive', 'LCB', 'Tiered', 'Random']
COLORS     = {'Naive':'#E74C3C','LCB':'#2ECC71','Tiered':'#3498DB','Random':'#95A5A6'}

# Load & group
print('Loading...')
df = pd.read_csv(PATH, usecols=['isFraud','card1'])
cs = df.groupby('card1').agg(n=('isFraud','count'), k=('isFraud','sum')).reset_index()
cs['true_rate'] = cs['k'] / cs['n']
warm_pool = cs[cs.n >= 10].reset_index(drop=True)
cold_pool = cs[(cs.n >= 2) & (cs.n <= 5)].reset_index(drop=True)
print(f'Warm: {len(warm_pool)} | Cold: {len(cold_pool)}')

def beta_post(k_obs, n_obs):
    a = PRIOR_A + k_obs
    b = PRIOR_B + n_obs - k_obs
    mu    = a / (a + b)
    sigma = np.sqrt(a*b / ((a+b)**2 * (a+b+1)))
    return float(mu), float(sigma)

def place(wr_sorted, mu, sigma, t, strat, rng):
    if strat == 'Naive':
        return min(int(np.searchsorted(-wr_sorted, -mu)), N_WARM)
    if strat == 'LCB':
        lcb = max(mu - LCB_K*sigma, 0.0)
        return min(int(np.searchsorted(-wr_sorted, -lcb)), N_WARM)
    if strat == 'Tiered':
        if t < GRAD_THRESH:
            lo, hi = min(TIER_BAND[0], N_WARM), min(TIER_BAND[1], N_WARM)
            return int(rng.integers(lo, hi+1))
        lcb = max(mu - LCB_K*sigma, 0.0)
        return min(int(np.searchsorted(-wr_sorted, -lcb)), N_WARM)
    return int(rng.integers(0, N_WARM+1))

def dcg(s, k):
    s = np.asarray(s)[:k]
    return float(np.sum(s / np.log2(np.arange(2, len(s)+2)))) if len(s) else 0.

def ndcg_k(merged, ideal, k):
    id_ = dcg(np.sort(np.asarray(ideal))[::-1], k)
    return dcg(merged, k) / id_ if id_ > 0 else 1.

def rd_fn(pre_topk, pos):
    sh = np.where(pre_topk >= pos, pre_topk+1, pre_topk)
    return float(np.abs(sh - pre_topk).mean())

def lss_fn(pre, pos):
    po = np.where(pre >= pos, pre+1, pre)
    tau, _ = kendalltau(pre, po)
    return float(tau)

def ptkr_fn(pos, op, sg, thr=0.05):
    return int(pos < K and sg > thr and pos < op)

def run_once(strat, rng):
    warm = warm_pool.sample(N_WARM, random_state=int(rng.integers(0,99999)))
    cold = cold_pool.sample(N_COLD, random_state=int(rng.integers(0,99999)))
    wr   = np.sort(warm.true_rate.values)[::-1]
    pre  = np.arange(N_WARM)
    rows = []
    for t in range(T_STEPS+1):
        nv,rv,lv,pv,cv,sv = [],[],[],[],[],[]
        for _, c in cold.iterrows():
            k_obs = int(rng.binomial(t, c.true_rate)) if t > 0 else 0
            mu, sg = beta_post(k_obs, t)
            pos    = place(wr, mu, sg, t, strat, rng)
            op     = int(np.searchsorted(-wr, -c.true_rate))
            merged = np.insert(wr, pos, c.true_rate)
            ideal  = np.sort(np.append(wr, c.true_rate))[::-1]
            nv.append(ndcg_k(merged, ideal, K))
            rv.append(rd_fn(pre[:K], pos))
            lv.append(lss_fn(pre, pos))
            pv.append(ptkr_fn(pos, op, sg))
            cv.append(1 if abs(pos-op) <= CONV_TOL else 0)
            sv.append(sg)
        rows.append({'t':t,'strategy':strat,
                     'ndcg':np.mean(nv),'rd':np.mean(rv),'lss':np.mean(lv),
                     'ptkr':np.mean(pv),'conv':np.mean(cv),'sigma':np.mean(sv)})
    return pd.DataFrame(rows)

print(f'Running {N_RUNS} runs x {len(STRATEGIES)} strategies...')
all_dfs = []
for ri in range(N_RUNS):
    rng = np.random.default_rng(SEED+ri)
    for s in STRATEGIES:
        d = run_once(s, rng); d['run'] = ri; all_dfs.append(d)
    if (ri+1)%5==0: print(f'  {ri+1}/{N_RUNS}')

full = pd.concat(all_dfs, ignore_index=True)
agg  = (full.groupby(['strategy','t'])
        .agg(ndcg_m=('ndcg','mean'), ndcg_s=('ndcg','std'),
             rd_m=('rd','mean'),     rd_s=('rd','std'),
             lss_m=('lss','mean'),   lss_s=('lss','std'),
             ptkr_m=('ptkr','mean'), conv_m=('conv','mean'),
             sigma_m=('sigma','mean')).reset_index())

full.to_csv(f'{OUTD}/ieee_metrics_all.csv', index=False)
agg.to_csv(f'{OUTD}/ieee_metrics_aggregated.csv', index=False)

rows = []
for s in STRATEGIES:
    sub = agg[agg.strategy==s].sort_values('t')
    hit = sub[sub.conv_m>=0.80]
    t80 = int(hit.t.iloc[0]) if len(hit) else f'>{T_STEPS}'
    rows.append({'Strategy':s,
        'NDCG@20 t=0' : round(sub[sub.t==0].ndcg_m.iloc[0],4),
        'NDCG@20 t=20': round(sub[sub.t==T_STEPS].ndcg_m.iloc[0],4),
        'RD t=0'      : round(sub[sub.t==0].rd_m.iloc[0],4),
        'RD t=20'     : round(sub[sub.t==T_STEPS].rd_m.iloc[0],4),
        'PTKR peak'   : round(sub.ptkr_m.max(),4),
        '80% Conv'    : t80})

tbl = pd.DataFrame(rows)
tbl.to_csv(f'{OUTD}/ieee_convergence.csv', index=False)
print('\n── IEEE-CIS Summary ────────────────────────────────────────')
print(tbl.to_string(index=False))

# Plots
def shade(ax, strat, ycol, scol):
    sub = agg[agg.strategy==strat].sort_values('t')
    t=sub.t.values; mu=sub[ycol].values; sd=sub[scol].values
    c=COLORS[strat]; ax.plot(t,mu,color=c,lw=2.2,label=strat)
    ax.fill_between(t,mu-sd,mu+sd,color=c,alpha=0.13)

fig = plt.figure(figsize=(18,12))
fig.suptitle('Cold-Start Placement — IEEE-CIS Fraud Detection (Real Data)\n'
    f'N_warm={N_WARM}, N_cold={N_COLD} real cards, K={K}, {N_RUNS} MC runs',
    fontsize=13, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2,3, figure=fig, hspace=0.45, wspace=0.35)

panels = [
    (gs[0,0],'ndcg_m','ndcg_s','NDCG@20','NDCG@20 (↑ better)'),
    (gs[0,1],'rd_m','rd_s','Rank Displacement','Mean |rank shift| top-20 warm cards (↓ better)'),
    (gs[0,2],'lss_m','lss_s','List Stability (Kendall τ)','Kendall τ (↑ more stable)'),
]
for spec,ycol,scol,title,ylabel in panels:
    ax = fig.add_subplot(spec)
    for s in STRATEGIES: shade(ax,s,ycol,scol)
    ax.set_title(title,fontsize=11,fontweight='bold')
    ax.set_xlabel('Transactions revealed (t)'); ax.set_ylabel(ylabel,fontsize=9)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1,0])
for s in STRATEGIES:
    sub=agg[agg.strategy==s].sort_values('t')
    ax4.plot(sub.t,sub.ptkr_m,color=COLORS[s],lw=2.2,label=s)
ax4.set_title('Premature Top-K Rate',fontsize=11,fontweight='bold')
ax4.set_xlabel('Transactions revealed (t)')
ax4.set_ylabel('Fraction uncertain cards in Top-20 (↓ better)',fontsize=9)
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1,1])
for s in STRATEGIES:
    sub=agg[agg.strategy==s].sort_values('t')
    ax5.plot(sub.t,sub.conv_m,color=COLORS[s],lw=2.2,label=s)
ax5.axhline(0.80,color='black',ls='--',lw=1.2,label='80% threshold')
ax5.set_title('Convergence Rate',fontsize=11,fontweight='bold')
ax5.set_xlabel('Transactions revealed (t)')
ax5.set_ylabel(f'Fraction within ±{CONV_TOL} ranks of oracle',fontsize=9)
ax5.set_ylim(0,1.05); ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[1,2])
sub = agg[agg.strategy=='LCB'].sort_values('t')
ax6.plot(sub.t,sub.sigma_m,color='#8E44AD',lw=2.2)
ax6.fill_between(sub.t,0,sub.sigma_m,color='#8E44AD',alpha=0.15)
ax6.axhline(0.05,color='black',ls='--',lw=1.2,label='σ threshold (0.05)')
ax6.set_title('Posterior σ Decay (Real Data)',fontsize=11,fontweight='bold')
ax6.set_xlabel('Transactions revealed (t)'); ax6.set_ylabel('Mean posterior σ')
ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

fig.savefig(f'{FIGD}/ieee_all_metrics.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Figure saved.')
print(