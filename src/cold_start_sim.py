"""
Cold-Start Ranking Simulation  (v2)
=====================================
Research question: Does naive score-based insertion of cold-start entities
degrade ranked list quality for existing (warm) entities, and by how much?
Does uncertainty-aware placement fix this?

Key design choice (v2):
  - We specifically study HIGH true-score cold entities: those whose true
    score would place them in the top-30% of the warm list.  This is the
    hardest and most consequential case — a strong new applicant or a potent
    new fraud signal that the model initially under/over-estimates.
  - N_WARM=100, K=10 — tighter list, effect sizes are visible.
  - Added metric: Premature Top-K Rate (PTKR) — fraction of time steps where
    a cold entity enters Top-K before its uncertainty has collapsed.

Four strategies:
  Naive   — rank by point estimate μ
  LCB     — rank by μ − k·σ  (conservative)
  Tiered  — cold entity locked in provisional band until t ≥ GRAD_THRESH,
             then graduates via LCB
  Random  — control condition
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kendalltau
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Simulation parameters ──────────────────────────────────────────────────────
N_WARM        = 100      # existing warm entities
N_COLD        = 30       # cold entities (only high-true-score ones kept)
T_STEPS       = 60       # time steps
K             = 10       # top-K for all metrics
LCB_K         = 1.5      # coefficient in μ − k·σ
TIER_BAND     = (70, 85) # provisional band positions in warm list
GRAD_THRESH   = 20       # interactions to graduate from tiered band
SCORE_ALPHA   = 5.0      # Beta shape — warm entity scores
SCORE_BETA    = 2.0
NOISE_SIGMA   = 0.10     # observation noise for cold entity
COLD_PCTILE   = 0.70     # only keep cold entities with true score > 70th pctile
                          # of warm scores  →  these SHOULD be top-ranked
CONV_TOL      = 5        # convergence: cold entity within ±CONV_TOL of oracle rank
N_RUNS        = 50

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

STRATEGIES = ["Naive", "LCB", "Tiered", "Random"]
COLORS     = {
    "Naive" : "#E74C3C",
    "LCB"   : "#2ECC71",
    "Tiered": "#3498DB",
    "Random": "#95A5A6",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_warm_list(rng):
    scores = rng.beta(SCORE_ALPHA, SCORE_BETA, size=N_WARM)
    return np.sort(scores)[::-1]   # rank-0 = highest score


def generate_cold_entities(warm_scores, rng):
    """
    Draw cold entity true scores until we have N_COLD with true score
    above the COLD_PCTILE threshold of warm scores.  These are the
    entities that SHOULD appear near the top of the list.
    """
    threshold = np.quantile(warm_scores, COLD_PCTILE)
    scores = []
    while len(scores) < N_COLD:
        s = rng.beta(SCORE_ALPHA, SCORE_BETA, size=N_COLD * 5)
        scores.extend(s[s > threshold])
    true_scores = np.array(scores[:N_COLD])

    pop_mean   = SCORE_ALPHA / (SCORE_ALPHA + SCORE_BETA)
    init_mu    = np.full(N_COLD, pop_mean)
    init_sigma = np.full(N_COLD, 0.30)
    return true_scores, init_mu, init_sigma


# ══════════════════════════════════════════════════════════════════════════════
# Bayesian update
# ══════════════════════════════════════════════════════════════════════════════

def update_estimate(mu0, sigma0, true_score, t, rng):
    if t == 0:
        return mu0, sigma0
    obs        = true_score + rng.normal(0, NOISE_SIGMA, size=t)
    obs_mean   = obs.mean()
    prior_prec = 1.0 / sigma0**2
    lik_prec   = t   / NOISE_SIGMA**2
    post_prec  = prior_prec + lik_prec
    post_mu    = (prior_prec * mu0 + lik_prec * obs_mean) / post_prec
    post_sigma = np.sqrt(1.0 / post_prec)
    return float(post_mu), float(post_sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Placement strategies
# ══════════════════════════════════════════════════════════════════════════════

def place_naive(warm_scores, mu, sigma, t, rng):
    return int(np.searchsorted(-warm_scores, -mu))

def place_lcb(warm_scores, mu, sigma, t, rng):
    lcb = np.clip(mu - LCB_K * sigma, 0.0, 1.0)
    return int(np.searchsorted(-warm_scores, -lcb))

def place_tiered(warm_scores, mu, sigma, t, rng):
    if t < GRAD_THRESH:
        lo, hi = TIER_BAND
        lo, hi = min(lo, N_WARM), min(hi, N_WARM)
        return int(rng.integers(lo, hi + 1))
    return place_lcb(warm_scores, mu, sigma, t, rng)

def place_random(warm_scores, mu, sigma, t, rng):
    return int(rng.integers(0, N_WARM + 1))

STRATEGY_FNS = {
    "Naive" : place_naive,
    "LCB"   : place_lcb,
    "Tiered": place_tiered,
    "Random": place_random,
}


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def dcg(scores, k):
    s = np.asarray(scores)[:k]
    if len(s) == 0:
        return 0.0
    return float(np.sum(s / np.log2(np.arange(2, len(s) + 2))))

def ndcg_at_k(ranked_scores, ideal_scores, k):
    ideal = np.sort(np.asarray(ideal_scores))[::-1]
    idcg  = dcg(ideal, k)
    return dcg(ranked_scores, k) / idcg if idcg > 0 else 1.0

def rank_displacement(pre_top_k_pos, cold_inserted_at):
    """
    Mean rank shift of the top-K warm entities after inserting cold entity
    at position `cold_inserted_at`.
    Warm entities whose pre-insertion rank >= cold_inserted_at shift down 1.
    """
    shifted = np.where(pre_top_k_pos >= cold_inserted_at,
                       pre_top_k_pos + 1,
                       pre_top_k_pos)
    return float(np.abs(shifted - pre_top_k_pos).mean())

def list_stability(pre_order, cold_inserted_at):
    """
    Kendall τ between warm entity ordering before and after cold insertion.
    """
    post_order = np.where(pre_order >= cold_inserted_at,
                          pre_order + 1,
                          pre_order)
    tau, _ = kendalltau(pre_order, post_order)
    return float(tau)

def premature_topk(pos, oracle_pos, sigma, sigma_threshold=0.08):
    """
    Returns 1 if cold entity enters top-K *before* uncertainty has collapsed
    (sigma > sigma_threshold) AND the model-assigned position is better than
    oracle (i.e., overconfident upward placement).
    """
    in_topk   = pos < K
    uncertain = sigma > sigma_threshold
    overplaced = pos < oracle_pos   # model placed it higher than it deserves
    return int(in_topk and uncertain and overplaced)


# ══════════════════════════════════════════════════════════════════════════════
# Single run
# ══════════════════════════════════════════════════════════════════════════════

def run_single(strategy_name, rng):
    warm_scores               = generate_warm_list(rng)
    true_cold, init_mu, init_sigma = generate_cold_entities(warm_scores, rng)
    place_fn  = STRATEGY_FNS[strategy_name]
    pre_order = np.arange(N_WARM)    # warm entity positions before insertion

    records = []
    for t in range(T_STEPS + 1):
        ndcg_v, rd_v, lss_v, ptkr_v, conv_v, sigma_v = [], [], [], [], [], []

        for c in range(N_COLD):
            mu_t, sig_t = update_estimate(init_mu[c], init_sigma[c],
                                          true_cold[c], t, rng)
            pos = min(place_fn(warm_scores, mu_t, sig_t, t, rng), N_WARM)

            oracle_pos = int(np.searchsorted(-warm_scores, -true_cold[c]))

            # NDCG: merged list, true scores as relevance
            merged      = np.insert(warm_scores, pos, true_cold[c])
            ideal_all   = np.sort(np.append(warm_scores, true_cold[c]))[::-1]
            ndcg_v.append(ndcg_at_k(merged, ideal_all, K))

            # Rank displacement of top-K warm entities
            rd_v.append(rank_displacement(pre_order[:K], pos))

            # List stability (Kendall τ over all warm entities)
            lss_v.append(list_stability(pre_order, pos))

            # Premature Top-K Rate
            ptkr_v.append(premature_topk(pos, oracle_pos, sig_t))

            # Convergence: within ±CONV_TOL of oracle rank
            conv_v.append(1 if abs(pos - oracle_pos) <= CONV_TOL else 0)

            sigma_v.append(sig_t)

        records.append({
            "t"        : t,
            "strategy" : strategy_name,
            "ndcg"     : np.mean(ndcg_v),
            "rd"       : np.mean(rd_v),
            "lss"      : np.mean(lss_v),
            "ptkr"     : np.mean(ptkr_v),
            "conv_rate": np.mean(conv_v),
            "sigma"    : np.mean(sigma_v),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    print(f"Running {N_RUNS} Monte Carlo runs × {len(STRATEGIES)} strategies "
          f"× {T_STEPS+1} steps  (N_warm={N_WARM}, N_cold={N_COLD}, K={K}) ...")
    all_dfs = []
    for run_i in range(N_RUNS):
        run_rng = np.random.default_rng(SEED + run_i)
        for s in STRATEGIES:
            df = run_single(s, run_rng)
            df["run"] = run_i
            all_dfs.append(df)
        if (run_i + 1) % 10 == 0:
            print(f"  {run_i+1}/{N_RUNS} done")

    full = pd.concat(all_dfs, ignore_index=True)
    agg  = (full.groupby(["strategy", "t"])
               .agg(
                   ndcg_mean  = ("ndcg",      "mean"),
                   ndcg_std   = ("ndcg",      "std"),
                   rd_mean    = ("rd",        "mean"),
                   rd_std     = ("rd",        "std"),
                   lss_mean   = ("lss",       "mean"),
                   lss_std    = ("lss",       "std"),
                   ptkr_mean  = ("ptkr",      "mean"),
                   ptkr_std   = ("ptkr",      "std"),
                   conv_mean  = ("conv_rate", "mean"),
                   sigma_mean = ("sigma",     "mean"),
               )
               .reset_index())
    return full, agg


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def summary_table(agg):
    rows = []
    for s in STRATEGIES:
        sub = agg[agg.strategy == s].sort_values("t")
        hit = sub[sub.conv_mean >= 0.80]
        t80 = int(hit.t.iloc[0]) if len(hit) else f">{T_STEPS}"

        rows.append({
            "Strategy"          : s,
            "NDCG@K t=0"        : round(sub[sub.t==0].ndcg_mean.iloc[0], 4),
            "NDCG@K t=60"       : round(sub[sub.t==T_STEPS].ndcg_mean.iloc[0], 4),
            "RD t=0"            : round(sub[sub.t==0].rd_mean.iloc[0], 3),
            "RD t=60"           : round(sub[sub.t==T_STEPS].rd_mean.iloc[0], 3),
            "PTKR t=0"          : round(sub[sub.t==0].ptkr_mean.iloc[0], 3),
            "PTKR peak"         : round(sub.ptkr_mean.max(), 3),
            "80% Conv. Step"    : t80,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def shade(ax, agg, strat, metric):
    sub = agg[agg.strategy == strat].sort_values("t")
    t   = sub.t.values
    mu  = sub[f"{metric}_mean"].values
    sd  = sub[f"{metric}_std"].values
    c   = COLORS[strat]
    ax.plot(t, mu, color=c, lw=2.2, label=strat)
    ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.13)


def make_plots(agg):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Cold-Start Placement Strategies — Ranking Quality for HIGH-SCORE Cold Entities\n"
        f"N_warm={N_WARM}, N_cold={N_COLD} (true score > 70th pctile), K={K}, {N_RUNS} MC runs",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    panels = [
        (gs[0,0], "ndcg",  "NDCG@K",           "NDCG@10\n(higher = better list quality)"),
        (gs[0,1], "rd",    "Rank Displacement", "Mean rank shift of top-10 warm entities\n(lower = less disruption)"),
        (gs[0,2], "lss",   "List Stability",    "Kendall τ of warm list\n(higher = more stable)"),
        (gs[1,0], "ptkr",  "Premature Top-K",   "Fraction entering Top-K before σ collapses\n(lower = fewer false promotions)"),
    ]

    for spec, metric, title, ylabel in panels:
        ax = fig.add_subplot(spec)
        for s in STRATEGIES:
            shade(ax, agg, s, metric)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Interactions observed (t)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Convergence
    ax5 = fig.add_subplot(gs[1,1])
    for s in STRATEGIES:
        sub = agg[agg.strategy == s].sort_values("t")
        ax5.plot(sub.t, sub.conv_mean, color=COLORS[s], lw=2.2, label=s)
    ax5.axhline(0.80, color="black", ls="--", lw=1.2, label="80% threshold")
    ax5.set_title("Convergence Rate", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Interactions observed (t)")
    ax5.set_ylabel(f"Fraction within ±{CONV_TOL} ranks of oracle")
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # Sigma decay
    ax6 = fig.add_subplot(gs[1,2])
    sub = agg[agg.strategy == "LCB"].sort_values("t")
    ax6.plot(sub.t, sub.sigma_mean, color="#8E44AD", lw=2.2)
    ax6.fill_between(sub.t, 0, sub.sigma_mean, color="#8E44AD", alpha=0.15)
    ax6.axhline(0.08, color="black", ls="--", lw=1.2, label="σ threshold (0.08)")
    ax6.set_title("Posterior σ Decay", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Interactions observed (t)")
    ax6.set_ylabel("Mean posterior σ")
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    fig.savefig(os.path.join(FIG_DIR, "all_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {FIG_DIR}/all_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    full, agg = run_all()
    full.to_csv(os.path.join(OUT_DIR, "metrics_all.csv"), index=False)
    agg.to_csv(os.path.join(OUT_DIR, "metrics_aggregated.csv"), index=False)

    tbl = summary_table(agg)
    tbl.to_csv(os.path.join(OUT_DIR, "convergence.csv"), index=False)

    print("\n── Summary Table ─────────────────────────────────────────────────")
    print(tbl.to_string(index=False))
    print()

    make_plots(agg)
    print("All outputs saved.")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        