# Cold Start in Fraud Detection: Scoring vs. Ranking

Code and experiments behind the article *"Scoring and Ranking Are Two Different
Problems: Rethinking Cold Start in Fraud Detection."*

The core idea: scoring a new entity (a card with no transaction history) and
deciding where it should sit in a ranked review queue are two different
problems. A new card's *uncertainty* — not just its point-estimate fraud
score — should determine its placement. This repo implements and benchmarks
four placement strategies (Naive, LCB, Tiered, Random) under a Bayesian
(Beta-Binomial) updating framework, on both a controlled synthetic dataset
and the real-world IEEE-CIS Fraud Detection dataset.

## What's here

```
src/
  cold_start_sim.py       synthetic experiment: 100 warm + 30 cold cards,
                           Beta(5,2)-distributed true rates, 60-step simulation
  ieee_experiment.py       real-data experiment on IEEE-CIS Fraud Detection
  plot_bayesian_update.py  generates the Beta-distribution illustration of
                           belief narrowing as transactions arrive
notebooks/
  notebook_simulation.ipynb   walkthrough of the synthetic experiment
  notebook_ieee_cis.ipynb     walkthrough of the IEEE-CIS experiment
figures/                  generated plots used in the article
results/                  CSV outputs (per-run and aggregated metrics)
```

## The four placement strategies

- **Naive** — insert at the position implied by the point estimate `mu`. No
  uncertainty adjustment. This is the status quo in most systems.
- **LCB (Lower Confidence Bound)** — insert at `mu - k * sigma`. Higher
  uncertainty means a lower initial placement; the card rises as sigma
  shrinks with evidence.
- **Tiered** — route new cards to a fixed holding band for an initial
  window, then release to LCB-based placement.
- **Random** — uniformly random insertion, used as a baseline control.

## The metric that matters: Premature Top-K Rate (PTKR)

Standard ranking metrics like NDCG are blind to *who* caused disruption in a
review queue. PTKR flags a placement as premature only when all three hold:
the card is in the top-K, the model is still highly uncertain about it
(`sigma` above a threshold), and its true (oracle) position is outside the
top-K. In the synthetic experiment, Naive placement peaks at 28.2% PTKR;
LCB and Tiered reduce this to 0–3% at negligible cost to NDCG.

## Setup

```bash
pip install -r requirements.txt
```

## Running the synthetic experiment

```bash
python src/cold_start_sim.py
```

Outputs land in `results/` and `figures/`.

## Running the IEEE-CIS (real data) experiment

This experiment uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection),
released by Vesta Corporation for the 2019 Kaggle competition. The raw data
is **not included in this repo** — review the dataset's Kaggle competition
rules before downloading and reusing it.

1. Download `train_transaction.csv` from Kaggle.
2. Place it at `data/train_transaction.csv` (relative to the repo root), or
   point to it directly:

   ```bash
   export IEEE_CSV_PATH=/path/to/train_transaction.csv
   python src/ieee_experiment.py
   ```

## Generating the Bayesian-updating illustration

```bash
python src/plot_bayesian_update.py
```

## Citation

If you reference this work, please link back to the original article.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

The author has no affiliation with Kaggle or Vesta Corporation. The IEEE-CIS
dataset is subject to its own competition terms; this repo only provides
code, not the data itself.
