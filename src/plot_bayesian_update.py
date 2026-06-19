import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta

# ── Illustrative example: one cold card's belief evolving ──────────
# True fraud rate (hidden from the model) = 0.15
# Prior is deliberately wide and centered lower than the truth.

TRUE_RATE = 0.15

stages = [
    {"label": "Prior\n(0 transactions seen)",      "a": 3,  "b": 27, "color": "#95A5A6"},
    {"label": "After 10 transactions\n(2 fraud)",   "a": 5,  "b": 35, "color": "#3498DB"},
    {"label": "After 40 transactions\n(7 fraud)",   "a": 10, "b": 60, "color": "#E74C3C"},
]

x = np.linspace(0, 0.4, 1000)

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

for s in stages:
    y = beta.pdf(x, s["a"], s["b"])
    mu = s["a"] / (s["a"] + s["b"])
    sigma = np.sqrt((s["a"]*s["b"]) / ((s["a"]+s["b"])**2 * (s["a"]+s["b"]+1)))
    ax.plot(x, y, color=s["color"], linewidth=2.8,
            label=f'{s["label"]}\nμ={mu:.3f}, σ={sigma:.3f}')
    ax.fill_between(x, y, alpha=0.15, color=s["color"])

ax.axvline(TRUE_RATE, color='#2C3E50', linestyle='--', linewidth=1.6, alpha=0.8)
ax.text(TRUE_RATE + 0.006, ax.get_ylim()[1]*0.92, "true fraud rate\n(unknown to model)",
        fontsize=9, color='#2C3E50', fontweight='bold')

ax.set_xlabel("Estimated fraud rate", fontsize=12, fontweight='bold')
ax.set_ylabel("Belief density", fontsize=12, fontweight='bold')
ax.set_title("Bayesian Updating: Belief Sharpens as Transactions Arrive",
             fontsize=14, fontweight='bold', pad=14)
ax.legend(fontsize=9.5, loc='upper right', frameon=True, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25, linestyle=':')
ax.set_ylim(bottom=0)

fig.text(0.5, 0.01,
         "Wider curve = more uncertain.  As transactions accumulate, the curve narrows and shifts toward the true fraud rate.",
         ha='center', fontsize=9.5, color='#555', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("bayesian_update_explainer.png", dpi=200, bbox_inches='tight')
print("saved")
