from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "results/defense_experiments_6cls/raw_lambda_sweep"
a = pd.read_csv(ROOT / "aggregate.csv")
e = pd.read_csv(ROOT / "ensembles.csv")
v = pd.read_csv(ROOT / "salux_val_view_metrics_aggregate.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].errorbar(a["lambda"], 100*a.droh_accuracy_mean, yerr=100*a.droh_accuracy_std,
                 marker="o", capsize=3, linewidth=2, label="Single models: mean ± SD")
axes[0].plot(e["lambda"], 100*e.accuracy, marker="s", linewidth=2, label="3-seed ensemble")
axes[0].axvline(0.3, color="#777777", linestyle="--", linewidth=1.2, label="All-view val selected λ=0.3")
axes[0].set(xlabel="Consistency weight λ", ylabel="DrOh accuracy (%)", title="External DrOh performance")
axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)

axes[1].plot(v["lambda"], 100*v.all9_mean, marker="o", linewidth=2, label="Anchor + 8-view accuracy")
axes[1].plot(v["lambda"], 100*v.anchor_mean, marker="s", linewidth=1.7, label="Anchor accuracy")
axes[1].axvline(0.3, color="#777777", linestyle="--", linewidth=1.2)
axes[1].set(xlabel="Consistency weight λ", ylabel="Salux validation accuracy (%)", title="Validation selection")
axes[1].grid(alpha=.25); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "raw_lambda_sweep.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "raw_lambda_sweep.pdf", bbox_inches="tight")
print(ROOT / "raw_lambda_sweep.png")
