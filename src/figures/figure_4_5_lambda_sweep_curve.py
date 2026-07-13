from __future__ import annotations

import json

import matplotlib.pyplot as plt

from figure_utils import load_json, save_figure


def main():
    data = load_json("model/results/fitted_anchor_multiview_lambda_sweep_6cls_summary.json")
    rows = sorted(data["rows"], key=lambda r: r["lambda"])
    lambdas = [r["lambda"] for r in rows]
    droh_raw = [100 * r["droh_raw"] for r in rows]
    droh_oracle = [100 * r["droh_fitted_oracle"] for r in rows]
    salux_fitted = [100 * r["salux_fitted"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(lambdas, droh_raw, marker="o", linewidth=2.4, label="DrOh raw")
    ax.plot(lambdas, droh_oracle, marker="s", linewidth=2.4, label="DrOh fitted oracle")
    ax.plot(lambdas, salux_fitted, marker="^", linewidth=1.8, linestyle="--", label="Salux fitted")
    ax.set_xlabel(r"Consistency weight $\lambda$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(r"Effect of $\lambda$ on Fitted Multi-view Training", pad=12)
    ax.set_xticks(lambdas)
    ax.set_ylim(20, 103)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    for x, y in zip(lambdas, droh_raw):
        ax.text(x, y + 1.2, f"{y:.1f}", ha="center", fontsize=8)
    for x, y in zip(lambdas, droh_oracle):
        ax.text(x, y + 1.2, f"{y:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    png, pdf = save_figure(fig, "figure_4_5_lambda_sweep_curve")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
