from __future__ import annotations

import matplotlib.pyplot as plt

from figure_utils import OUT_DIR, load_json, save_figure


def pct(x):
    return 100.0 * float(x)


def main():
    raw_base = load_json("model/results/paper_controlled_raw_baseline_6cls/summary.json")
    raw_mv = load_json("model/results/paper_controlled_raw_mv_6cls/summary.json")
    fitted_mv = load_json("model/results/fitted_anchor_multiview_6cls/summary.json")
    
    labels = [
        "Raw\nbaseline",
        "Raw +\n8-view MV",
        "Fitted + MV\n(test raw)",
        "Fitted + MV\noracle",
    ]
    values = [
        pct(raw_base["evaluations"]["droh_raw"]["accuracy"]),
        pct(raw_mv["evaluations"]["droh_raw"]["accuracy"]),
        pct(fitted_mv["evaluations"]["droh_raw"]["accuracy"]),
        pct(fitted_mv["evaluations"]["droh_fitted_oracle"]["accuracy"]),
    ]
    colors = [
        "#4C78A8", "#4C78A8", "#F58518", "#E45756",
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylabel("DrOh accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Main DrOh Accuracy Comparison", pad=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    fig.tight_layout()
    png, pdf = save_figure(fig, "figure_4_4_main_droh_accuracy_comparison")
    print(png)
    print(pdf)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
