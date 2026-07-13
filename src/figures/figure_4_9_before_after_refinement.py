from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from figure_utils import find_fitting_logs, plot_skeleton, project_template_from_log, save_figure, set_equal_limits


def select_best_example():
    best = None
    for path in find_fitting_logs(limit=1500):
        try:
            log = json.loads(Path(path).read_text(encoding="utf-8"))
            init = float(log["initial_errors"]["all"])
            final = float(log["final_errors"]["all"])
            tips_gain = float(log["initial_errors"]["tips"]) - float(log["final_errors"]["tips"])
            score = (init - final) + 0.5 * tips_gain
            if best is None or score > best[0]:
                best = (score, path)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("No fitting log found")
    return best[1]


def main():
    log_path = select_best_example()
    observed, initial, refined, log = project_template_from_log(log_path)
    sample = log["sample_id"]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7))
    titles = ["Observed skeleton", "Initial fitting", "Refined fitting", "Overlay comparison"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    plot_skeleton(axes[0], observed, color="#E45756", linewidth=2.6)
    plot_skeleton(axes[1], initial, color="#4C78A8", linewidth=2.2)
    plot_skeleton(axes[2], refined, color="#54A24B", linewidth=2.2)
    plot_skeleton(axes[3], observed, color="#E45756", label="Observed", linewidth=2.5, joint_size=28)
    plot_skeleton(axes[3], initial, color="#4C78A8", label="Initial", linewidth=1.7, alpha=0.65, linestyle="--")
    plot_skeleton(axes[3], refined, color="#54A24B", label="Refined", linewidth=1.9, alpha=0.9)
    axes[3].legend(loc="lower right", fontsize=8)

    for ax in axes:
        set_equal_limits(ax, observed, initial, refined)

    init_all = log["initial_errors"]["all"]
    final_all = log["final_errors"]["all"]
    init_tip = log["initial_errors"]["tips"]
    final_tip = log["final_errors"]["tips"]
    fig.suptitle(
        f"Before/After Hierarchical Refinement — {sample} "
        f"(all {init_all:.3f}→{final_all:.3f}, tips {init_tip:.3f}→{final_tip:.3f})",
        y=0.99,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    png, pdf = save_figure(fig, "figure_4_9_before_after_hierarchical_refinement")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
