from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from figure_utils import find_fitting_logs, plot_skeleton, project_template_from_log, save_figure, set_equal_limits


def select_examples(k=3):
    candidates = []
    for path in find_fitting_logs(limit=1000):
        try:
            log = json.loads(Path(path).read_text(encoding="utf-8"))
            init = float(log["initial_errors"]["all"])
            final = float(log["final_errors"]["all"])
            improvement = init - final
            candidates.append((improvement, path))
        except Exception:
            continue
    candidates.sort(reverse=True, key=lambda x: x[0])
    selected = []
    seen_parent = set()
    for _, path in candidates:
        group = path.parent.parent.parent.name + "/" + path.parent.parent.name
        if group in seen_parent and len(seen_parent) < k:
            continue
        selected.append(path)
        seen_parent.add(group)
        if len(selected) == k:
            break
    return selected


def main():
    logs = select_examples(3)
    fig, axes = plt.subplots(len(logs), 3, figsize=(10.5, 3.1 * len(logs)))
    if len(logs) == 1:
        axes = axes[None, :]
    col_titles = ["Observed skeleton", "Initial projected template", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for r, log_path in enumerate(logs):
        observed, initial, refined, log = project_template_from_log(log_path)
        sample = log["sample_id"]
        init_err = log["initial_errors"]["all"]
        final_err = log["final_errors"]["all"]

        plot_skeleton(axes[r, 0], observed, color="#E45756", label="Observed")
        set_equal_limits(axes[r, 0], observed, initial)
        axes[r, 0].set_ylabel(f"{sample}\n{init_err:.3f}→{final_err:.3f}", fontsize=8)

        plot_skeleton(axes[r, 1], initial, color="#4C78A8", label="Initial")
        set_equal_limits(axes[r, 1], observed, initial)

        plot_skeleton(axes[r, 2], observed, color="#E45756", label="Observed", linewidth=2.6, joint_size=28)
        plot_skeleton(axes[r, 2], initial, color="#4C78A8", label="Initial", linewidth=1.8, alpha=0.75, linestyle="--")
        set_equal_limits(axes[r, 2], observed, initial)
        if r == 0:
            axes[r, 2].legend(loc="lower right", fontsize=8)

    fig.suptitle("Template Fitting Examples", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout()
    png, pdf = save_figure(fig, "figure_4_8_template_fitting_examples")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
