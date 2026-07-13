"""Aggregate post-submission experiments into defense-ready tables and figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "results/defense_experiments_6cls"
OUT_ROOT = INPUT_ROOT / "summary"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PATTERN = re.compile(
    r"(?P<model>gru|mlp)_(?P<normalization>wrist_middle|palm_robust)_"
    r"views(?P<num_views>\d+)_lambda(?P<weight>\d+p\d+)_seed(?P<seed>\d+)"
)


def load_runs():
    rows = []
    for path in sorted(INPUT_ROOT.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "tag" not in data:
            continue
        match = PATTERN.fullmatch(data["tag"])
        if not match or data.get("status") != "complete":
            continue
        fields = match.groupdict()
        row = {
            "tag": data["tag"],
            "model": fields["model"],
            "normalization": fields["normalization"],
            "num_views": int(fields["num_views"]),
            "lambda_weight": float(fields["weight"].replace("p", ".")),
            "seed": int(fields["seed"]),
            "best_epoch": data["best_epoch"],
            "salux_val_accuracy": data["best_salux_val_accuracy"],
            "elapsed_seconds": data["elapsed_seconds"],
            "directory": str(path.parent),
        }
        for dataset, metrics in data["evaluations"].items():
            for metric, value in metrics.items():
                row[f"{dataset}_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(runs):
    group_cols = ["model", "normalization", "num_views", "lambda_weight"]
    metrics = [
        "salux_raw_accuracy", "salux_raw_macro_f1",
        "droh_raw_accuracy", "droh_raw_balanced_accuracy", "droh_raw_macro_f1",
    ]
    output = []
    for keys, frame in runs.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["num_seeds"] = len(frame)
        row["seeds"] = ",".join(str(value) for value in sorted(frame.seed))
        for metric in metrics:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1) if len(frame) > 1 else np.nan
        output.append(row)
    return pd.DataFrame(output)


def prediction_path(runs, model, normalization, views, weight, seed):
    selected = runs[
        (runs.model == model)
        & (runs.normalization == normalization)
        & (runs.num_views == views)
        & np.isclose(runs.lambda_weight, weight)
        & (runs.seed == seed)
    ]
    if len(selected) != 1:
        return None
    return Path(selected.iloc[0].directory) / "predictions_droh_raw.csv"


def paired_comparison(left_path, right_path, seed, left_name, right_name):
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    merged = left[["sample_id", "true_class", "correct"]].merge(
        right[["sample_id", "true_class", "correct"]],
        on=["sample_id", "true_class"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    left_correct = merged.correct_left.to_numpy(dtype=bool)
    right_correct = merged.correct_right.to_numpy(dtype=bool)
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    p_value = float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue) if discordant else 1.0

    rng = np.random.default_rng(20260706 + seed)
    differences = np.empty(10000, dtype=np.float32)
    for index in range(len(differences)):
        sample = rng.integers(0, len(merged), len(merged))
        differences[index] = right_correct[sample].mean() - left_correct[sample].mean()
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "seed": seed,
        "left": left_name,
        "right": right_name,
        "rows": len(merged),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "difference_percentage_points": float((right_correct.mean() - left_correct.mean()) * 100),
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "mcnemar_exact_p": p_value,
        "paired_bootstrap_95ci_pp": [float(lower * 100), float(upper * 100)],
    }


def build_paired_tests(runs):
    rows = []
    comparisons = [
        ((0, 0.0, "Raw baseline"), (8, 0.0, "MV CE-only")),
        ((0, 0.0, "Raw baseline"), (8, 0.3, "MV + consistency")),
        ((8, 0.0, "MV CE-only"), (8, 0.3, "MV + consistency")),
    ]
    for seed in [0, 1, 42]:
        for left, right in comparisons:
            left_path = prediction_path(runs, "gru", "wrist_middle", left[0], left[1], seed)
            right_path = prediction_path(runs, "gru", "wrist_middle", right[0], right[1], seed)
            if left_path and right_path and left_path.exists() and right_path.exists():
                rows.append(paired_comparison(left_path, right_path, seed, left[2], right[2]))
    return rows


def core_figure(aggregate):
    specs = [(0, 0.0, "Raw baseline"), (8, 0.0, "MV CE-only"), (8, 0.3, "MV + consistency")]
    selected = []
    for views, weight, label in specs:
        row = aggregate[
            (aggregate.model == "gru")
            & (aggregate.normalization == "wrist_middle")
            & (aggregate.num_views == views)
            & np.isclose(aggregate.lambda_weight, weight)
        ]
        if len(row) == 1:
            selected.append((label, row.iloc[0]))
    if not selected:
        return
    labels = [item[0] for item in selected]
    accuracy = np.array([item[1].droh_raw_accuracy_mean for item in selected]) * 100
    accuracy_std = np.array([item[1].droh_raw_accuracy_std for item in selected]) * 100
    macro_f1 = np.array([item[1].droh_raw_macro_f1_mean for item in selected]) * 100
    macro_f1_std = np.array([item[1].droh_raw_macro_f1_std for item in selected]) * 100
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - width / 2, accuracy, width, yerr=accuracy_std, capsize=4, label="Accuracy", color="#4C78A8")
    ax.bar(x + width / 2, macro_f1, width, yerr=macro_f1_std, capsize=4, label="Macro F1", color="#F58518")
    ax.set_xticks(x, labels)
    ax.set_ylabel("DrOh score (%)")
    ax.set_ylim(max(0, min(np.r_[accuracy, macro_f1]) - 8), min(100, max(np.r_[accuracy, macro_f1]) + 5))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "core_ablation_multiseed.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "core_ablation_multiseed.pdf", bbox_inches="tight")
    plt.close(fig)


def view_count_figure(aggregate):
    frame = aggregate[(aggregate.model == "gru") & (aggregate.normalization == "wrist_middle")]
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    for weight, label, color in [(0.0, "CE-only", "#4C78A8"), (0.3, "CE + consistency", "#F58518")]:
        selected = frame[np.isclose(frame.lambda_weight, weight)].sort_values("num_views")
        if not len(selected):
            continue
        ax.errorbar(
            selected.num_views,
            selected.droh_raw_accuracy_mean * 100,
            yerr=selected.droh_raw_accuracy_std * 100,
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=label,
            color=color,
        )
    ax.set_xticks([0, 2, 4, 8])
    ax.set_xlabel("Number of synthetic training views")
    ax.set_ylabel("DrOh accuracy (%)")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "view_count_ablation_multiseed.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "view_count_ablation_multiseed.pdf", bbox_inches="tight")
    plt.close(fig)


def comparison_figure(aggregate):
    specs = [
        ("GRU raw", "gru", "wrist_middle", 0, 0.0),
        ("GRU MV CE", "gru", "wrist_middle", 8, 0.0),
        ("GRU MV cons.", "gru", "wrist_middle", 8, 0.3),
        ("Palm raw", "gru", "palm_robust", 0, 0.0),
        ("Palm MV CE", "gru", "palm_robust", 8, 0.0),
        ("Palm MV cons.", "gru", "palm_robust", 8, 0.3),
        ("MLP raw", "mlp", "wrist_middle", 0, 0.0),
        ("MLP MV CE", "mlp", "wrist_middle", 8, 0.0),
        ("MLP MV cons.", "mlp", "wrist_middle", 8, 0.3),
    ]
    rows = []
    for label, model, normalization, views, weight in specs:
        selected = aggregate[
            (aggregate.model == model)
            & (aggregate.normalization == normalization)
            & (aggregate.num_views == views)
            & np.isclose(aggregate.lambda_weight, weight)
        ]
        if len(selected) == 1:
            rows.append((label, selected.iloc[0]))
    if not rows:
        return
    labels = [row[0] for row in rows]
    means = np.array([row[1].droh_raw_accuracy_mean for row in rows]) * 100
    errors = np.array([row[1].droh_raw_accuracy_std for row in rows]) * 100
    colors = ["#4C78A8" if label.startswith("GRU") else "#54A24B" if label.startswith("Palm") else "#B279A2" for label in labels]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    bars = ax.bar(np.arange(len(labels)), means, yerr=errors, capsize=3, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    ax.set_ylabel("DrOh accuracy (%)")
    ax.set_ylim(max(0, means.min() - 8), min(100, means.max() + 6))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "extended_ablation_multiseed.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "extended_ablation_multiseed.pdf", bbox_inches="tight")
    plt.close(fig)


def ensemble_table():
    path = INPUT_ROOT / "ensembles/summary.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, result in data.items():
        rows.append({"ensemble": name, **result["salux_raw"], **{f"droh_{key}": value for key, value in result["droh_raw"].items()}})
    return pd.DataFrame(rows)


def ensemble_figures(ensembles):
    if not len(ensembles):
        return
    labels_map = {
        "raw_baseline_3seed": "Raw ensemble",
        "mv_ce_only_3seed": "MV CE-only ensemble",
        "mv_consistency_3seed": "MV consistency ensemble",
    }
    ordered = ensembles.set_index("ensemble").loc[list(labels_map)]
    labels = [labels_map[index] for index in ordered.index]
    accuracy = ordered.droh_accuracy.to_numpy() * 100
    macro_f1 = ordered.droh_macro_f1.to_numpy() * 100
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    bars_a = ax.bar(x - width / 2, accuracy, width, color="#4C78A8", label="Accuracy")
    bars_f = ax.bar(x + width / 2, macro_f1, width, color="#F58518", label="Macro F1")
    ax.set_xticks(x, labels)
    ax.set_ylabel("DrOh score (%)")
    ax.set_ylim(min(np.r_[accuracy, macro_f1]) - 6, max(np.r_[accuracy, macro_f1]) + 4)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    for bars in [bars_a, bars_f]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "ensemble_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "ensemble_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    cm_root = INPUT_ROOT / "ensembles"
    matrix_specs = [
        ("raw_baseline_3seed", "(a) Raw three-seed ensemble"),
        ("mv_ce_only_3seed", "(b) MV CE-only three-seed ensemble"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3))
    image = None
    class_labels = ["OK", "Paper", "Rock", "Scissors", "The-finger", "Thumb"]
    for ax, (name, title) in zip(axes, matrix_specs):
        matrix = np.load(cm_root / f"confmat_{name}_droh_raw.npy").astype(float)
        matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0) * 100
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        ticks = np.arange(len(class_labels))
        ax.set_xticks(ticks, class_labels, rotation=35, ha="right")
        ax.set_yticks(ticks, class_labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for row in range(6):
            for column in range(6):
                value = matrix[row, column]
                ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value >= 50 else "#222222")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.18, top=0.90, wspace=0.30)
    colorbar_ax = fig.add_axes([0.925, 0.19, 0.018, 0.675])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Percentage (%)")
    fig.savefig(OUT_ROOT / "ensemble_confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "ensemble_confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(runs, aggregate, paired, ensembles):
    lines = [
        "# Post-submission defense experiments", "",
        "DrOh was used only for final external evaluation. Checkpoints were selected using Salux validation accuracy.", "",
        "## Aggregate results", "",
        aggregate.to_markdown(index=False, floatfmt=".6f") if len(aggregate) else "No complete runs.", "",
        "## Paired DrOh comparisons", "",
    ]
    if paired:
        lines.append(pd.DataFrame(paired).to_markdown(index=False, floatfmt=".6f"))
    else:
        lines.append("Core paired comparisons are not complete yet.")
    lines.extend(["", "## Fixed three-seed ensembles", ""])
    lines.append(ensembles.to_markdown(index=False, floatfmt=".6f") if len(ensembles) else "No ensemble results.")
    lines.extend(["", "## Individual runs", "", runs.to_markdown(index=False, floatfmt=".6f")])
    (OUT_ROOT / "DEFENSE_EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    runs = load_runs()
    if not len(runs):
        raise SystemExit(f"No completed experiments found under {INPUT_ROOT}")
    runs.to_csv(OUT_ROOT / "individual_runs.csv", index=False)
    aggregate = aggregate_runs(runs)
    aggregate.to_csv(OUT_ROOT / "aggregate_results.csv", index=False)
    paired = build_paired_tests(runs)
    (OUT_ROOT / "paired_significance.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    core_figure(aggregate)
    view_count_figure(aggregate)
    comparison_figure(aggregate)
    ensembles = ensemble_table()
    ensembles.to_csv(OUT_ROOT / "ensemble_results.csv", index=False)
    ensemble_figures(ensembles)
    write_markdown(runs, aggregate, paired, ensembles)
    print(OUT_ROOT / "DEFENSE_EXPERIMENT_REPORT.md")
    print(OUT_ROOT / "aggregate_results.csv")
    print(OUT_ROOT / "paired_significance.json")


if __name__ == "__main__":
    main()
