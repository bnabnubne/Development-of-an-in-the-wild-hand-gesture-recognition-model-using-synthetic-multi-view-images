"""Aggregate clean 2x2 runs and evaluate held-out synthetic camera angles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_clean_viewpoint_factorial_6cls as exp


UNSEEN_ANGLES = [-30, 15, 75, 105, 150]
OUT = exp.OUT_ROOT


def predict(model, values):
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 256):
            x = torch.from_numpy(values[start:start + 256]).to(exp.DEVICE)
            probabilities.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(probabilities)


def evaluate_unseen(run_dir, config, seed, test):
    canonical = config.startswith("canonical")
    model = exp.GRU().to(exp.DEVICE)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=exp.DEVICE, weights_only=False)["model_state_dict"])
    truth = np.asarray([exp.LABELS[exp.label(value)] for value in test.action])
    results = {}
    for angle in UNSEEN_ANGLES:
        values = []
        for row in test.itertuples(index=False):
            anchor = exp.minimal_preprocess(np.load(row.raw_path), row.handedness)
            view = exp.camera_view(anchor, angle)
            values.append(exp.palm_canonicalize(view) if canonical else view)
        probs = predict(model, np.stack(values).astype(np.float32))
        pred = probs.argmax(1)
        results[str(angle)] = exp.metrics(truth, pred)
    payload = {"config": config, "seed": seed, "unseen_angles": UNSEEN_ANGLES, "results": results}
    (run_dir / "unseen_angle_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    salux, _ = exp.build_manifests()
    test = salux[salux.split == "test"].copy().reset_index(drop=True)
    records, angle_records = [], []
    for config in exp.CONFIGS:
        for seed in exp.SEEDS:
            run_dir = OUT / f"{config}_seed{seed}"
            summary = json.loads((run_dir / "summary.json").read_text())
            unseen = evaluate_unseen(run_dir, config, seed, test)
            records.append({
                "config": config, "seed": seed,
                "salux_accuracy": summary["evaluations"]["salux"]["accuracy"],
                "droh_accuracy": summary["evaluations"]["droh"]["accuracy"],
                "droh_macro_f1": summary["evaluations"]["droh"]["macro_f1"],
            })
            for angle, metrics in unseen["results"].items():
                angle_records.append({"config": config, "seed": seed, "angle": int(angle), **metrics})
    runs = pd.DataFrame(records)
    angles = pd.DataFrame(angle_records)
    aggregate = runs.groupby("config").agg(
        salux_accuracy_mean=("salux_accuracy", "mean"), salux_accuracy_std=("salux_accuracy", "std"),
        droh_accuracy_mean=("droh_accuracy", "mean"), droh_accuracy_std=("droh_accuracy", "std"),
        droh_macro_f1_mean=("droh_macro_f1", "mean"), droh_macro_f1_std=("droh_macro_f1", "std"),
    ).reset_index()
    angle_aggregate = angles.groupby(["config", "angle"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    runs.to_csv(OUT / "all_runs.csv", index=False)
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    angle_aggregate.to_csv(OUT / "unseen_angle_aggregate.csv", index=False)

    ensemble = {}
    probability_columns = [f"prob_{name}" for name in exp.CLASSES]
    for config in exp.CONFIGS:
        frames = [pd.read_csv(OUT / f"{config}_seed{seed}/predictions_droh.csv") for seed in exp.SEEDS]
        probabilities = np.mean([frame[probability_columns].to_numpy() for frame in frames], axis=0)
        truth = frames[0].true_class.map(exp.LABELS).to_numpy()
        prediction = probabilities.argmax(1)
        ensemble[config] = {"truth": truth, "prediction": prediction, "correct": truth == prediction, "frame": frames[0]}

    rng = np.random.default_rng(42)
    paired = {}
    for name, first, second in [
        ("camera_mv_minus_single", "camera_single", "camera_mv"),
        ("canonical_mv_minus_single", "canonical_single", "canonical_mv"),
    ]:
        a, b = ensemble[first]["correct"], ensemble[second]["correct"]
        first_only, second_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
        indices = rng.integers(0, len(a), size=(10000, len(a)))
        differences = (b[indices].mean(1) - a[indices].mean(1)) * 100
        paired[name] = {
            "first": first, "second": second,
            "first_accuracy": float(a.mean()), "second_accuracy": float(b.mean()),
            "difference_percentage_points": float(100 * (b.mean() - a.mean())),
            "first_only_correct": first_only, "second_only_correct": second_only,
            "mcnemar_exact_p": float(binomtest(min(first_only, second_only), first_only + second_only, 0.5).pvalue),
            "bootstrap_95_ci_percentage_points": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        }
    (OUT / "paired_statistics.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")

    bins = [-0.01, 15, 30, 45, 60, 90.01]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-90"]
    angle_frame = ensemble["camera_single"]["frame"][["sample_id", "raw_palm_angle"]].copy()
    angle_frame["angle_bin"] = pd.cut(angle_frame.raw_palm_angle, bins, labels=labels)
    for config in exp.CONFIGS:
        angle_frame[config] = ensemble[config]["correct"]
    angle_rows = []
    for label_name, group in angle_frame.groupby("angle_bin", observed=False):
        row = {"angle_bin": str(label_name), "rows": len(group)}
        for config in exp.CONFIGS:
            row[f"{config}_accuracy"] = float(group[config].mean())
        angle_rows.append(row)
    pd.DataFrame(angle_rows).to_csv(OUT / "droh_angle_bin_ensemble.csv", index=False)

    def pct(value): return f"{100*value:.2f}"
    lines = [
        "# Clean Viewpoint Factorial 6-Class Results", "",
        "All values are mean ± sample standard deviation across seeds 0, 1, and 42.", "",
        "| Config | Salux test acc | DrOh acc | DrOh macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.config} | {pct(row.salux_accuracy_mean)} ± {pct(row.salux_accuracy_std)} | {pct(row.droh_accuracy_mean)} ± {pct(row.droh_accuracy_std)} | {pct(row.droh_macro_f1_mean)} ± {pct(row.droh_macro_f1_std)} |")
    lines += ["", "## Unseen-angle Salux accuracy", "", "| Config | -30° | 15° | 75° | 105° | 150° |", "|---|---:|---:|---:|---:|---:|"]
    for config in exp.CONFIGS:
        subset = angle_aggregate[angle_aggregate.config == config].set_index("angle")
        values = [f"{pct(subset.loc[a].accuracy_mean)} ± {pct(subset.loc[a].accuracy_std)}" for a in UNSEEN_ANGLES]
        lines.append("| " + config + " | " + " | ".join(values) + " |")
    lines += [
        "", "## Paired 3-seed probability ensembles on DrOh", "",
        f"- Camera-coordinate MV − single: **{paired['camera_mv_minus_single']['difference_percentage_points']:+.2f} pp**, exact McNemar p = {paired['camera_mv_minus_single']['mcnemar_exact_p']:.3g}, bootstrap 95% CI [{paired['camera_mv_minus_single']['bootstrap_95_ci_percentage_points'][0]:.2f}, {paired['camera_mv_minus_single']['bootstrap_95_ci_percentage_points'][1]:.2f}] pp.",
        f"- Canonical MV − single: **{paired['canonical_mv_minus_single']['difference_percentage_points']:+.2f} pp**, exact McNemar p = {paired['canonical_mv_minus_single']['mcnemar_exact_p']:.3g}, bootstrap 95% CI [{paired['canonical_mv_minus_single']['bootstrap_95_ci_percentage_points'][0]:.2f}, {paired['canonical_mv_minus_single']['bootstrap_95_ci_percentage_points'][1]:.2f}] pp.",
        "", "## Protocol", "",
        "- Same 4,582 Salux train anchors and fixed train/val/test split.",
        "- Same GRU, CE-only loss, batch size, optimizer, early stopping, and optimizer-step budget.",
        "- MV samples exactly one of identity + Blender8 per anchor/update.",
        "- Shuffle RNG is separated from view-selection RNG.",
        "- DrOh is single-view evaluation only: no fitting, no TTA, no checkpoint selection.",
        "- `camera_*` keeps camera-relative palm orientation; `canonical_*` applies palm-axis alignment.",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
