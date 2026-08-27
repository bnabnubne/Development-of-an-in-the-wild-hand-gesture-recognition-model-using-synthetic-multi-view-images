
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent / "results/defense_experiments_6cls"
OUT = ROOT / "raw_lambda_sweep"
CLASSES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
SEEDS = [0, 1, 42]
WEIGHTS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def tag(weight, seed):
    return f"gru_wrist_middle_views8_lambda{str(weight).replace('.', 'p')}_seed{seed}"


def paired(left, right, rng):
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    left_only = int(np.sum(left & ~right)); right_only = int(np.sum(~left & right))
    sample = rng.integers(0, len(left), size=(10000, len(left)))
    delta = (right[sample].mean(1) - left[sample].mean(1)) * 100
    return {
        "left_accuracy": float(left.mean()), "right_accuracy": float(right.mean()),
        "difference_pp": float(100 * (right.mean() - left.mean())),
        "left_only_correct": left_only, "right_only_correct": right_only,
        "mcnemar_exact_p": float(binomtest(min(left_only, right_only), left_only + right_only, 0.5).pvalue),
        "bootstrap_95ci_pp": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ensembles = [], {}
    probability_columns = [f"prob_{name}" for name in CLASSES]
    for weight in WEIGHTS:
        prediction_frames = []
        for seed in SEEDS:
            run = ROOT / tag(weight, seed)
            required = [run / "best.pt", run / "summary.json", run / "history.json", run / "predictions_droh_raw.csv"]
            missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
            if missing:
                raise RuntimeError(f"Incomplete lambda={weight}, seed={seed}: {missing}")
            summary = json.loads((run / "summary.json").read_text())
            history = json.loads((run / "history.json").read_text())
            prediction = pd.read_csv(run / "predictions_droh_raw.csv")
            if len(prediction) != 675 or prediction[["action", "sample_id"]].duplicated().any():
                raise RuntimeError(f"Invalid DrOh predictions: lambda={weight}, seed={seed}")
            best_row = history[summary["best_epoch"] - 1]
            rows.append({
                "lambda": weight, "seed": seed, "best_epoch": summary["best_epoch"],
                "salux_val_accuracy": summary["best_salux_val_accuracy"],
                "salux_test_accuracy": summary["evaluations"]["salux_raw"]["accuracy"],
                "droh_accuracy": summary["evaluations"]["droh_raw"]["accuracy"],
                "droh_macro_f1": summary["evaluations"]["droh_raw"]["macro_f1"],
                "best_epoch_cls_loss": best_row["classification_loss"],
                "best_epoch_consistency_loss": best_row["consistency_loss"],
                "droh_correct": int(prediction.correct.sum()), "droh_rows": len(prediction),
            })
            prediction_frames.append(prediction)
        reference = prediction_frames[0]
        for frame in prediction_frames[1:]:
            if not frame.sample_id.equals(reference.sample_id) or not frame.true_class.equals(reference.true_class):
                raise RuntimeError(f"Misaligned prediction rows for lambda={weight}")
        probability = np.mean([frame[probability_columns].to_numpy() for frame in prediction_frames], axis=0)
        prediction_index = probability.argmax(1)
        truth = reference.true_class.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
        correct = truth == prediction_index
        ensembles[weight] = correct
        ensembles_row = {
            "lambda": weight, "correct": int(correct.sum()), "rows": len(correct),
            "accuracy": accuracy_score(truth, prediction_index),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction_index),
            "macro_f1": f1_score(truth, prediction_index, average="macro"),
        }
        output = reference[["action", "sample_id", "true_class"]].copy()
        output["predicted_class"] = np.asarray(CLASSES)[prediction_index]
        output["correct"] = correct
        for index, class_name in enumerate(CLASSES):
            output[f"prob_{class_name}"] = probability[:, index]
        output.to_csv(OUT / f"ensemble_lambda_{str(weight).replace('.', 'p')}_droh.csv", index=False)
        ensembles_row["prediction_file"] = str(OUT / f"ensemble_lambda_{str(weight).replace('.', 'p')}_droh.csv")
        ensembles.setdefault("rows", []).append(ensembles_row)

    runs = pd.DataFrame(rows)
    aggregate = runs.groupby("lambda").agg(
        salux_val_mean=("salux_val_accuracy", "mean"), salux_val_std=("salux_val_accuracy", "std"),
        salux_test_mean=("salux_test_accuracy", "mean"), salux_test_std=("salux_test_accuracy", "std"),
        droh_accuracy_mean=("droh_accuracy", "mean"), droh_accuracy_std=("droh_accuracy", "std"),
        droh_macro_f1_mean=("droh_macro_f1", "mean"), droh_macro_f1_std=("droh_macro_f1", "std"),
        consistency_loss_mean=("best_epoch_consistency_loss", "mean"),
    ).reset_index()
    ensemble_frame = pd.DataFrame(ensembles.pop("rows"))
    selected_weight = float(aggregate.loc[aggregate.salux_val_mean.idxmax(), "lambda"])
    descriptive_droh_best = float(aggregate.loc[aggregate.droh_accuracy_mean.idxmax(), "lambda"])
    descriptive_ensemble_best = float(ensemble_frame.loc[ensemble_frame.accuracy.idxmax(), "lambda"])
    rng = np.random.default_rng(42)
    comparisons = {
        "selected_vs_ce_only": paired(ensembles[0.0], ensembles[selected_weight], rng),
        "lambda_0p3_vs_ce_only": paired(ensembles[0.0], ensembles[0.3], rng),
    }
    protocol = {
        "weights": WEIGHTS, "seeds": SEEDS, "selection_rule": "maximum mean Salux validation anchor accuracy",
        "selected_lambda": selected_weight,
        "descriptive_best_mean_droh_lambda_not_for_selection": descriptive_droh_best,
        "descriptive_best_ensemble_droh_lambda_not_for_selection": descriptive_ensemble_best,
        "accuracy_definition": "number correct / 675 DrOh samples",
        "std_definition": "sample standard deviation across seeds 0,1,42 (ddof=1)",
        "ensemble_definition": "equal arithmetic mean of three class-probability vectors, then argmax",
        "comparisons": comparisons,
    }
    runs.to_csv(OUT / "all_runs.csv", index=False)
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    ensemble_frame.to_csv(OUT / "ensembles.csv", index=False)
    (OUT / "protocol_and_selection.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    lines = [
        "Normalized Blender8 consistency lambda sweep", "",
        "All mean +/- SD values use three seeds (0, 1, 42); SD is sample SD (ddof=1).", "",
    ]
    for _, row in aggregate.iterrows():
        weight = float(row["lambda"])
        ens = ensemble_frame[ensemble_frame["lambda"] == weight].iloc[0]
        lines.append(f"lambda {weight:g}: Salux val {100*row['salux_val_mean']:.2f} +/- {100*row['salux_val_std']:.2f}, Salux test {100*row['salux_test_mean']:.2f} +/- {100*row['salux_test_std']:.2f}, DrOh {100*row['droh_accuracy_mean']:.2f} +/- {100*row['droh_accuracy_std']:.2f}, macro F1 {100*row['droh_macro_f1_mean']:.2f} +/- {100*row['droh_macro_f1_std']:.2f}, ensemble {100*ens.accuracy:.2f} ({int(ens.correct)}/675)")
    lines += [
        "", f"Validation-selected lambda: {selected_weight:g}.",
        f"Descriptive best mean DrOh lambda (not selection): {descriptive_droh_best:g}.",
        f"Descriptive best ensemble DrOh lambda (not selection): {descriptive_ensemble_best:g}.",
        "", "The selected model must be reported using the validation-selected lambda; DrOh sweep maxima are descriptive only.",
    ]
    (OUT / "final_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
