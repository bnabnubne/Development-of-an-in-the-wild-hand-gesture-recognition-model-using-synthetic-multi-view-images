"""Build the locked five-seed baseline/final comparison used by webcam demos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT / "results/defense_experiments_6cls"
FINAL_ROOT = ROOT / "results/consistency_finetune_6cls"
OUT = ROOT / "results/webcam_final_comparison_6cls"
SEEDS = [0, 1, 7, 21, 42]
CLASSES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
PROBABILITY_COLUMNS = [f"prob_{name}" for name in CLASSES]


def ensemble(paths, output_name):
    frames = [pd.read_csv(path) for path in paths]
    key_columns = ["action", "sample_id", "true_class"]
    reference = frames[0][key_columns]
    if not all(len(frame) == 675 and frame[key_columns].equals(reference) for frame in frames):
        raise ValueError(f"Prediction alignment failed for {output_name}")
    probabilities = np.mean([frame[PROBABILITY_COLUMNS].to_numpy() for frame in frames], axis=0)
    truth = frames[0].true_class.to_numpy()
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    result = reference.copy()
    result["predicted_class"] = predicted
    result["correct"] = predicted == truth
    result["confidence"] = probabilities.max(axis=1)
    for index, class_name in enumerate(CLASSES):
        result[f"prob_{class_name}"] = probabilities[:, index]
    result.to_csv(OUT / output_name, index=False)
    return result, {
        "correct": int(result.correct.sum()),
        "rows": len(result),
        "accuracy": float(result.correct.mean()),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_paths = [
        BASE_ROOT / f"gru_wrist_middle_views0_lambda0p0_seed{seed}/predictions_droh_raw.csv"
        for seed in SEEDS
    ]
    final_paths = [
        FINAL_ROOT / f"lambda0p3_lr1em4_seed{seed}/predictions_droh_raw.csv"
        for seed in SEEDS
    ]
    missing = [str(path) for path in baseline_paths + final_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))

    baseline, baseline_metrics = ensemble(baseline_paths, "predictions_raw_baseline_5seed_droh.csv")
    final, final_metrics = ensemble(final_paths, "predictions_final_consistency_5seed_droh.csv")
    baseline_correct = baseline.correct.to_numpy(dtype=bool)
    final_correct = final.correct.to_numpy(dtype=bool)
    baseline_only = int((baseline_correct & ~final_correct).sum())
    final_only = int((~baseline_correct & final_correct).sum())
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(baseline), (10000, len(baseline)))
    differences = (
        final_correct[indices].mean(axis=1) - baseline_correct[indices].mean(axis=1)
    ) * 100
    metrics = {
        "protocol": "fair five-seed probability ensembles; same raw normalized single-view inference",
        "seeds": SEEDS,
        "baseline": {
            "name": "Raw single-view baseline",
            "training_views": 0,
            "consistency_lambda": 0.0,
            **baseline_metrics,
        },
        "final": {
            "name": "Blender8 + consistency fine-tune",
            "training_views": 8,
            "consistency_lambda": 0.3,
            **final_metrics,
        },
        "difference_pp": 100 * (final_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "baseline_only_correct": baseline_only,
        "final_only_correct": final_only,
        "mcnemar_p": float(binomtest(min(baseline_only, final_only), baseline_only + final_only, 0.5).pvalue),
        "bootstrap_95ci_pp": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
