"""Evaluate fixed equal-weight three-seed ensembles from saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parent / "results/defense_experiments_6cls"
OUT_DIR = ROOT / "ensembles"
CLASSES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
SEEDS = [0, 1, 42]
CONFIGS = {
    "raw_baseline_3seed": (0, 0.0),
    "mv_ce_only_3seed": (8, 0.0),
    "mv_consistency_3seed": (8, 0.3),
}


def tag(views, weight, seed):
    return f"gru_wrist_middle_views{views}_lambda{str(weight).replace('.', 'p')}_seed{seed}"


def metrics(true, predicted):
    return {
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, average="macro"),
        "weighted_f1": f1_score(true, predicted, average="weighted"),
        "rows": len(true),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (views, weight) in CONFIGS.items():
        summary[name] = {}
        for dataset in ["salux_raw", "droh_raw"]:
            frames = [
                pd.read_csv(ROOT / tag(views, weight, seed) / f"predictions_{dataset}.csv")
                for seed in SEEDS
            ]
            reference = frames[0]
            for frame in frames[1:]:
                if not frame.sample_id.equals(reference.sample_id) or not frame.true_class.equals(reference.true_class):
                    raise RuntimeError(f"Prediction rows do not align for {name}/{dataset}")
            columns = [f"prob_{class_name}" for class_name in CLASSES]
            probabilities = np.mean([frame[columns].to_numpy() for frame in frames], axis=0)
            predicted_indices = probabilities.argmax(axis=1)
            predicted = np.asarray(CLASSES)[predicted_indices]
            true = reference.true_class.to_numpy()
            output = reference[["action", "sample_id", "true_class"]].copy()
            output["predicted_class"] = predicted
            output["correct"] = predicted == true
            output["confidence"] = probabilities.max(axis=1)
            for index, class_name in enumerate(CLASSES):
                output[f"prob_{class_name}"] = probabilities[:, index]
            output.to_csv(OUT_DIR / f"predictions_{name}_{dataset}.csv", index=False)
            np.save(
                OUT_DIR / f"confmat_{name}_{dataset}.npy",
                confusion_matrix(true, predicted, labels=CLASSES),
            )
            summary[name][dataset] = metrics(true, predicted)
        summary[name]["protocol"] = {
            "members": [tag(views, weight, seed) for seed in SEEDS],
            "fusion": "equal-weight arithmetic mean of class probabilities",
            "selection": "fixed seeds [0,1,42]; no DrOh tuning",
            "single_view_inference": True,
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
