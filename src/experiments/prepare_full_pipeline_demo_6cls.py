"""Lock the naive-camera baseline versus full proposed pipeline comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VALID_MANIFEST = RESULTS / "droh_postfilter_audit_605_6cls/droh_postfilter_manifest_605.csv"
BASE_ROOT = RESULTS / "clean_viewpoint_factorial_v2_6cls"
FINAL_PRED = RESULTS / "droh_postfilter_audit_605_6cls/predictions_low_lr_consistency_5ensemble.csv"
OUT = RESULTS / "full_pipeline_vs_naive_demo_6cls"
SEEDS = [0, 1, 7, 21, 42]
CLASSES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
PROB_COLUMNS = [f"prob_{name}" for name in CLASSES]


def summary(frame):
    return {
        "rows": len(frame), "correct": int(frame.correct.sum()),
        "accuracy": float(frame.correct.mean()),
        "macro_f1": float(f1_score(frame.true_class, frame.predicted_class, average="macro")),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    valid = pd.read_csv(VALID_MANIFEST); valid_ids = set(valid.sample_id.astype(str))
    paths = [BASE_ROOT / f"camera_single_seed{seed}/predictions_droh.csv" for seed in SEEDS]
    missing = [str(path) for path in paths + [FINAL_PRED] if not path.is_file()]
    if missing: raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    frames = [pd.read_csv(path) for path in paths]
    frames = [frame[frame.sample_id.astype(str).isin(valid_ids)].sort_values("sample_id").reset_index(drop=True) for frame in frames]
    key = frames[0][["action", "sample_id", "true_class"]]
    if not all(len(frame) == 605 and frame[["action", "sample_id", "true_class"]].equals(key) for frame in frames):
        raise ValueError("Naive baseline predictions are not aligned")
    probabilities = np.mean([frame[PROB_COLUMNS].to_numpy() for frame in frames], axis=0)
    baseline = key.copy(); baseline["predicted_class"] = np.asarray(CLASSES)[probabilities.argmax(1)]
    baseline["correct"] = baseline.true_class == baseline.predicted_class
    baseline["confidence"] = probabilities.max(1)
    for index, name in enumerate(CLASSES): baseline[f"prob_{name}"] = probabilities[:, index]
    final = pd.read_csv(FINAL_PRED).sort_values("sample_id").reset_index(drop=True)
    if not final[["action", "sample_id", "true_class"]].equals(key):
        raise ValueError("Final predictions are not aligned with the baseline")
    a=baseline.correct.to_numpy(bool);b=final.correct.to_numpy(bool)
    a_only=int((a&~b).sum());b_only=int((~a&b).sum())
    rng=np.random.default_rng(42);indices=rng.integers(0,len(a),(10000,len(a)));diff=(b[indices].mean(1)-a[indices].mean(1))*100
    metrics={
        "comparison_type":"combined system-level comparison, not a multiview-only ablation",
        "seeds":SEEDS,
        "naive_baseline":{"pipeline":"RGB -> MediaPipe -> handedness + wrist center + scale -> camera-single GRU",**summary(baseline)},
        "full_proposed":{"pipeline":"RGB -> MediaPipe -> canonicalization -> Blender8 + lambda=0.3 consistency ensemble",**summary(final)},
        "difference_pp":float((b.mean()-a.mean())*100),"baseline_only_correct":a_only,"proposed_only_correct":b_only,
        "mcnemar_p":float(binomtest(min(a_only,b_only),a_only+b_only,.5).pvalue),
        "bootstrap_95ci_pp":[float(value) for value in np.quantile(diff,[.025,.975])],
    }
    baseline.to_csv(OUT/"predictions_naive_camera_baseline_5ensemble.csv",index=False)
    final.to_csv(OUT/"predictions_full_proposed_5ensemble.csv",index=False)
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
