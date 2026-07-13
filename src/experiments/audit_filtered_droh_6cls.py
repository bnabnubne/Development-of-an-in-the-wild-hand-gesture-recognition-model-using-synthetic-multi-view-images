"""Re-audit all saved DrOh predictions against the post-filter RGB dataset snapshot.

This script never edits old checkpoints, manifests, or result folders. It derives the
valid subset by requiring both membership in the locked 675-row manifest and existence
of the source RGB image after the later Scissors filtering step.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
APRIL = Path(".")
OLD_MANIFEST = APRIL / "metadata/droh_baseline.csv"
MEDIA = APRIL / "test/mediapipe_metadata.csv"
OUT = RESULTS / "droh_postfilter_audit_605_6cls"
SEEDS = [0, 1, 7, 21, 42]


def digest_ids(values):
    text = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def metrics(frame):
    truth = frame.true_class.astype(str).to_numpy()
    pred = frame.predicted_class.astype(str).to_numpy()
    correct = truth == pred
    return {
        "rows": len(frame),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
    }


def filter_predictions(path, valid_ids):
    frame = pd.read_csv(path)
    required = {"sample_id", "true_class", "predicted_class"}
    if not required.issubset(frame.columns):
        return None
    subset = frame[frame.sample_id.astype(str).isin(valid_ids)].copy()
    if len(frame) != 675 or len(subset) != 605:
        return None
    return frame, subset


def paired_stats(left, right):
    left = left.sort_values("sample_id").reset_index(drop=True)
    right = right.sort_values("sample_id").reset_index(drop=True)
    if not left[["sample_id", "true_class"]].equals(right[["sample_id", "true_class"]]):
        raise ValueError("Paired predictions are not aligned")
    a = left.true_class.astype(str).to_numpy() == left.predicted_class.astype(str).to_numpy()
    b = right.true_class.astype(str).to_numpy() == right.predicted_class.astype(str).to_numpy()
    left_only = int((a & ~b).sum()); right_only = int((~a & b).sum())
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(a), (10000, len(a)))
    differences = (b[indices].mean(1) - a[indices].mean(1)) * 100
    return {
        "left_accuracy": float(a.mean()),
        "right_accuracy": float(b.mean()),
        "difference_pp": float((b.mean() - a.mean()) * 100),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_p": float(binomtest(min(left_only, right_only), left_only + right_only, 0.5).pvalue),
        "bootstrap_95ci_pp": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
    }


def evaluate_paper_checkpoint(checkpoint_path, manifest, output_path):
    sys.path.insert(0, str(ROOT))
    import train_paper_controlled_raw_6cls as paper

    dataset = paper.SingleDataset(manifest, "input_path")
    loader = DataLoader(dataset, paper.BATCH_SIZE, shuffle=False)
    checkpoint = torch.load(checkpoint_path, map_location=paper.DEVICE, weights_only=False)
    model = paper.GRU6().to(paper.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    truth, pred, probabilities = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(paper.DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs); truth.extend(y.tolist()); pred.extend(probs.argmax(1).tolist())
    probabilities = np.concatenate(probabilities)
    names = list(paper.LABELS)
    out = manifest[["action", "sample_id"]].copy().reset_index(drop=True)
    out["true_class"] = [names[index] for index in truth]
    out["predicted_class"] = [names[index] for index in pred]
    out["correct"] = out.true_class == out.predicted_class
    out["confidence"] = probabilities.max(1)
    for index, name in enumerate(names): out[f"prob_{name}"] = probabilities[:, index]
    out.to_csv(output_path, index=False)
    return metrics(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD_MANIFEST)
    media = pd.read_csv(MEDIA)
    media = media[media.status == "ok"].copy()
    media["sample_id"] = media.raw_path.map(lambda value: Path(value).stem)
    media["source_rgb_exists_now"] = media.image_path.map(lambda value: Path(value).is_file())
    joined = old.merge(
        media[["sample_id", "class", "image_path", "raw_path", "source_rgb_exists_now"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    valid = joined[joined.source_rgb_exists_now == True].copy()
    excluded = joined[joined.source_rgb_exists_now != True].copy()
    if len(old) != 675 or len(valid) != 605 or len(excluded) != 70:
        raise ValueError(f"Unexpected snapshot sizes: old={len(old)}, valid={len(valid)}, excluded={len(excluded)}")
    valid.to_csv(OUT / "droh_postfilter_manifest_605.csv", index=False)
    excluded.to_csv(OUT / "excluded_stale_scissors_70.csv", index=False)
    valid_ids = set(valid.sample_id.astype(str))

    artifact_rows = []
    for path in sorted(RESULTS.rglob("*predictions*droh*.csv")):
        if OUT in path.parents:
            continue
        try:
            pair = filter_predictions(path, valid_ids)
        except Exception:
            continue
        if pair is None:
            continue
        full, subset = pair
        old_metrics = metrics(full); new_metrics = metrics(subset)
        artifact_rows.append({
            "path": str(path.relative_to(ROOT)),
            "old_rows": old_metrics["rows"], "old_accuracy": old_metrics["accuracy"],
            "filtered_rows": new_metrics["rows"], "filtered_correct": new_metrics["correct"],
            "filtered_accuracy": new_metrics["accuracy"],
            "filtered_balanced_accuracy": new_metrics["balanced_accuracy"],
            "filtered_macro_f1": new_metrics["macro_f1"],
        })
    pd.DataFrame(artifact_rows).sort_values("path").to_csv(OUT / "all_prediction_artifacts_metrics.csv", index=False)

    key_paths = {
        "raw_baseline_5ensemble": RESULTS / "webcam_final_comparison_6cls/predictions_raw_baseline_5seed_droh.csv",
        "ce_checkpoint_5ensemble": RESULTS / "consistency_finetune_6cls/five_seed_control/predictions_ce_checkpoint_droh.csv",
        "low_lr_ce_control_5ensemble": RESULTS / "consistency_finetune_6cls/five_seed_control/predictions_low_lr_ce_control_droh.csv",
        "low_lr_consistency_5ensemble": RESULTS / "consistency_finetune_6cls/five_seed_control/predictions_low_lr_consistency_0p3_droh.csv",
    }
    key_frames = {name: filter_predictions(path, valid_ids)[1] for name, path in key_paths.items()}
    key_metrics = {name: metrics(frame) for name, frame in key_frames.items()}
    for name, frame in key_frames.items(): frame.to_csv(OUT / f"predictions_{name}.csv", index=False)

    comparisons = {
        "raw_baseline_vs_final": paired_stats(key_frames["raw_baseline_5ensemble"], key_frames["low_lr_consistency_5ensemble"]),
        "ce_checkpoint_vs_final": paired_stats(key_frames["ce_checkpoint_5ensemble"], key_frames["low_lr_consistency_5ensemble"]),
        "matched_ce_control_vs_final": paired_stats(key_frames["low_lr_ce_control_5ensemble"], key_frames["low_lr_consistency_5ensemble"]),
    }

    groups = {
        "raw_baseline": [RESULTS / f"defense_experiments_6cls/gru_wrist_middle_views0_lambda0p0_seed{seed}/predictions_droh_raw.csv" for seed in SEEDS],
        "ce_checkpoint": [RESULTS / f"defense_experiments_6cls/gru_wrist_middle_views8_lambda0p0_seed{seed}/predictions_droh_raw.csv" for seed in SEEDS],
        "low_lr_ce_control": [RESULTS / f"consistency_finetune_6cls/lambda0p0_lr1em4_seed{seed}/predictions_droh_raw.csv" for seed in SEEDS],
        "low_lr_consistency": [RESULTS / f"consistency_finetune_6cls/lambda0p3_lr1em4_seed{seed}/predictions_droh_raw.csv" for seed in SEEDS],
    }
    seed_rows=[]; seed_arrays={}
    for name, paths in groups.items():
        values=[]
        for seed, path in zip(SEEDS, paths):
            frame=filter_predictions(path,valid_ids)[1]; value=metrics(frame);values.append(value["accuracy"])
            seed_rows.append({"system":name,"seed":seed,**value})
        seed_arrays[name]=np.asarray(values)
    pd.DataFrame(seed_rows).to_csv(OUT / "five_seed_single_models.csv",index=False)
    seed_summary=[]
    for name, values in seed_arrays.items():
        rows=[row for row in seed_rows if row["system"]==name]
        seed_summary.append({"system":name,"accuracy_mean":float(values.mean()),"accuracy_sample_sd":float(values.std(ddof=1)),
                             "macro_f1_mean":float(np.mean([row["macro_f1"] for row in rows])),
                             "macro_f1_sample_sd":float(np.std([row["macro_f1"] for row in rows],ddof=1))})
    pd.DataFrame(seed_summary).to_csv(OUT / "five_seed_summary.csv",index=False)

    paper_metrics={}
    for name in ["baseline","mv"]:
        checkpoint=RESULTS/f"paper_controlled_raw_{name}_6cls/best.pt"
        paper_metrics[name]=evaluate_paper_checkpoint(checkpoint,valid,OUT/f"predictions_paper_seed42_{name}.csv")

    snapshot = {
        "old_manifest": str(OLD_MANIFEST), "old_rows": len(old),
        "postfilter_rows": len(valid), "excluded_stale_rows": len(excluded),
        "class_counts": valid.action.value_counts().sort_index().to_dict(),
        "valid_ids_sha256": digest_ids(valid.sample_id),
        "excluded_ids_sha256": digest_ids(excluded.sample_id),
        "key_metrics": key_metrics,
        "comparisons": comparisons,
        "paper_seed42_filtered": paper_metrics,
        "matched_seed_level_p": float(ttest_rel(seed_arrays["low_lr_ce_control"], seed_arrays["low_lr_consistency"]).pvalue),
        "note": "No checkpoint was retrained; unchanged valid samples were re-scored from saved predictions. Paper seed42 checkpoints were evaluated directly on the 605-row snapshot.",
    }
    (OUT / "audit.json").write_text(json.dumps(snapshot,indent=2),encoding="utf-8")

    report=f"""# DrOh Post-filter Audit (605 samples)

The 675-row manifest was created before the later Scissors RGB filtering. Seventy rows
still had cached skeletons but no longer had a source RGB image. The reproducible current
snapshot contains 605 samples: 106 OK, 123 Paper, 109 Rock, 42 Scissors, 103 The-Finger,
52 Thumbdown, and 70 Thumbup.

## Key five-model ensembles

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Raw single-view baseline | {100*key_metrics['raw_baseline_5ensemble']['accuracy']:.2f}% ({key_metrics['raw_baseline_5ensemble']['correct']}/605) | {100*key_metrics['raw_baseline_5ensemble']['macro_f1']:.2f}% |
| Blender8 CE checkpoint | {100*key_metrics['ce_checkpoint_5ensemble']['accuracy']:.2f}% ({key_metrics['ce_checkpoint_5ensemble']['correct']}/605) | {100*key_metrics['ce_checkpoint_5ensemble']['macro_f1']:.2f}% |
| Matched low-LR CE control | {100*key_metrics['low_lr_ce_control_5ensemble']['accuracy']:.2f}% ({key_metrics['low_lr_ce_control_5ensemble']['correct']}/605) | {100*key_metrics['low_lr_ce_control_5ensemble']['macro_f1']:.2f}% |
| Low-LR consistency lambda=0.3 | {100*key_metrics['low_lr_consistency_5ensemble']['accuracy']:.2f}% ({key_metrics['low_lr_consistency_5ensemble']['correct']}/605) | {100*key_metrics['low_lr_consistency_5ensemble']['macro_f1']:.2f}% |

Full pipeline: raw baseline to final is +{comparisons['raw_baseline_vs_final']['difference_pp']:.2f} pp,
McNemar p={comparisons['raw_baseline_vs_final']['mcnemar_p']:.6g}.

Consistency-only matched comparison: CE control to lambda=0.3 is
{comparisons['matched_ce_control_vs_final']['difference_pp']:+.2f} pp,
McNemar p={comparisons['matched_ce_control_vs_final']['mcnemar_p']:.6g}. Therefore the
filtered snapshot does not support a statistically significant consistency-only gain.

## Paper-controlled seed42 checkpoints on the filtered snapshot

- Raw baseline: {100*paper_metrics['baseline']['accuracy']:.2f}% ({paper_metrics['baseline']['correct']}/605).
- Blender8 consistency lambda=0.3: {100*paper_metrics['mv']['accuracy']:.2f}% ({paper_metrics['mv']['correct']}/605).

Old 675-row artifacts remain untouched and must be labelled pre-filter/historical.
"""
    (OUT/"FINAL_REPORT.md").write_text(report,encoding="utf-8")
    print(report)


if __name__ == "__main__": main()
