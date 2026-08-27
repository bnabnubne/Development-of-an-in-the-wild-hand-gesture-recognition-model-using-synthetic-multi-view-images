
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import batch_fit_ok as fitter
from train_fitted_baseline_6cls import GRU6, LABELS, normalize


TEMPLATES = {
    "ok": PROJECT_ROOT / "templates/ok_template_joints.npy",
    "paper": PROJECT_ROOT / "templates/paper_template_joints.npy",
    "rock": PROJECT_ROOT / "templates/rock_template_joints.npy",
    "scissors": PROJECT_ROOT / "templates/scrissors_template_joints.npy",
    "the-finger": PROJECT_ROOT / "templates/thefinger_template_joints.npy",
    "thumb": PROJECT_ROOT / "templates/thumb_template_joints.npy",
}
CLASSES = list(LABELS)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_fitter(out_root, template_path):
    fitter.POSE_NAME = "label_free"
    fitter.TEMPLATE_PATH = template_path
    fitter.OUT_ROOT = out_root
    fitter.OUT_2D_DIR = out_root / "2d"
    fitter.OUT_FIT_DIR = out_root / "fitted"
    fitter.OUT_LOG_DIR = out_root / "logs"
    fitter.OUT_OVERLAY_DIR = out_root / "overlays"
    fitter.SAVE_OVERLAY = False
    for d in [fitter.OUT_2D_DIR, fitter.OUT_FIT_DIR, fitter.OUT_LOG_DIR, fitter.OUT_OVERLAY_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def fit_one_template(class_name, template_path, records, out_root, force=False):
    rows = []
    template_out = Path(out_root) / class_name
    configure_fitter(template_out, Path(template_path))
    template = np.load(template_path, allow_pickle=False).astype(np.float32)
    template_centered = template - template[0]
    for row in records:
        sample_id = str(row["sample_id"])
        fitted_path = template_out / "fitted" / f"{sample_id}.npy"
        log_path = template_out / "logs" / f"{sample_id}.json"
        status = "ok"
        error = None
        if force or not fitted_path.exists() or not log_path.exists():
            try:
                fitter.fit_one(Path(row["raw_path"]), template_centered)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        if status == "ok":
            try:
                log = json.loads(log_path.read_text())
                final = log["final_errors"]
                best_view = log["best_view"]
                cost = (
                    0.45 * float(final["all"])
                    + 0.35 * float(final["tips"])
                    + 0.20 * float(final["palm"])
                )
                rows.append({
                    "sample_id": sample_id,
                    "true_action": row["action"],
                    "source_action": row.get("source_action", row["action"]),
                    "template": class_name,
                    "raw_path": row["raw_path"],
                    "fitted_path": str(fitted_path.resolve()),
                    "cost": cost,
                    "final_all": float(final["all"]),
                    "final_tips": float(final["tips"]),
                    "final_palm": float(final["palm"]),
                    "initial_score": float(best_view["score"]),
                    "status": "ok",
                })
            except Exception as exc:
                rows.append({
                    "sample_id": sample_id,
                    "true_action": row["action"],
                    "source_action": row.get("source_action", row["action"]),
                    "template": class_name,
                    "raw_path": row["raw_path"],
                    "fitted_path": str(fitted_path.resolve()),
                    "status": "failed",
                    "error": str(exc),
                })
        else:
            rows.append({
                "sample_id": sample_id,
                "true_action": row["action"],
                "source_action": row.get("source_action", row["action"]),
                "template": class_name,
                "raw_path": row["raw_path"],
                "fitted_path": str(fitted_path.resolve()),
                "status": status,
                "error": error,
            })
    pd.DataFrame(rows).to_csv(Path(out_root) / f"fits_{class_name}.csv", index=False)
    return pd.DataFrame(rows)


def fit_manifest(manifest, out_root, limit=None, force=False, workers=1):
    manifest = manifest.copy()
    if limit is not None:
        manifest = manifest.iloc[:limit].copy()
    records = manifest.to_dict("records")
    out_root = Path(out_root)
    if workers <= 1:
        parts = [
            fit_one_template(class_name, template_path, records, out_root, force)
            for class_name, template_path in TEMPLATES.items()
        ]
    else:
        parts = []
        max_workers = min(workers, len(TEMPLATES))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(fit_one_template, class_name, str(template_path), records, str(out_root), force): class_name
                for class_name, template_path in TEMPLATES.items()
            }
            for fut in as_completed(futures):
                class_name = futures[fut]
                part = fut.result()
                print(f"finished template={class_name} rows={len(part)}", flush=True)
                parts.append(part)
                pd.concat(parts, ignore_index=True).to_csv(out_root / "all_template_fits.csv", index=False)
    fits = pd.concat(parts, ignore_index=True)
    fits.to_csv(out_root / "all_template_fits.csv", index=False)
    return fits


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = GRU6().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_probs(model, paths, batch_size=256):
    probs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            x = torch.from_numpy(np.stack([
                normalize(np.load(path, allow_pickle=False)) for path in chunk
            ])).to(DEVICE)
            logits, _ = model(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def metric_dict(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "rows": len(y_true),
    }


def evaluate_decisions(fits, model, out_root, checkpoint_path):
    ok = fits[fits.status == "ok"].copy()
    counts = ok.groupby("sample_id").template.nunique()
    complete_ids = set(counts[counts == len(CLASSES)].index)
    ok = ok[ok.sample_id.isin(complete_ids)].copy()
    ok["true_idx"] = ok.true_action.map(LABELS)
    ok["template_idx"] = ok.template.map(LABELS)

    probs = predict_probs(model, ok.fitted_path.tolist())
    ok[[f"prob_{c}" for c in CLASSES]] = probs
    ok["prob_own_template_class"] = [probs[i, idx] for i, idx in enumerate(ok.template_idx)]
    ok["pred_by_classifier"] = probs.argmax(axis=1)
    ok["max_prob"] = probs.max(axis=1)

    y_true = []
    pred_cost = []
    pred_cost_then_classifier = []
    pred_best_classifier_any_template = []
    pred_own_template_prob = []

    for sample_id, group in ok.groupby("sample_id", sort=False):
        group = group.sort_values("template")
        true_idx = int(group.true_idx.iloc[0])
        y_true.append(true_idx)

        min_cost = group.loc[group.cost.idxmin()]
        pred_cost.append(int(min_cost.template_idx))
        pred_cost_then_classifier.append(int(min_cost.pred_by_classifier))

        best_any = group.loc[group.max_prob.idxmax()]
        pred_best_classifier_any_template.append(int(best_any.pred_by_classifier))

        best_own = group.loc[group.prob_own_template_class.idxmax()]
        pred_own_template_prob.append(int(best_own.template_idx))

    evaluations = {
        "template_argmin_cost": metric_dict(y_true, pred_cost),
        "argmin_cost_then_classifier": metric_dict(y_true, pred_cost_then_classifier),
        "best_classifier_confidence_over_6_fits": metric_dict(y_true, pred_best_classifier_any_template),
        "best_own_template_probability": metric_dict(y_true, pred_own_template_prob),
    }

    for name, pred in [
        ("template_argmin_cost", pred_cost),
        ("argmin_cost_then_classifier", pred_cost_then_classifier),
        ("best_classifier_confidence_over_6_fits", pred_best_classifier_any_template),
        ("best_own_template_probability", pred_own_template_prob),
    ]:
        (out_root / f"report_{name}.txt").write_text(classification_report(
            y_true, pred, labels=range(6), target_names=CLASSES, digits=6, zero_division=0
        ), encoding="utf-8")
        np.save(out_root / f"confmat_{name}.npy", confusion_matrix(y_true, pred, labels=range(6)))

    ok.to_csv(out_root / "all_template_fits_with_classifier_probs.csv", index=False)
    summary = {
        "protocol": "label-free 6-template fitting; no ground-truth label is used for template selection at test time",
        "checkpoint": str(checkpoint_path),
        "complete_samples": len(y_true),
        "template_classes": CLASSES,
        "evaluations": evaluations,
        "notes": {
            "template_argmin_cost": "prediction is the class whose template has the lowest fitting cost",
            "argmin_cost_then_classifier": "choose the lowest-cost fitted skeleton, then use the skeleton classifier output",
            "best_classifier_confidence_over_6_fits": "run classifier on all six fitted skeletons and take the most confident predicted class",
            "best_own_template_probability": "for each template c, use classifier probability P(c | fitted_by_template_c), then choose the largest",
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "metadata/droh_refined_skeleton_6cls.csv"))
    parser.add_argument("--out-root", default=str(ROOT / "results/label_free_template_pipeline_6cls_droh"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/fitted_anchor_multiview_6cls/best.pt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    set_seed(42)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    fits = fit_manifest(manifest, out_root, limit=args.limit, force=args.force, workers=args.workers)
    failed = fits[fits.status != "ok"]
    if not failed.empty:
        failed.to_csv(out_root / "failed_fits.csv", index=False)
    model = load_model(Path(args.checkpoint))
    summary = evaluate_decisions(fits, model, out_root, Path(args.checkpoint))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
