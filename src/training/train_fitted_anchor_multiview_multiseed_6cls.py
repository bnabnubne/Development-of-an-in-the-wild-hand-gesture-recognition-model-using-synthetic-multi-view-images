"""Run fitted-anchor Blender8 6-class training across multiple seeds.

This script is intentionally separate from the older single-seed fitted scripts so
that defense-slide multi-seed results can be generated without overwriting the
historical seed-42 artifacts.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from train_fitted_baseline_6cls import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    GRU6,
    LABELS,
    LR,
    PATIENCE,
    SkeletonDataset,
    normalize,
)


MODEL_ROOT = Path(__file__).resolve().parent
MV_CSV = MODEL_ROOT / "metadata/salux_fitted_multiview_6cls_blender.csv"
SALUX_CSV = MODEL_ROOT / "metadata/salux_refined_skeleton_6cls.csv"
DROH_CSV = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls.csv"
POSTFILTER_CSV = MODEL_ROOT / "results/droh_postfilter_audit_605_6cls/droh_postfilter_manifest_605.csv"
OUT_ROOT = MODEL_ROOT / "results/fitted_anchor_multiview_multiseed_6cls"
CAMERA_ANGLES_DEG = [-45, 0, 30, 45, 60, 90, 120, 135]
CLASSES = list(LABELS)


def tag_float(value):
    return str(value).replace(".", "p")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FittedMVDataset(Dataset):
    def __init__(self, df, cam_cols, include_views):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.include_views = include_views
        self.anchors = torch.from_numpy(np.stack([
            normalize(np.load(row.fitted_anchor_path, allow_pickle=False))
            for row in self.df.itertuples(index=False)
        ]).astype(np.float32))
        self.labels = torch.tensor([LABELS[action] for action in self.df.action], dtype=torch.long)
        self.views = None
        if include_views:
            self.views = torch.from_numpy(np.stack([
                np.stack([normalize(np.load(getattr(row, c), allow_pickle=False)) for c in cam_cols])
                for row in self.df.itertuples(index=False)
            ]).astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.include_views:
            return self.anchors[idx], self.labels[idx]
        return self.anchors[idx], self.views[idx], self.labels[idx]


def predict_proba(model, loader):
    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.extend(y.tolist())
    return np.asarray(y_true), np.concatenate(probs, axis=0)


def metric_dict(y_true, probs):
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "rows": int(len(y_true)),
    }


def save_prediction_frame(out_dir, name, frame, y_true, probs):
    pred = probs.argmax(axis=1)
    result = frame[["action", "sample_id"]].copy()
    if "source_action" in frame.columns:
        result["source_action"] = frame["source_action"].to_numpy()
    result["true_class"] = [CLASSES[i] for i in y_true]
    result["predicted_class"] = [CLASSES[i] for i in pred]
    result["correct"] = pred == y_true
    result["confidence"] = probs.max(axis=1)
    for idx, class_name in enumerate(CLASSES):
        result[f"prob_{class_name}"] = probs[:, idx]
    result.to_csv(out_dir / f"predictions_{name}.csv", index=False)


def save_eval(out_dir, name, frame, y_true, probs):
    pred = probs.argmax(axis=1)
    (out_dir / f"report_{name}.txt").write_text(
        classification_report(
            y_true,
            pred,
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            digits=6,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    np.save(out_dir / f"confmat_{name}.npy", confusion_matrix(y_true, pred, labels=range(len(CLASSES))))
    save_prediction_frame(out_dir, name, frame, y_true, probs)
    return metric_dict(y_true, probs)


def filter_droh_postfilter(droh):
    if not POSTFILTER_CSV.exists():
        return None
    manifest = pd.read_csv(POSTFILTER_CSV)
    keys = set(zip(manifest["class"], manifest["sample_id"]))
    keep = [(row.source_action, row.sample_id) in keys for row in droh.itertuples(index=False)]
    return droh.loc[keep].copy()


def make_eval_frames():
    salux = pd.read_csv(SALUX_CSV)
    droh = pd.read_csv(DROH_CSV)
    salux_test = salux[salux.split == "test"].copy()
    frames = {
        "salux_fitted": (salux_test, "refined_path"),
        "salux_raw": (salux_test, "raw_path"),
        "droh_raw_675": (droh, "raw_path"),
        "droh_fitted_oracle_675": (droh, "refined_path"),
    }
    postfilter = filter_droh_postfilter(droh)
    if postfilter is not None:
        frames["droh_raw_605"] = (postfilter, "raw_path")
        frames["droh_fitted_oracle_605"] = (postfilter, "refined_path")
    return frames


def train_one(seed, lambda_weight, force):
    out_dir = OUT_ROOT / f"lambda{tag_float(lambda_weight)}_seed{seed}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MV_CSV)
    cam_cols = [f"cam_{i}_path" for i in range(8)]
    if len(df) != 6547 or any(c not in df for c in cam_cols):
        raise ValueError("Incomplete fitted 8-camera metadata")

    train, val = [df[df.split == split].copy() for split in ["train", "val"]]
    train_loader = DataLoader(FittedMVDataset(train, cam_cols, True), BATCH_SIZE, True, num_workers=0)
    val_loader = DataLoader(FittedMVDataset(val, cam_cols, False), BATCH_SIZE, False, num_workers=0)

    model = GRU6().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_epoch, stale, history = -1.0, -1, 0, []
    best_path = out_dir / "best.pt"

    print(
        f"device={DEVICE} seed={seed} lambda={lambda_weight:g} "
        f"train/val={len(train)}/{len(val)}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = cls_total = con_total = 0.0
        for anchor, views, y in train_loader:
            anchor, views, y = anchor.to(DEVICE), views.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits_a, z_a = model(anchor)
            b, v, j, c = views.shape
            logits_v, z_v = model(views.reshape(b * v, j, c))
            logits_v = logits_v.reshape(b, v, -1)
            z_v = z_v.reshape(b, v, -1)
            cls = (criterion(logits_a, y) + sum(criterion(logits_v[:, i], y) for i in range(v))) / (v + 1)
            con = (1 - F.cosine_similarity(z_a[:, None, :], z_v, dim=2)).mean()
            loss = cls + lambda_weight * con
            loss.backward()
            optimizer.step()
            total += loss.item()
            cls_total += cls.item()
            con_total += con.item()

        y_val, p_val = predict_proba(model, val_loader)
        val_acc = float(accuracy_score(y_val, p_val.argmax(axis=1)))
        row = {
            "epoch": epoch,
            "loss": total / len(train_loader),
            "cls_loss": cls_total / len(train_loader),
            "cons_loss": con_total / len(train_loader),
            "val_fitted_anchor_acc": val_acc,
        }
        history.append(row)
        if val_acc > best + 1e-4:
            best, best_epoch, stale = val_acc, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": LABELS,
                    "config": {
                        "anchor": "fitted",
                        "views": "8 camera-space Blender views generated from fitted anchor",
                        "camera_azimuths_deg": CAMERA_ANGLES_DEG,
                        "normalization": "wrist-middle-MCP",
                        "hidden_dim": 128,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "seed": seed,
                        "max_epochs": EPOCHS,
                        "early_stopping_patience": PATIENCE,
                        "selection_metric": "Salux validation fitted-anchor accuracy",
                        "consistency_weight": lambda_weight,
                        "num_views": 8,
                    },
                },
                best_path,
            )
        else:
            stale += 1
        print(
            f"seed={seed} epoch={epoch:03d} loss={row['loss']:.5f} "
            f"cls={row['cls_loss']:.5f} con={row['cons_loss']:.5f} "
            f"val={val_acc:.5f} best={best:.5f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluations = {}
    for name, (frame, col) in make_eval_frames().items():
        y_true, probs = predict_proba(model, DataLoader(SkeletonDataset(frame, col), BATCH_SIZE, False))
        evaluations[name] = save_eval(out_dir, name, frame, y_true, probs)

    summary = {
        "protocol": "fitted Salux anchor + 8 Blender camera views generated from fitted anchor",
        "seed": seed,
        "lambda_weight": lambda_weight,
        "best_epoch": best_epoch,
        "best_val_acc": best,
        "evaluations": evaluations,
        "settings": checkpoint["config"],
        "classes": LABELS,
        "warning": "DrOh fitted is a ground-truth class-template oracle diagnostic.",
        "checkpoint": str(best_path),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def summarize(seeds, lambda_weight):
    rows = []
    for seed in seeds:
        summary_path = OUT_ROOT / f"lambda{tag_float(lambda_weight)}_seed{seed}" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for eval_name, metrics in summary["evaluations"].items():
            row = {
                "seed": seed,
                "lambda_weight": lambda_weight,
                "eval": eval_name,
                "best_epoch": summary["best_epoch"],
                "best_val_acc": summary["best_val_acc"],
            }
            row.update(metrics)
            rows.append(row)

    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(OUT_ROOT / "all_runs.csv", index=False)
    aggregate_rows = []
    for eval_name, part in all_runs.groupby("eval", sort=True):
        aggregate_rows.append({
            "eval": eval_name,
            "num_seeds": int(len(part)),
            "seeds": ",".join(str(s) for s in part.seed),
            "accuracy_mean": float(part.accuracy.mean()),
            "accuracy_std": float(part.accuracy.std(ddof=1)) if len(part) > 1 else 0.0,
            "balanced_accuracy_mean": float(part.balanced_accuracy.mean()),
            "balanced_accuracy_std": float(part.balanced_accuracy.std(ddof=1)) if len(part) > 1 else 0.0,
            "macro_f1_mean": float(part.macro_f1.mean()),
            "macro_f1_std": float(part.macro_f1.std(ddof=1)) if len(part) > 1 else 0.0,
            "weighted_f1_mean": float(part.weighted_f1.mean()),
            "weighted_f1_std": float(part.weighted_f1.std(ddof=1)) if len(part) > 1 else 0.0,
            "rows": int(part.rows.iloc[0]),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT_ROOT / "aggregate.csv", index=False)

    ensemble_rows = []
    for eval_name in sorted(all_runs["eval"].unique()):
        frames, prob_list, key = [], [], None
        for seed in seeds:
            path = OUT_ROOT / f"lambda{tag_float(lambda_weight)}_seed{seed}" / f"predictions_{eval_name}.csv"
            frame = pd.read_csv(path)
            current_key = frame[["action", "sample_id", "true_class"]].copy()
            if "source_action" in frame.columns:
                current_key["source_action"] = frame["source_action"]
            if key is None:
                key = current_key
            elif not current_key.equals(key):
                raise RuntimeError(f"Prediction order mismatch for {eval_name}, seed {seed}")
            frames.append(frame)
            prob_list.append(frame[[f"prob_{class_name}" for class_name in CLASSES]].to_numpy())

        probs = np.mean(prob_list, axis=0)
        y_true = frames[0].true_class.map(LABELS).to_numpy()
        metrics = metric_dict(y_true, probs)
        pred = probs.argmax(axis=1)
        out = frames[0][[c for c in ["action", "sample_id", "source_action", "true_class"] if c in frames[0].columns]].copy()
        out["predicted_class"] = [CLASSES[i] for i in pred]
        out["correct"] = pred == y_true
        out["confidence"] = probs.max(axis=1)
        for idx, class_name in enumerate(CLASSES):
            out[f"prob_{class_name}"] = probs[:, idx]
        out.to_csv(OUT_ROOT / f"ensemble_predictions_{eval_name}.csv", index=False)
        np.save(OUT_ROOT / f"ensemble_confmat_{eval_name}.npy", confusion_matrix(y_true, pred, labels=range(len(CLASSES))))
        ensemble_row = {"eval": eval_name, "members": len(seeds), "seeds": ",".join(str(s) for s in seeds)}
        ensemble_row.update(metrics)
        ensemble_rows.append(ensemble_row)

    ensembles = pd.DataFrame(ensemble_rows)
    ensembles.to_csv(OUT_ROOT / "ensembles.csv", index=False)
    report = {
        "protocol": "fitted Salux anchor + 8 Blender camera views generated from fitted anchor",
        "lambda_weight": lambda_weight,
        "seeds": seeds,
        "aggregate": aggregate.to_dict(orient="records"),
        "ensembles": ensembles.to_dict(orient="records"),
        "warning": "DrOh fitted/oracle uses ground-truth class-template selection and should be reported as an upper-bound diagnostic.",
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-weight", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7, 21, 42])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        train_one(seed, args.lambda_weight, args.force)
    report = summarize(args.seeds, args.lambda_weight)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
