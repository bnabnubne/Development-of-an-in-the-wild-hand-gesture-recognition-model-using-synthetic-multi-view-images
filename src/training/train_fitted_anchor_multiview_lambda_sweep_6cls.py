
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from train_fitted_baseline_6cls import BATCH_SIZE, DEVICE, GRU6, HIDDEN_DIM, LABELS, LR, PATIENCE, EPOCHS, SEED, SkeletonDataset, normalize


MODEL_ROOT = Path(__file__).resolve().parent
MV_CSV = MODEL_ROOT / "metadata/salux_fitted_multiview_6cls_blender.csv"
SALUX_CSV = MODEL_ROOT / "metadata/salux_refined_skeleton_6cls.csv"
DROH_CSV = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls.csv"
CAMERA_ANGLES_DEG = [-45, 0, 30, 45, 60, 90, 120, 135]


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


def evaluate(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            yp.extend(model(x.to(DEVICE))[0].argmax(1).cpu().tolist())
            yt.extend(y.tolist())
    return yt, yp


def save_eval(out_dir, name, yt, yp):
    names = list(LABELS)
    (out_dir / f"report_{name}.txt").write_text(classification_report(
        yt, yp, labels=range(6), target_names=names, digits=6, zero_division=0), encoding="utf-8")
    np.save(out_dir / f"confmat_{name}.npy", confusion_matrix(yt, yp, labels=range(6)))
    return {
        "accuracy": accuracy_score(yt, yp),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "weighted_f1": f1_score(yt, yp, average="weighted"),
        "rows": len(yt),
    }


def tag_float(value):
    return str(value).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-weight", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.lambda_weight < 0:
        raise ValueError("lambda-weight must be non-negative")

    tag = tag_float(args.lambda_weight)
    out_dir = MODEL_ROOT / "results" / f"fitted_anchor_multiview_lambda_{tag}_6cls"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(), flush=True)
        return

    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MV_CSV)
    salux = pd.read_csv(SALUX_CSV)
    droh = pd.read_csv(DROH_CSV)
    cam_cols = [f"cam_{i}_path" for i in range(8)]
    if len(df) != 6547 or any(c not in df for c in cam_cols):
        raise ValueError("Incomplete fitted 8-camera metadata")
    if not set(df.action.unique()) <= set(LABELS):
        raise ValueError(f"Unexpected labels in MV CSV: {sorted(df.action.unique())}")

    train, val = [df[df.split == split].copy() for split in ["train", "val"]]
    salux_test = salux[salux.split == "test"].copy()
    train_loader = DataLoader(FittedMVDataset(train, cam_cols, True), BATCH_SIZE, True, num_workers=0)
    val_loader = DataLoader(FittedMVDataset(val, cam_cols, False), BATCH_SIZE, False, num_workers=0)

    model = GRU6().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_epoch, stale, history = -1.0, -1, 0, []
    best_path = out_dir / "best.pt"

    print(f"device={DEVICE} lambda={args.lambda_weight:g} train/val={len(train)}/{len(val)}", flush=True)
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
            loss = cls + args.lambda_weight * con
            loss.backward()
            optimizer.step()
            total += loss.item()
            cls_total += cls.item()
            con_total += con.item()

        yt, yp = evaluate(model, val_loader)
        val_acc = accuracy_score(yt, yp)
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": LABELS,
                "config": {
                    "anchor": "fitted",
                    "views": "8 camera-space Blender views generated from fitted anchor",
                    "camera_azimuths_deg": CAMERA_ANGLES_DEG,
                    "normalization": "wrist-middle-MCP",
                    "hidden_dim": HIDDEN_DIM,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "seed": SEED,
                    "max_epochs": EPOCHS,
                    "early_stopping_patience": PATIENCE,
                    "selection_metric": "Salux validation fitted-anchor accuracy",
                    "consistency_weight": args.lambda_weight,
                    "num_views": 8,
                },
            }, best_path)
        else:
            stale += 1
        print(
            f"lambda={args.lambda_weight:g} epoch={epoch:03d} loss={row['loss']:.5f} "
            f"cls={row['cls_loss']:.5f} con={row['cons_loss']:.5f} val={val_acc:.5f} best={best:.5f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluations = {}
    for name, frame, col in [
        ("salux_fitted", salux_test, "refined_path"),
        ("salux_raw", salux_test, "raw_path"),
        ("droh_raw", droh, "raw_path"),
        ("droh_fitted_oracle", droh, "refined_path"),
    ]:
        yt, yp = evaluate(model, DataLoader(SkeletonDataset(frame, col), BATCH_SIZE, False))
        evaluations[name] = save_eval(out_dir, name, yt, yp)

    summary = {
        "protocol": "fitted Salux anchor + 8 Blender camera views generated from fitted anchor",
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


if __name__ == "__main__":
    main()
