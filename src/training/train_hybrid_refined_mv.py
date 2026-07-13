import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from train_refined_skeleton import CAMERA_ROTATIONS, GRUClassifier, preprocess


MODEL_ROOT = Path(__file__).resolve().parent
SALUX_CSV = MODEL_ROOT / "metadata" / "salux_refined_skeleton_5cls.csv"
DROH_CSV = MODEL_ROOT / "metadata" / "droh_refined_skeleton_5cls.csv"
OUT_DIR = MODEL_ROOT / "results" / "skeleton_mediapipe_anchor_refined_views_mv"
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
CONS_WEIGHT = 0.3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HybridDataset(Dataset):
    def __init__(self, df, labels, include_views):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.include_views = include_views

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Anchor matches the representation available at external inference.
        anchor = preprocess(np.load(row["mediapipe_path"]), "scale")
        y = self.labels[row["action"]]
        if not self.include_views:
            return torch.from_numpy(anchor), y

        # Only the training-time view branch uses anatomically fitted skeletons.
        refined = preprocess(np.load(row["refined_path"]), "scale")
        views = np.stack([(r @ refined.T).T for r in CAMERA_ROTATIONS]).astype(np.float32)
        views -= views[:, 0:1, :]
        return torch.from_numpy(anchor), torch.from_numpy(views), y


def evaluate_anchor(model, loader):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for batch in loader:
            anchor, y = batch[0], batch[-1]
            logits, _ = model(anchor.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    return accuracy_score(targets, preds), targets, preds


def save_eval(name, acc, targets, preds, class_names):
    report = classification_report(
        targets, preds, labels=range(len(class_names)), target_names=class_names,
        digits=4, zero_division=0,
    )
    (OUT_DIR / f"report_{name}.txt").write_text(report, encoding="utf-8")
    np.save(OUT_DIR / f"confmat_{name}.npy",
            confusion_matrix(targets, preds, labels=range(len(class_names))))
    return acc


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    salux = pd.read_csv(SALUX_CSV)
    droh = pd.read_csv(DROH_CSV)
    labels = {name: i for i, name in enumerate(sorted(salux["action"].unique()))}
    train_df = salux[salux["split"] == "train"]
    val_df = salux[salux["split"] == "val"]
    test_df = salux[salux["split"] == "test"]
    train_loader = DataLoader(HybridDataset(train_df, labels, True), BATCH_SIZE, True)
    val_loader = DataLoader(HybridDataset(val_df, labels, False), BATCH_SIZE, False)
    test_loader = DataLoader(HybridDataset(test_df, labels, False), BATCH_SIZE, False)
    droh_loader = DataLoader(HybridDataset(droh, labels, False), BATCH_SIZE, False)

    model = GRUClassifier(len(labels)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, stale = -1., -1, 0
    best_path = OUT_DIR / "best.pt"
    print(f"device={DEVICE} train/val/test/droh={len(train_df)}/{len(val_df)}/{len(test_df)}/{len(droh)}", flush=True)
    print("anchor=mediapipe_scale views=refined_scale num_views=8", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, cls_total, cons_total = 0., 0., 0.
        for anchor, views, y in train_loader:
            anchor, views, y = anchor.to(DEVICE), views.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits_a, z_a = model(anchor)
            b, v, j, c = views.shape
            logits_v, z_v = model(views.reshape(b * v, j, c))
            logits_v, z_v = logits_v.reshape(b, v, -1), z_v.reshape(b, v, -1)
            loss_cls = (criterion(logits_a, y) +
                        sum(criterion(logits_v[:, i], y) for i in range(v))) / (v + 1)
            loss_cons = (1 - F.cosine_similarity(z_a[:, None, :], z_v, dim=2)).mean()
            loss = loss_cls + CONS_WEIGHT * loss_cons
            loss.backward()
            optimizer.step()
            total += loss.item()
            cls_total += loss_cls.item()
            cons_total += loss_cons.item()

        val_acc, _, _ = evaluate_anchor(model, val_loader)
        if val_acc > best_acc + 1e-4:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "label_to_idx": labels,
                "config": {"anchor_source": "mediapipe", "view_source": "refined",
                           "normalization": "scale", "num_views": 8,
                           "cons_weight": CONS_WEIGHT, "hidden_dim": 128},
            }, best_path)
        else:
            stale += 1
        n = len(train_loader)
        print(f"epoch={epoch:03d} loss={total/n:.4f} cls={cls_total/n:.4f} cons={cons_total/n:.4f} val_anchor={val_acc:.4f} best={best_acc:.4f}", flush=True)
        if stale >= PATIENCE:
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = [x for x, _ in sorted(labels.items(), key=lambda z: z[1])]
    salux_acc, yt, yp = evaluate_anchor(model, test_loader)
    save_eval("salux_mediapipe_anchor", salux_acc, yt, yp, class_names)
    droh_acc, yt, yp = evaluate_anchor(model, droh_loader)
    save_eval("droh_mediapipe_singleview", droh_acc, yt, yp, class_names)
    summary = {
        "experiment": "MediaPipe anchor + refined 8-view MV-Consistency",
        "best_val_anchor_acc": best_acc, "best_epoch": best_epoch,
        "salux_mediapipe_test_acc": salux_acc,
        "droh_mediapipe_singleview_acc": droh_acc,
        "rows": {"train": len(train_df), "val": len(val_df),
                 "salux_test": len(test_df), "droh_test": len(droh)},
        "classes": labels, "checkpoint": str(best_path),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
