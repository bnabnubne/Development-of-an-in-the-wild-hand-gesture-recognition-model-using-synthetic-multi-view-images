import argparse
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


MODEL_ROOT = Path(__file__).resolve().parent
APRIL_ROOT = MODEL_ROOT.parent.parent / "April"
SALUX_CSV = APRIL_ROOT / "metadata" / "salux_baseline.csv"
SALUX_MV_CSV = APRIL_ROOT / "metadata" / "salux_multiview3d_8cam_front.csv"
DROH_CSV = APRIL_ROOT / "metadata" / "droh_baseline.csv"
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_DIM = 128
SEED = 42
CONS_WEIGHT = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_thumb(df):
    out = df.copy()
    out["action"] = out["action"].replace({"thumbup": "thumb", "thumbdown": "thumb"})
    return out


class SingleDataset(Dataset):
    def __init__(self, df, path_col, labels):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row[self.path_col]).astype(np.float32).reshape(21, 3)
        return torch.from_numpy(x), self.labels[row["action"]]


class MVDataset(Dataset):
    def __init__(self, df, cam_cols, labels):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor = torch.from_numpy(np.load(row["orig_input_path"]).astype(np.float32).reshape(21, 3))
        views = torch.stack([
            torch.from_numpy(np.load(row[col]).astype(np.float32).reshape(21, 3))
            for col in self.cam_cols
        ])
        return anchor, views, self.labels[row["action"]]


class GRU6(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.gru = nn.GRU(3, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.fc(z), z


def evaluate(model, loader, multiview=False):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[-1]
            logits, _ = model(x.to(DEVICE))
            targets.extend(y.tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
    return accuracy_score(targets, preds), targets, preds


def save_eval(out_dir, name, acc, targets, preds, class_names):
    report = classification_report(targets, preds, labels=range(len(class_names)),
                                   target_names=class_names, digits=4, zero_division=0)
    (out_dir / f"report_{name}.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / f"confmat_{name}.npy",
            confusion_matrix(targets, preds, labels=range(len(class_names))))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "mv_consistency"], required=True)
    args = parser.parse_args()
    set_seed(SEED)
    out_dir = MODEL_ROOT / "results" / f"old_pipeline_6cls_{args.experiment}_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    salux = merge_thumb(pd.read_csv(SALUX_CSV))
    droh = merge_thumb(pd.read_csv(DROH_CSV))
    labels = {name: i for i, name in enumerate(sorted(salux["action"].unique()))}
    class_names = [x for x, _ in sorted(labels.items(), key=lambda z: z[1])]
    train_df, val_df, test_df = [salux[salux["split"] == s] for s in ["train", "val", "test"]]
    droh_loader = DataLoader(SingleDataset(droh, "input_path", labels), BATCH_SIZE, False)

    if args.experiment == "baseline":
        train_loader = DataLoader(SingleDataset(train_df, "input_path", labels), BATCH_SIZE, True)
        val_loader = DataLoader(SingleDataset(val_df, "input_path", labels), BATCH_SIZE, False)
        test_loader = DataLoader(SingleDataset(test_df, "input_path", labels), BATCH_SIZE, False)
        epochs, patience, multiview = 100, None, False
    else:
        mv = merge_thumb(pd.read_csv(SALUX_MV_CSV))
        cam_cols = sorted([c for c in mv if c.startswith("cam_") and c.endswith("_path")],
                          key=lambda c: int(c.split("_")[1]))
        train_loader = DataLoader(MVDataset(mv[mv["split"] == "train"], cam_cols, labels), BATCH_SIZE, True)
        val_loader = DataLoader(MVDataset(mv[mv["split"] == "val"], cam_cols, labels), BATCH_SIZE, False)
        test_loader = DataLoader(MVDataset(mv[mv["split"] == "test"], cam_cols, labels), BATCH_SIZE, False)
        epochs, patience, multiview = 150, 20, True

    model = GRU6(len(labels)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, stale = -1., -1, 0
    best_path = out_dir / "best.pt"
    print(f"experiment={args.experiment} device={DEVICE} labels={labels}", flush=True)
    print(f"train/val/test/droh={len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}/{len(droh)}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        for batch in train_loader:
            optimizer.zero_grad()
            if not multiview:
                x, y = batch
                logits, _ = model(x.to(DEVICE))
                loss = criterion(logits, y.to(DEVICE))
            else:
                anchor, views, y = batch
                anchor, views, y = anchor.to(DEVICE), views.to(DEVICE), y.to(DEVICE)
                logits_a, z_a = model(anchor)
                b, v, j, c = views.shape
                logits_v, z_v = model(views.reshape(b * v, j, c))
                logits_v, z_v = logits_v.reshape(b, v, -1), z_v.reshape(b, v, -1)
                cls = (criterion(logits_a, y) +
                       sum(criterion(logits_v[:, i], y) for i in range(v))) / (v + 1)
                cons = (1 - F.cosine_similarity(z_a[:, None], z_v, dim=2)).mean()
                loss = cls + CONS_WEIGHT * cons
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_acc, _, _ = evaluate(model, val_loader, multiview)
        if val_acc > best_acc + (1e-4 if multiview else 0):
            best_acc, best_epoch, stale = val_acc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": labels,
                        "config": {"experiment": args.experiment, "hidden_dim": HIDDEN_DIM,
                                   "cons_weight": CONS_WEIGHT if multiview else 0.,
                                   "num_views": 8 if multiview else 1}}, best_path)
        else:
            stale += 1
        print(f"epoch={epoch:03d} loss={loss_sum/len(train_loader):.4f} val={val_acc:.4f} best={best_acc:.4f}", flush=True)
        if patience is not None and stale >= patience:
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    salux_acc, yt, yp = evaluate(model, test_loader, multiview)
    save_eval(out_dir, "salux_singleview", salux_acc, yt, yp, class_names)
    droh_acc, yt, yp = evaluate(model, droh_loader, False)
    save_eval(out_dir, "droh_singleview", droh_acc, yt, yp, class_names)
    summary = {"experiment": args.experiment, "protocol": "old MediaPipe pipeline 6-class",
               "merge": {"thumbup": "thumb", "thumbdown": "thumb"},
               "best_val_acc": best_acc, "best_epoch": best_epoch,
               "salux_test_acc": salux_acc, "droh_test_acc": droh_acc,
               "classes": labels, "rows": {"salux": len(salux), "droh": len(droh)},
               "checkpoint": str(best_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
