import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


CSV_PATH = Path("./metadata/rgb_skeleton_multiview.csv")
OUT_DIR = Path("./results/skeleton_only")

BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 12
LR = 1e-3
SEED = 42
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.0
TRAIN_SKELETON_MODE = "random_cam"
EVAL_MODE = "avg_all_cams"
EVAL_CAM_ID = 0
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SkeletonDataset(Dataset):
    def __init__(self, df, skeleton_cols, label_to_idx=None, mode="eval"):
        self.df = df.reset_index(drop=True)
        self.skeleton_cols = skeleton_cols
        self.mode = mode

        if label_to_idx is None:
            labels = sorted(self.df["action"].unique().tolist())
            self.label_to_idx = {label: i for i, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def choose_cam_idx(self):
        if self.mode == "train" and TRAIN_SKELETON_MODE == "random_cam":
            return random.randrange(len(self.skeleton_cols))
        return EVAL_CAM_ID

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cam_idx = self.choose_cam_idx()
        skeleton = np.load(row[self.skeleton_cols[cam_idx]]).astype(np.float32)
        skeleton = torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return skeleton, label


class SkeletonEvalDataset(Dataset):
    def __init__(self, df, skeleton_cols, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.skeleton_cols = skeleton_cols
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        views = []
        for col in self.skeleton_cols:
            skeleton = np.load(row[col]).astype(np.float32)
            views.append(torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32))
        views = torch.stack(views, dim=0)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return views, label


class SkeletonGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def evaluate_single(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for skeleton, label in loader:
            skeleton = skeleton.to(DEVICE)
            label = label.to(DEVICE)
            logits = model(skeleton)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())
    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def evaluate_multiview(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for views, label in loader:
            views = views.to(DEVICE)
            label = label.to(DEVICE)
            if EVAL_MODE == "single_cam":
                logits = model(views[:, EVAL_CAM_ID])
            elif EVAL_MODE == "avg_all_cams":
                logits_sum = None
                for cam_idx in range(views.shape[1]):
                    logits_cam = model(views[:, cam_idx])
                    logits_sum = logits_cam if logits_sum is None else logits_sum + logits_cam
                logits = logits_sum / views.shape[1]
            else:
                raise ValueError(EVAL_MODE)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())
    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(cm, class_names, out_path):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    skeleton_cols = sorted(
        [c for c in df.columns if c.startswith("skeleton_cam_") and c.endswith("_path")],
        key=lambda x: int(x.split("_")[2]),
    )

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_ds = SkeletonDataset(train_df, skeleton_cols, mode="train")
    label_to_idx = train_ds.label_to_idx
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    val_ds = SkeletonDataset(val_df, skeleton_cols, label_to_idx=label_to_idx, mode="eval")
    test_ds = SkeletonEvalDataset(test_df, skeleton_cols, label_to_idx=label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("DEVICE:", DEVICE, flush=True)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds), flush=True)
    print("Classes:", label_to_idx, flush=True)

    model = SkeletonGRU(num_classes=len(label_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    best_path = OUT_DIR / "best_skeleton_only.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for skeleton, label in train_loader:
            skeleton = skeleton.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(skeleton)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, _, _ = evaluate_single(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "skeleton_cols": skeleton_cols,
                    "eval_mode": EVAL_MODE,
                    "eval_cam_id": EVAL_CAM_ID,
                    "train_skeleton_mode": TRAIN_SKELETON_MODE,
                },
            }, best_path)
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | loss={running_loss:.4f} | val_acc={val_acc:.4f}", flush=True)
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    test_acc, y_true, y_pred = evaluate_multiview(model, test_loader)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    (OUT_DIR / "report_avg_all_cams.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, OUT_DIR / "confmat_avg_all_cams.png")
    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "eval_mode": EVAL_MODE,
        "classes": label_to_idx,
        "csv_path": str(CSV_PATH),
    }
    with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SKELETON-ONLY SUMMARY =====", flush=True)
    print("Best val acc:", best_val_acc, flush=True)
    print("Test acc    :", test_acc, flush=True)
    print("Saved to    :", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
