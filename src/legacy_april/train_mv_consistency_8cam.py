import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SALUX_MV_CSV = "./metadata/salux_multiview3d_8cam_front.csv"
SALUX_ORIG_CSV = "./metadata/salux_baseline.csv"

MERGE_THUMB = False  # False = 7 classes, True = 6 classes

if MERGE_THUMB:
    OUT_DIR = Path("./results/mv_consistency_anchor_8cam_model_6cls")
else:
    OUT_DIR = Path("./results/mv_consistency_anchor_8cam_model_lambda0.3")

OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 20
MIN_DELTA = 1e-4
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.0
CONS_WEIGHT = 0.3
SEED = 42
POOLING = "last"  # "last", "mean", "attention"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# SEED
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================
# DATASET
# =========================
class AnchorConsistencyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cam_cols, label_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols

        if label_to_idx is None:
            labels = sorted(self.df["action"].unique().tolist())
            self.label_to_idx = {lb: i for i, lb in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x_orig = np.load(row["orig_path"]).astype(np.float32)
        x_orig = torch.tensor(x_orig, dtype=torch.float32)

        views = []
        for col in self.cam_cols:
            x = np.load(row[col]).astype(np.float32)
            x = torch.tensor(x, dtype=torch.float32)
            views.append(x)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x_orig, views, y


def collate_anchor(batch):
    x_orig_list, views_list, y_list = zip(*batch)

    x_orig = torch.stack(x_orig_list, dim=0)
    y = torch.stack(y_list, dim=0)

    num_views = len(views_list[0])
    views = []
    for i in range(num_views):
        views.append(torch.stack([v[i] for v in views_list], dim=0))

    return x_orig, views, y

# =========================
# MODEL
# =========================
class SingleViewGRU3D(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        num_layers=1,
        num_classes=7,
        dropout=0.0,
        pooling="mean"  # "last", "mean", "attention"
    ):
        super().__init__()

        self.pooling = pooling

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if pooling == "attention":
            self.attn = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  # out: [B, 21, H]

        if self.pooling == "last":
            z = out[:, -1, :]

        elif self.pooling == "mean":
            z = out.mean(dim=1)

        elif self.pooling == "attention":
            attn_score = self.attn(out)              # [B, 21, 1]
            attn_weight = torch.softmax(attn_score, dim=1)
            z = torch.sum(attn_weight * out, dim=1)  # [B, H]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        logits = self.fc(z)
        return logits, z

# =========================
# LOSSES / EVAL
# =========================
def anchor_consistency_loss(z_orig, z_views):
    losses = []
    for z in z_views:
        losses.append(1.0 - F.cosine_similarity(z_orig, z, dim=1).mean())
    return sum(losses) / len(losses)


def evaluate_original(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_orig, views, y in loader:
            x_orig = x_orig.to(device)
            y = y.to(device)

            logits, _ = model(x_orig)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(cm, class_names, out_path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# LOAD DATA
# =========================
mv_df = pd.read_csv(SALUX_MV_CSV)
orig_df = pd.read_csv(SALUX_ORIG_CSV)

cam_cols = sorted(
    [c for c in mv_df.columns if c.startswith("cam_") and c.endswith("_path")],
    key=lambda x: int(x.split("_")[1])
)

if len(cam_cols) != 8:
    raise ValueError(f"Expected 8 cameras, got {len(cam_cols)}: {cam_cols}")

orig_df = orig_df.rename(columns={"input_path": "orig_path"})

df = pd.merge(
    mv_df,
    orig_df[["sample_id", "action", "split", "orig_path"]],
    on=["sample_id", "action", "split"],
    how="inner"
)

if MERGE_THUMB:
    df["action"] = df["action"].replace({
        "thumbup": "thumb",
        "thumbdown": "thumb"
    })

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

train_ds = AnchorConsistencyDataset(train_df, cam_cols)
label_to_idx = train_ds.label_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}

val_ds = AnchorConsistencyDataset(val_df, cam_cols, label_to_idx=label_to_idx)
test_ds = AnchorConsistencyDataset(test_df, cam_cols, label_to_idx=label_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_anchor)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_anchor)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_anchor)

print("DEVICE:", DEVICE)
print("MERGE_THUMB:", MERGE_THUMB)
print("CAM_COLS:", cam_cols)
print("Classes:", label_to_idx)
print("Merged rows:", len(df))
print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

# =========================
# TRAIN
# =========================
model = SingleViewGRU3D(
    input_dim=3,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=len(label_to_idx),
    dropout=DROPOUT,
    pooling=POOLING
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = -1.0
best_epoch = -1
epochs_no_improve = 0
best_path = OUT_DIR / "best_mv_consistency_anchor.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_cons_loss = 0.0

    for x_orig, views, y in train_loader:
        x_orig = x_orig.to(DEVICE)
        views = [v.to(DEVICE) for v in views]
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits_orig, z_orig = model(x_orig)
        loss_cls = criterion(logits_orig, y)

        z_views = []
        for v in views:
            logits_v, z_v = model(v)
            loss_cls += criterion(logits_v, y)
            z_views.append(z_v)

        loss_cls = loss_cls / (1 + len(views))

        loss_cons = anchor_consistency_loss(z_orig, z_views)
        loss = loss_cls + CONS_WEIGHT * loss_cons

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += loss_cls.item()
        running_cons_loss += loss_cons.item()

    val_acc, _, _ = evaluate_original(model, val_loader, DEVICE)

    if val_acc > best_val_acc + MIN_DELTA:
        best_val_acc = val_acc
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "config": {
                "input_dim": 3,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "cons_weight": CONS_WEIGHT,
                "pooling": POOLING,
                "merge_thumb": MERGE_THUMB,
                "model_type": "mv_consistency_anchor_8cam",
                "num_cameras": len(cam_cols),
                "cam_cols": cam_cols
            }
        }, best_path)
    else:
        epochs_no_improve += 1

    print(
        f"Epoch {epoch:03d} | "
        f"loss={running_loss:.4f} | "
        f"cls={running_cls_loss:.4f} | "
        f"cons={running_cons_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (best epoch = {best_epoch}, best val_acc = {best_val_acc:.4f})")
        break

print(f"\nBest val acc: {best_val_acc:.6f} at epoch {best_epoch}")

# =========================
# INTERNAL TEST
# =========================
ckpt = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
test_acc, y_true, y_pred = evaluate_original(model, test_loader, DEVICE)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)
cm = confusion_matrix(y_true, y_pred)

(OUT_DIR / "report_salux_original_test.txt").write_text(report, encoding="utf-8")
save_confmat(
    cm,
    class_names,
    OUT_DIR / "confmat_salux_original_test.png",
    "MV-Consistency Anchor 8cam - Salux Original Test"
)

summary = {
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "salux_original_test_acc": test_acc,
    "classes": label_to_idx,
    "merge_thumb": MERGE_THUMB,
    "cons_weight": CONS_WEIGHT,
    "num_cameras": len(cam_cols),
    "cam_cols": cam_cols
}

with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== TRAIN SUMMARY =====")
print("Best val acc:", best_val_acc)
print("Salux original test acc:", test_acc)
print("Saved model:", best_path)
print("Saved to:", OUT_DIR)