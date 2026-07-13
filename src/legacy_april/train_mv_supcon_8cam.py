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

MERGE_THUMB = False  # True = 6 classes, False = 7 classes

if MERGE_THUMB:
    OUT_DIR = Path("./results/mv_supcon_anchor_8cam_model_6cls")
else:
    OUT_DIR = Path("./results/mv_supcon_anchor_8cam_model_7cls_alpha1.0")

OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 20
MIN_DELTA = 1e-4
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.0
SUPCON_WEIGHT = 1.0
TEMPERATURE = 0.1
SEED = 42

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
class SupConAnchorDataset(Dataset):
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
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=1, num_classes=7, dropout=0.0):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1, :]
        logits = self.fc(z)
        return logits, z

# =========================
# LOSSES / EVAL
# =========================
def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device

    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask).to(device)
    logits_mask.fill_diagonal_(0)

    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    positive_count = mask.sum(dim=1)
    valid = positive_count > 0

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-8)
    loss = -mean_log_prob_pos[valid].mean()

    return loss


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

train_ds = SupConAnchorDataset(train_df, cam_cols)
label_to_idx = train_ds.label_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}

val_ds = SupConAnchorDataset(val_df, cam_cols, label_to_idx=label_to_idx)
test_ds = SupConAnchorDataset(test_df, cam_cols, label_to_idx=label_to_idx)

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
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = -1.0
best_epoch = -1
epochs_no_improve = 0
best_path = OUT_DIR / "best_mv_supcon_anchor.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_supcon_loss = 0.0

    for x_orig, views, y in train_loader:
        x_orig = x_orig.to(DEVICE)
        views = [v.to(DEVICE) for v in views]
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits_orig, z_orig = model(x_orig)

        loss_cls = criterion(logits_orig, y)
        features = [z_orig]
        labels = [y]

        for v in views:
            logits_v, z_v = model(v)
            loss_cls += criterion(logits_v, y)
            features.append(z_v)
            labels.append(y)

        loss_cls = loss_cls / (1 + len(views))

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        loss_supcon = supervised_contrastive_loss(
            features,
            labels,
            temperature=TEMPERATURE
        )

        loss = loss_cls + SUPCON_WEIGHT * loss_supcon

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += loss_cls.item()
        running_supcon_loss += loss_supcon.item()

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
                "supcon_weight": SUPCON_WEIGHT,
                "temperature": TEMPERATURE,
                "merge_thumb": MERGE_THUMB,
                "model_type": "mv_supcon_anchor_8cam",
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
        f"supcon={running_supcon_loss:.4f} | "
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
    "MV-SupCon Anchor 8cam - Salux Original Test"
)

summary = {
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "salux_original_test_acc": test_acc,
    "classes": label_to_idx,
    "merge_thumb": MERGE_THUMB,
    "supcon_weight": SUPCON_WEIGHT,
    "temperature": TEMPERATURE,
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