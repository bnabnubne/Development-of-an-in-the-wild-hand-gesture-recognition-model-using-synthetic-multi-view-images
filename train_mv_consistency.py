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
SALUX_MV_CSV = "./metadata/salux_multiview3d.csv"
SALUX_ORIG_CSV = "./metadata/salux_baseline.csv"

MERGE_THUMB = False  # True: 6 classes, False: 7 classes

if MERGE_THUMB:
    OUT_DIR = Path("./results/mv_consistency_anchor_model_6cls")
else:
    OUT_DIR = Path("./results/mv_consistency_anchor_model_7cls")

OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 20
MIN_DELTA = 1e-4
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.0
CONS_WEIGHT = 0.05
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
class AnchorConsistencyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx=None):
        self.df = df.reset_index(drop=True)

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

        cams = []
        for col in ["cam_0_path", "cam_1_path", "cam_2_path", "cam_3_path"]:
            x = np.load(row[col]).astype(np.float32)
            x = torch.tensor(x, dtype=torch.float32)
            cams.append(x)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)

        return x_orig, cams[0], cams[1], cams[2], cams[3], y

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
# LOSSES
# =========================
def anchor_consistency_loss(z_orig, z0, z1, z2, z3):
    losses = [
        1.0 - F.cosine_similarity(z_orig, z0, dim=1).mean(),
        1.0 - F.cosine_similarity(z_orig, z1, dim=1).mean(),
        1.0 - F.cosine_similarity(z_orig, z2, dim=1).mean(),
        1.0 - F.cosine_similarity(z_orig, z3, dim=1).mean(),
    ]
    return sum(losses) / len(losses)

# =========================
# EVALUATION
# =========================
def evaluate_original(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_orig, cam0, cam1, cam2, cam3, y in loader:
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
# LOAD + MERGE DATA
# =========================
mv_df = pd.read_csv(SALUX_MV_CSV)
orig_df = pd.read_csv(SALUX_ORIG_CSV)

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

train_ds = AnchorConsistencyDataset(train_df)
label_to_idx = train_ds.label_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}

val_ds = AnchorConsistencyDataset(val_df, label_to_idx=label_to_idx)
test_ds = AnchorConsistencyDataset(test_df, label_to_idx=label_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("DEVICE:", DEVICE)
print("MERGE_THUMB:", MERGE_THUMB)
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
best_path = OUT_DIR / "best_mv_consistency_anchor.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_cons_loss = 0.0

    for x_orig, cam0, cam1, cam2, cam3, y in train_loader:
        x_orig = x_orig.to(DEVICE)
        cam0 = cam0.to(DEVICE)
        cam1 = cam1.to(DEVICE)
        cam2 = cam2.to(DEVICE)
        cam3 = cam3.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits_orig, z_orig = model(x_orig)
        logits0, z0 = model(cam0)
        logits1, z1 = model(cam1)
        logits2, z2 = model(cam2)
        logits3, z3 = model(cam3)

        loss_cls = (
            criterion(logits_orig, y)
            + criterion(logits0, y)
            + criterion(logits1, y)
            + criterion(logits2, y)
            + criterion(logits3, y)
        ) / 5.0

        loss_cons = anchor_consistency_loss(z_orig, z0, z1, z2, z3)
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
                "merge_thumb": MERGE_THUMB,
                "model_type": "mv_consistency_with_original_anchor"
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
# INTERNAL TEST ON SALUX ORIGINAL
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
    "MV-Consistency Anchor - Salux Original Test"
)

summary = {
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "salux_original_test_acc": test_acc,
    "classes": label_to_idx,
    "merge_thumb": MERGE_THUMB,
    "cons_weight": CONS_WEIGHT
}

with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== TRAIN SUMMARY =====")
print("Best val acc:", best_val_acc)
print("Salux original test acc:", test_acc)
print("Saved model:", best_path)
print("Saved to:", OUT_DIR)