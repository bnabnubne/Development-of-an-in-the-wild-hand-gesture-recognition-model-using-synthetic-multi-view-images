import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "./metadata/salux_multiview3d.csv"
OUT_DIR = Path("./results/mv3d_plain_model_6cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 20
MIN_DELTA = 1e-4
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.0
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
class MultiView3DDataset(Dataset):
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

        cams = []
        for col in ["cam_0_path", "cam_1_path", "cam_2_path", "cam_3_path"]:
            x = np.load(row[col]).astype(np.float32)   # (21,3)
            x = torch.tensor(x, dtype=torch.float32)   # (21,3)
            cams.append(x)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return cams[0], cams[1], cams[2], cams[3], y

# =========================
# MODEL
# =========================
class ViewEncoder3D(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]

class MV3DModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, num_classes=6, dropout=0.0):
        super().__init__()
        self.encoder = ViewEncoder3D(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, cam0, cam1, cam2, cam3):
        z0 = self.encoder(cam0)
        z1 = self.encoder(cam1)
        z2 = self.encoder(cam2)
        z3 = self.encoder(cam3)

        z = (z0 + z1 + z2 + z3) / 4.0
        logits = self.fc(z)

        # vẫn return đủ để dùng chung test file với bản alignment
        return logits, z0, z1, z2, z3

# =========================
# UTILS
# =========================
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for cam0, cam1, cam2, cam3, y in loader:
            cam0 = cam0.to(device)
            cam1 = cam1.to(device)
            cam2 = cam2.to(device)
            cam3 = cam3.to(device)
            y = y.to(device)

            logits, _, _, _, _ = model(cam0, cam1, cam2, cam3)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds

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
df = pd.read_csv(CSV_PATH)

# merge thumbup + thumbdown -> thumb
df["action"] = df["action"].replace({
    "thumbup": "thumb",
    "thumbdown": "thumb"
})

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

train_ds = MultiView3DDataset(train_df)
label_to_idx = train_ds.label_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}

val_ds = MultiView3DDataset(val_df, label_to_idx=label_to_idx)
test_ds = MultiView3DDataset(test_df, label_to_idx=label_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", label_to_idx)
print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

# =========================
# TRAIN
# =========================
model = MV3DModel(
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
best_path = OUT_DIR / "best_mv3d_plain.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for cam0, cam1, cam2, cam3, y in train_loader:
        cam0 = cam0.to(DEVICE)
        cam1 = cam1.to(DEVICE)
        cam2 = cam2.to(DEVICE)
        cam3 = cam3.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits, _, _, _, _ = model(cam0, cam1, cam2, cam3)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc, _, _ = evaluate(model, val_loader, DEVICE)

    if val_acc > best_val_acc + MIN_DELTA:
        best_val_acc = val_acc
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "config": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "type": "mv3d_plain_6cls"
            }
        }, best_path)
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch:03d} | loss={running_loss:.4f} | val_acc={val_acc:.4f}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (best epoch = {best_epoch}, best val_acc = {best_val_acc:.4f})")
        break

print(f"\nBest val acc: {best_val_acc:.6f} at epoch {best_epoch}")

# =========================
# INTERNAL SALUX TEST
# =========================
ckpt = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
test_acc, y_true, y_pred = evaluate(model, test_loader, DEVICE)

report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)
cm = confusion_matrix(y_true, y_pred)

(OUT_DIR / "report_salux_test.txt").write_text(report, encoding="utf-8")
save_confmat(cm, class_names, OUT_DIR / "confmat_salux_test.png", "MV3D Plain 6cls - Salux Test")

summary = {
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "salux_test_acc": test_acc,
    "classes": label_to_idx
}

with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== TRAIN SUMMARY =====")
print("Best val acc:", best_val_acc)
print("Salux test acc:", test_acc)
print("Saved model:", best_path)
print("Saved to:", OUT_DIR)