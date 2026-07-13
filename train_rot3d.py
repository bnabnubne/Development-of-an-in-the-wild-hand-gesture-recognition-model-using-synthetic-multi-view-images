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

CSV_PATH = "./metadata/salux_rot3d.csv"
OUT_DIR = Path("./results/rot3d_model_6cls")
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

class Rot3DDataset(Dataset):
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
        x = np.load(row["input_path"]).astype(np.float32)   # (21,3)
        x = torch.tensor(x, dtype=torch.float32)            # (21,3)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

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
        out, _ = self.gru(x)      # (B,21,3)
        feat = out[:, -1, :]      # (B,H)
        logits = self.fc(feat)
        return logits

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
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

df = pd.read_csv(CSV_PATH)
df["action"] = df["action"].replace({
    "thumbup": "thumb",
    "thumbdown": "thumb"
})

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

train_ds = Rot3DDataset(train_df)
label_to_idx = train_ds.label_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}

val_ds = Rot3DDataset(val_df, label_to_idx=label_to_idx)
test_ds = Rot3DDataset(test_df, label_to_idx=label_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

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
best_path = OUT_DIR / "best_rot3d.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
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
                "input_dim": 3,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            }
        }, best_path)
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch:03d} | loss={running_loss:.4f} | val_acc={val_acc:.4f}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (best epoch = {best_epoch}, best val_acc = {best_val_acc:.4f})")
        break

print(f"\nBest val acc: {best_val_acc:.6f} at epoch {best_epoch}")

ckpt = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
test_acc, y_true, y_pred = evaluate(model, test_loader, DEVICE)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

(OUT_DIR / "report_salux_test.txt").write_text(report, encoding="utf-8")
save_confmat(cm, class_names, OUT_DIR / "confmat_salux_test.png", "ROT3D - Salux Test")

summary = {
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "salux_test_acc": test_acc,
}
with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== TRAIN SUMMARY =====")
print("Best val acc:", best_val_acc)
print("Salux test acc:", test_acc)
print("Saved model:", best_path)
print("Saved to:", OUT_DIR)