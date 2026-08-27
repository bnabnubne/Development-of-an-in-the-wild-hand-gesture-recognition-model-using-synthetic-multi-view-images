import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

CKPT_PATH = "./results/baseline_salux/best_baseline_salux.pt"
SALUX_CSV = "./metadata/salux_baseline.csv"
DROH_CSV  = "./metadata/droh_baseline.csv"

OUT_DIR = Path("./results/baseline_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Skeleton3DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["input_path"]).astype(np.float32)   # (21,3)
        x = x.reshape(-1, 3)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

class BaselineGRU3D(nn.Module):
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
        feat = out[:, -1, :]
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

def save_confmat(cm, class_names, out_path, normalize=True):
    cm = np.asarray(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        ) * 100.0
        data = cm_percent
        fmt = "{:.1f}"
        cbar_label = "Percentage (%)"
    else:
        data = cm
        fmt = "{:.0f}"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(data, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100 if normalize else None)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
 
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    threshold = data.max() * 0.55 if data.max() > 0 else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]

            if value == 0:
                continue

            text_color = "white" if value > threshold else "black"
            ax.text(
                j, i,
                fmt.format(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9
            )

    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
label_to_idx = ckpt["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}
cfg = ckpt["config"]

model = BaselineGRU3D(
    input_dim=cfg["input_dim"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    num_classes=len(label_to_idx),
    dropout=cfg["dropout"]
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])

salux_df = pd.read_csv(SALUX_CSV)
droh_df = pd.read_csv(DROH_CSV)


salux_test_df = salux_df[salux_df["split"] == "test"].copy()


salux_ds = Skeleton3DDataset(salux_test_df, label_to_idx=label_to_idx)
droh_ds = Skeleton3DDataset(droh_df, label_to_idx=label_to_idx)

salux_loader = DataLoader(salux_ds, batch_size=64, shuffle=False)
droh_loader = DataLoader(droh_ds, batch_size=64, shuffle=False)

class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

salux_acc, y_true_salux, y_pred_salux = evaluate(model, salux_loader, DEVICE)
report_salux = classification_report(y_true_salux, y_pred_salux, target_names=class_names, digits=4)
cm_salux = confusion_matrix(y_true_salux, y_pred_salux)

(OUT_DIR / "report_salux_test.txt").write_text(report_salux, encoding="utf-8")
save_confmat(cm_salux, class_names, OUT_DIR / "confmat_salux_test.png")

droh_acc, y_true_droh, y_pred_droh = evaluate(model, droh_loader, DEVICE)
report_droh = classification_report(y_true_droh, y_pred_droh, target_names=class_names, digits=4)
cm_droh = confusion_matrix(y_true_droh, y_pred_droh)

(OUT_DIR / "report_droh_test.txt").write_text(report_droh, encoding="utf-8")
save_confmat(cm_droh, class_names, OUT_DIR / "confmat_droh_test.png")

summary = {
    "salux_test_acc": salux_acc,
    "droh_test_acc": droh_acc
}
with open(OUT_DIR / "summary_test.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("===== BASELINE TEST SUMMARY =====")
print("Salux test acc:", salux_acc)
print("DrOh test acc :", droh_acc)
print("Saved to:", OUT_DIR)