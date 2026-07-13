import json
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
EXPERIMENT_NAME = "4view"

CKPT_PATH = "./results/multiview_salux/best_multiview_salux.pt"
SALUX_CSV = "./metadata/salux_multiview.csv"
DROH_REAL2D_CSV = "./metadata/droh_real2d_wristnorm.csv"

OUT_DIR = Path("./results/multiview_test_4view_droh")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# =========================
# DATASET
# =========================
class SelectedViewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx, selected_views):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.selected_views = selected_views

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cams = []

        for col in self.selected_views:
            x = np.load(row[col]).astype(np.float32)   # (21,2)
            x = x.reshape(-1)                          # (42,)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,42)
            cams.append(x)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return cams, y


class RealSingleViewRepeatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx, num_views: int):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.num_views = num_views

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["input_path"]).astype(np.float32)   # (21,2)
        x = x.reshape(-1)                                   # (42,)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,42)

        cams = [x.clone() for _ in range(self.num_views)]
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return cams, y

# =========================
# MODEL
# =========================
class ViewEncoder(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, num_layers=1, dropout=0.0):
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

class MultiViewGRU(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, num_layers=1, num_classes=7, dropout=0.0, num_views=4):
        super().__init__()
        self.encoder = ViewEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.num_views = num_views
        self.fc = nn.Linear(hidden_dim * num_views, num_classes)

    def forward(self, cams):
        feats = [self.encoder(cam) for cam in cams]
        feat = torch.cat(feats, dim=1)
        return self.fc(feat)

# =========================
# UTILS
# =========================
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for cams, y in loader:
            cams = [cam.to(device) for cam in cams]
            y = y.to(device)

            logits = model(cams)
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
# LOAD MODEL
# =========================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
label_to_idx = ckpt["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}
cfg = ckpt["config"]

selected_views = cfg.get("selected_views", ["cam_0_path", "cam_2_path", "cam_4_path", "cam_5_path"])

model = MultiViewGRU(
    input_dim=cfg["input_dim"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    num_classes=len(label_to_idx),
    dropout=cfg["dropout"],
    num_views=len(selected_views)
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

# =========================
# LOAD DATA
# =========================
salux_df = pd.read_csv(SALUX_CSV)
salux_test_df = salux_df[salux_df["split"] == "test"].copy()

droh_df = pd.read_csv(DROH_REAL2D_CSV)

salux_ds = SelectedViewsDataset(salux_test_df, label_to_idx=label_to_idx, selected_views=selected_views)
droh_ds = RealSingleViewRepeatDataset(droh_df, label_to_idx=label_to_idx, num_views=len(selected_views))

salux_loader = DataLoader(salux_ds, batch_size=BATCH_SIZE, shuffle=False)
droh_loader = DataLoader(droh_ds, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# TEST SALUX
# =========================
salux_acc, y_true_salux, y_pred_salux = evaluate(model, salux_loader, DEVICE)
report_salux = classification_report(y_true_salux, y_pred_salux, target_names=class_names, digits=4, zero_division=0)
cm_salux = confusion_matrix(y_true_salux, y_pred_salux)

(OUT_DIR / "report_salux_test.txt").write_text(report_salux, encoding="utf-8")
save_confmat(cm_salux, class_names, OUT_DIR / "confmat_salux_test.png", f"{EXPERIMENT_NAME} - Salux Test")

# =========================
# TEST DROH REAL 2D
# =========================
droh_acc, y_true_droh, y_pred_droh = evaluate(model, droh_loader, DEVICE)
report_droh = classification_report(y_true_droh, y_pred_droh, target_names=class_names, digits=4, zero_division=0)
cm_droh = confusion_matrix(y_true_droh, y_pred_droh)

(OUT_DIR / "report_droh_test.txt").write_text(report_droh, encoding="utf-8")
save_confmat(cm_droh, class_names, OUT_DIR / "confmat_droh_test.png", f"{EXPERIMENT_NAME} - DrOh REAL 2D Test")

summary = {
    "experiment_name": EXPERIMENT_NAME,
    "selected_views": selected_views,
    "salux_test_acc": salux_acc,
    "droh_real2d_test_acc": droh_acc
}

with open(OUT_DIR / "summary_test.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("===== MULTIVIEW TEST SUMMARY (REAL DrOh) =====")
print("Experiment:", EXPERIMENT_NAME)
print("Selected views:", selected_views)
print("Salux test acc:", salux_acc)
print("DrOh REAL 2D test acc:", droh_acc)
print("Saved to:", OUT_DIR)