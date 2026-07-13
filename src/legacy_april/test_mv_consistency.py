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
MERGE_THUMB = False   # True = 6 classes, False = 7 classes

if MERGE_THUMB:
    CKPT_PATH = "./results/mv_supcon_anchor_model_6cls/best_mv_supcon_anchor.pt"
    OUT_DIR = Path("./results/mv_supcon_anchor_test_6cls")

else:
    CKPT_PATH = "./results/mv_supcon_anchor_model_7cls/best_mv_supcon_anchor.pt"
    OUT_DIR = Path("./results/mv_supcon_anchor_test_7cls")

SALUX_MV_CSV = "./metadata/salux_multiview3d.csv"
SALUX_ORIG_CSV = "./metadata/salux_baseline.csv"
DROH_CSV = "./metadata/droh_baseline.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# =========================
# DATASETS
# =========================
class SaluxMultiViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cams = []
        for col in ["cam_0_path", "cam_1_path", "cam_2_path", "cam_3_path"]:
            x = np.load(row[col]).astype(np.float32)  # (21,3)
            x = torch.tensor(x, dtype=torch.float32)
            cams.append(x)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return cams[0], cams[1], cams[2], cams[3], y


class SingleView3DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["input_path"]).astype(np.float32)  # (21,3)
        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

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
# EVALUATION
# =========================
def evaluate_single_view(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def evaluate_salux_cam(model, loader, device, view_index=0):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for cam0, cam1, cam2, cam3, y in loader:
            cams = [cam0, cam1, cam2, cam3]
            x = cams[view_index].to(device)
            y = y.to(device)

            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def evaluate_salux_avg4(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for cam0, cam1, cam2, cam3, y in loader:
            cam0 = cam0.to(device)
            cam1 = cam1.to(device)
            cam2 = cam2.to(device)
            cam3 = cam3.to(device)
            y = y.to(device)

            logits0, _ = model(cam0)
            logits1, _ = model(cam1)
            logits2, _ = model(cam2)
            logits3, _ = model(cam3)

            logits = (logits0 + logits1 + logits2 + logits3) / 4.0
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
# LOAD CHECKPOINT
# =========================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

label_to_idx = ckpt["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}
cfg = ckpt["config"]

model = SingleViewGRU3D(
    input_dim=cfg.get("input_dim", 3),
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    num_classes=len(label_to_idx),
    dropout=cfg["dropout"]
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

print("DEVICE:", DEVICE)
print("MERGE_THUMB:", MERGE_THUMB)
print("CKPT_PATH:", CKPT_PATH)
print("SALUX_MV_CSV:", SALUX_MV_CSV)
print("SALUX_ORIG_CSV:", SALUX_ORIG_CSV)
print("DROH_CSV:", DROH_CSV)
print("Classes:", label_to_idx)

# =========================
# LOAD DATA
# =========================
salux_mv_df = pd.read_csv(SALUX_MV_CSV)
salux_orig_df = pd.read_csv(SALUX_ORIG_CSV)
droh_df = pd.read_csv(DROH_CSV)

if MERGE_THUMB:
    for df in [salux_mv_df, salux_orig_df, droh_df]:
        df["action"] = df["action"].replace({
            "thumbup": "thumb",
            "thumbdown": "thumb"
        })

salux_mv_test_df = salux_mv_df[salux_mv_df["split"] == "test"].copy()
salux_orig_test_df = salux_orig_df[salux_orig_df["split"] == "test"].copy()

salux_mv_ds = SaluxMultiViewDataset(salux_mv_test_df, label_to_idx=label_to_idx)
salux_orig_ds = SingleView3DDataset(salux_orig_test_df, label_to_idx=label_to_idx)
droh_ds = SingleView3DDataset(droh_df, label_to_idx=label_to_idx)

salux_mv_loader = DataLoader(salux_mv_ds, batch_size=BATCH_SIZE, shuffle=False)
salux_orig_loader = DataLoader(salux_orig_ds, batch_size=BATCH_SIZE, shuffle=False)
droh_loader = DataLoader(droh_ds, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# EVALUATE
# =========================
salux_orig_acc, y_true_salux_orig, y_pred_salux_orig = evaluate_single_view(
    model, salux_orig_loader, DEVICE
)

salux_cam0_acc, y_true_cam0, y_pred_cam0 = evaluate_salux_cam(
    model, salux_mv_loader, DEVICE, view_index=0
)

salux_cam1_acc, _, _ = evaluate_salux_cam(
    model, salux_mv_loader, DEVICE, view_index=1
)

salux_cam2_acc, _, _ = evaluate_salux_cam(
    model, salux_mv_loader, DEVICE, view_index=2
)

salux_cam3_acc, _, _ = evaluate_salux_cam(
    model, salux_mv_loader, DEVICE, view_index=3
)

salux_avg4_acc, _, _ = evaluate_salux_avg4(
    model, salux_mv_loader, DEVICE
)

droh_acc, y_true_droh, y_pred_droh = evaluate_single_view(
    model, droh_loader, DEVICE
)

# =========================
# REPORTS
# =========================
report_salux_orig = classification_report(
    y_true_salux_orig,
    y_pred_salux_orig,
    target_names=class_names,
    digits=4,
    zero_division=0
)

report_salux_cam0 = classification_report(
    y_true_cam0,
    y_pred_cam0,
    target_names=class_names,
    digits=4,
    zero_division=0
)

report_droh = classification_report(
    y_true_droh,
    y_pred_droh,
    target_names=class_names,
    digits=4,
    zero_division=0
)

cm_salux_orig = confusion_matrix(y_true_salux_orig, y_pred_salux_orig)
cm_salux_cam0 = confusion_matrix(y_true_cam0, y_pred_cam0)
cm_droh = confusion_matrix(y_true_droh, y_pred_droh)

(OUT_DIR / "report_salux_original.txt").write_text(report_salux_orig, encoding="utf-8")
(OUT_DIR / "report_salux_cam0.txt").write_text(report_salux_cam0, encoding="utf-8")
(OUT_DIR / "report_droh.txt").write_text(report_droh, encoding="utf-8")

save_confmat(
    cm_salux_orig,
    class_names,
    OUT_DIR / "confmat_salux_original.png",
    "MV-Consistency - Salux Original"
)

save_confmat(
    cm_salux_cam0,
    class_names,
    OUT_DIR / "confmat_salux_cam0.png",
    "MV-Consistency - Salux Cam0"
)

save_confmat(
    cm_droh,
    class_names,
    OUT_DIR / "confmat_droh.png",
    "MV-Consistency - DrOh"
)

summary = {
    "salux_original_acc": salux_orig_acc,
    "salux_cam0_acc": salux_cam0_acc,
    "salux_cam1_acc": salux_cam1_acc,
    "salux_cam2_acc": salux_cam2_acc,
    "salux_cam3_acc": salux_cam3_acc,
    "salux_avg4_acc": salux_avg4_acc,
    "droh_acc": droh_acc,
    "merge_thumb": MERGE_THUMB,
    "classes": label_to_idx
}

with open(OUT_DIR / "summary_test.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== MV-CONSISTENCY TEST SUMMARY =====")
print("Salux original acc:", salux_orig_acc)
print("Salux cam0 acc    :", salux_cam0_acc)
print("Salux cam1 acc    :", salux_cam1_acc)
print("Salux cam2 acc    :", salux_cam2_acc)
print("Salux cam3 acc    :", salux_cam3_acc)
print("Salux avg4 acc    :", salux_avg4_acc)
print("DrOh acc          :", droh_acc)
print("Saved to:", OUT_DIR)