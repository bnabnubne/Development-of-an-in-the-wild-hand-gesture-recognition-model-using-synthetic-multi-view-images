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
MERGE_THUMB = False
POOLING = "last"
EXPERIMENT = "consistency"
# options:
# "consistency"
# "supcon"

if EXPERIMENT == "consistency":
    if MERGE_THUMB:
        CKPT_PATH = "./results/mv_consistency_anchor_8cam_model_6cls/best_mv_consistency_anchor.pt"
        OUT_DIR = Path("./results/mv_consistency_anchor_8cam_test_6cls")
    else:
        CKPT_PATH = "./results/mv_consistency_anchor_8cam_model_lambda0.3/best_mv_consistency_anchor.pt"
        OUT_DIR = Path("./results/mv_consistency_anchor_8cam_test_7cls_lambda0.3")

elif EXPERIMENT == "supcon":
    if MERGE_THUMB:
        CKPT_PATH = "./results/mv_supcon_anchor_8cam_model_6cls/best_mv_supcon_anchor.pt"
        OUT_DIR = Path("./results/mv_supcon_anchor_8cam_test_6cls")
    else:
        CKPT_PATH = "./results/mv_supcon_anchor_8cam_model_7cls_alpha1.0/best_mv_supcon_anchor.pt"
        OUT_DIR = Path("./results/mv_supcon_anchor_8cam_test_7cls_alpha1.0")

else:
    raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

SALUX_MV_CSV = "./metadata/salux_multiview3d_8cam_front.csv"
SALUX_ORIG_CSV = "./metadata/salux_baseline.csv"
DROH_CSV = "./metadata/droh_baseline.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# =========================
# DATASETS
# =========================
class SaluxMultiViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cam_cols, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        views = []
        for col in self.cam_cols:
            x = np.load(row[col]).astype(np.float32)
            x = torch.tensor(x, dtype=torch.float32)
            views.append(x)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return views, y


def collate_mv(batch):
    views_list, y_list = zip(*batch)

    y = torch.stack(y_list, dim=0)

    num_views = len(views_list[0])
    views = []
    for i in range(num_views):
        views.append(torch.stack([v[i] for v in views_list], dim=0))

    return views, y


class SingleView3DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["input_path"]).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

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
        pooling="last"  # "last", "mean", "attention"
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


def evaluate_salux_each_cam(model, loader, device, num_cams):
    accs = {}

    for cam_idx in range(num_cams):
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for views, y in loader:
                x = views[cam_idx].to(device)
                y = y.to(device)

                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())

        accs[f"cam{cam_idx}"] = accuracy_score(all_targets, all_preds)

    return accs


def evaluate_salux_avgN(model, loader, device, num_cams):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for views, y in loader:
            y = y.to(device)

            logits_sum = None

            for i in range(num_cams):
                x = views[i].to(device)
                logits, _ = model(x)

                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

            logits_avg = logits_sum / num_cams
            preds = torch.argmax(logits_avg, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(cm, class_names, out_path, normalize=True):
    """
    Save confusion matrix as percentage heatmap.
    Rows = actual labels, columns = predicted labels.
    If normalize=True, each row sums to 100%.
    """
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
 
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("Actual", fontsize=15)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=18)
    ax.set_yticklabels(class_names, fontsize=18)

    # Draw values inside cells
    threshold = data.max() * 0.55 if data.max() > 0 else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]

            # Hide zero values for cleaner paper figure
            if value == 0:
                continue

            text_color = "white" if value > threshold else "black"
            ax.text(
                j, i,
                fmt.format(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=18
            )

    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
    dropout=cfg["dropout"],
    pooling=POOLING
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

# =========================
# LOAD DATA
# =========================
salux_mv_df = pd.read_csv(SALUX_MV_CSV)
salux_orig_df = pd.read_csv(SALUX_ORIG_CSV)
droh_df = pd.read_csv(DROH_CSV)

cam_cols = sorted(
    [c for c in salux_mv_df.columns if c.startswith("cam_") and c.endswith("_path")],
    key=lambda x: int(x.split("_")[1])
)

if MERGE_THUMB:
    for df in [salux_mv_df, salux_orig_df, droh_df]:
        df["action"] = df["action"].replace({
            "thumbup": "thumb",
            "thumbdown": "thumb"
        })

salux_mv_test_df = salux_mv_df[salux_mv_df["split"] == "test"].copy()
salux_orig_test_df = salux_orig_df[salux_orig_df["split"] == "test"].copy()

salux_mv_ds = SaluxMultiViewDataset(salux_mv_test_df, cam_cols, label_to_idx)
salux_orig_ds = SingleView3DDataset(salux_orig_test_df, label_to_idx)
droh_ds = SingleView3DDataset(droh_df, label_to_idx)

salux_mv_loader = DataLoader(salux_mv_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mv)
salux_orig_loader = DataLoader(salux_orig_ds, batch_size=BATCH_SIZE, shuffle=False)
droh_loader = DataLoader(droh_ds, batch_size=BATCH_SIZE, shuffle=False)

print("DEVICE:", DEVICE)
print("EXPERIMENT:", EXPERIMENT)
print("MERGE_THUMB:", MERGE_THUMB)
print("CKPT_PATH:", CKPT_PATH)
print("CAM_COLS:", cam_cols)
print("Classes:", label_to_idx)

# =========================
# EVALUATE
# =========================
salux_orig_acc, y_true_salux_orig, y_pred_salux_orig = evaluate_single_view(
    model, salux_orig_loader, DEVICE
)

cam_accs = evaluate_salux_each_cam(
    model, salux_mv_loader, DEVICE, num_cams=len(cam_cols)
)

salux_avgN_acc, _, _ = evaluate_salux_avgN(
    model, salux_mv_loader, DEVICE, num_cams=len(cam_cols)
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

report_droh = classification_report(
    y_true_droh,
    y_pred_droh,
    target_names=class_names,
    digits=4,
    zero_division=0
)

cm_salux_orig = confusion_matrix(y_true_salux_orig, y_pred_salux_orig)
cm_droh = confusion_matrix(y_true_droh, y_pred_droh)

(OUT_DIR / "report_salux_original.txt").write_text(report_salux_orig, encoding="utf-8")
(OUT_DIR / "report_droh.txt").write_text(report_droh, encoding="utf-8")

save_confmat(
    cm_salux_orig,
    class_names,
    OUT_DIR / "confmat_salux_original.png"
)

save_confmat(
    cm_droh,
    class_names,
    OUT_DIR / "confmat_droh.png"
)

summary = {
    "experiment": EXPERIMENT,
    "merge_thumb": MERGE_THUMB,
    "salux_original_acc": salux_orig_acc,
    "salux_avgN_acc": salux_avgN_acc,
    "droh_acc": droh_acc,
    "num_cameras": len(cam_cols),
    "cam_accs": cam_accs,
    "classes": label_to_idx
}

with open(OUT_DIR / "summary_test.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== MV 8CAM TEST SUMMARY =====")
print("Experiment        :", EXPERIMENT)
print("MERGE_THUMB       :", MERGE_THUMB)
print("Salux original acc:", salux_orig_acc)
for k, v in cam_accs.items():
    print(f"Salux {k} acc     :", v)
print("Salux avgN acc    :", salux_avgN_acc)
print("DrOh acc          :", droh_acc)
print("Saved to:", OUT_DIR)