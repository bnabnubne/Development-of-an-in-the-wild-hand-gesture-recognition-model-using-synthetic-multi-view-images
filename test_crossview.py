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
SALUX_CSV = "./metadata/salux_multiview.csv"
DROH_CSV  = "./metadata/droh_multiview.csv"

CKPT_ROOT = Path("./results/crossview")
OUT_ROOT = Path("./results/crossview_test")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

EXPERIMENTS = [
    "train_cam024_test_cam5",
    "train_cam025_test_cam4",
    "train_cam045_test_cam2",
    "train_cam245_test_cam0",
]

# =========================
# DATASET
# =========================
class CrossViewEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, view_col: str):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.view_col = view_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row[self.view_col]).astype(np.float32)   # (21,2)
        x = x.reshape(-1)                                    # (42,)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

# =========================
# MODEL
# =========================
class SingleViewGRU2D(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, num_layers=1, num_classes=7, dropout=0.0):
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
        return self.fc(feat)

# =========================
# UTILS
# =========================
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

# =========================
# LOAD CSV
# =========================
salux_df = pd.read_csv(SALUX_CSV)
salux_test_df = salux_df[salux_df["split"] == "test"].copy()

droh_df = pd.read_csv(DROH_CSV)

all_results = {}

for exp_name in EXPERIMENTS:
    print("\n" + "=" * 80)
    print("TEST EXPERIMENT:", exp_name)
    print("=" * 80)

    ckpt_path = CKPT_ROOT / exp_name / "best_crossview.pt"
    if not ckpt_path.exists():
        print(f"[SKIP] Missing checkpoint: {ckpt_path}")
        continue

    out_dir = OUT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    cfg = ckpt["config"]

    model = SingleViewGRU2D(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=len(label_to_idx),
        dropout=cfg["dropout"]
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])

    test_view = cfg["test_view"]
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # Salux
    salux_ds = CrossViewEvalDataset(salux_test_df, label_to_idx=label_to_idx, view_col=test_view)
    salux_loader = DataLoader(salux_ds, batch_size=BATCH_SIZE, shuffle=False)

    salux_acc, y_true_salux, y_pred_salux = evaluate(model, salux_loader, DEVICE)
    report_salux = classification_report(y_true_salux, y_pred_salux, target_names=class_names, digits=4, zero_division=0)
    cm_salux = confusion_matrix(y_true_salux, y_pred_salux)

    (out_dir / "report_salux.txt").write_text(report_salux, encoding="utf-8")
    save_confmat(cm_salux, class_names, out_dir / "confmat_salux.png", f"{exp_name} - Salux")

    # DrOh
    droh_ds = CrossViewEvalDataset(droh_df, label_to_idx=label_to_idx, view_col=test_view)
    droh_loader = DataLoader(droh_ds, batch_size=BATCH_SIZE, shuffle=False)

    droh_acc, y_true_droh, y_pred_droh = evaluate(model, droh_loader, DEVICE)
    report_droh = classification_report(y_true_droh, y_pred_droh, target_names=class_names, digits=4, zero_division=0)
    cm_droh = confusion_matrix(y_true_droh, y_pred_droh)

    (out_dir / "report_droh.txt").write_text(report_droh, encoding="utf-8")
    save_confmat(cm_droh, class_names, out_dir / "confmat_droh.png", f"{exp_name} - DrOh")

    summary = {
        "experiment": exp_name,
        "train_views": cfg["train_views"],
        "test_view": test_view,
        "salux_test_acc": salux_acc,
        "droh_test_acc": droh_acc,
    }

    with open(out_dir / "summary_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    all_results[exp_name] = summary

    print("Salux test acc:", salux_acc)
    print("DrOh  test acc:", droh_acc)

with open(OUT_ROOT / "summary_all_test.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\nDONE")
print("Saved to:", OUT_ROOT)