import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# =========================================================
# CONFIG
# =========================================================
DROH_CSV = Path("./metadata/droh_rgb_skeleton_7cls.csv")
CKPT_PATH = Path(
    "./results/mv_consistency_anchor_8cam_model_lambda0.3/"
    "best_mv_consistency_anchor.pt"
)
OUT_DIR = Path("./results/droh_mvconsistency_skeleton")

BATCH_SIZE = 64
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class DrOhSkeletonDataset(Dataset):
    def __init__(self, df, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["skeleton_path"]).astype(np.float32)
        x = torch.tensor(x.reshape(21, 3), dtype=torch.float32)
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
        out, _ = self.gru(x)
        z = out[:, -1, :]
        logits = self.fc(z)
        return logits, z


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(cm, class_names, out_path):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    cfg = ckpt["config"]

    df = pd.read_csv(DROH_CSV)
    df = df[df["action"].isin(label_to_idx)].copy()

    ds = DrOhSkeletonDataset(df, label_to_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SingleViewGRU3D(
        input_dim=cfg.get("input_dim", 3),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=len(label_to_idx),
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, y_true, y_pred = evaluate(model, loader)
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    (OUT_DIR / "report_droh.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, OUT_DIR / "confmat_droh.png")
    summary = {
        "test_acc": acc,
        "rows": len(df),
        "classes": label_to_idx,
        "csv_path": str(DROH_CSV),
        "ckpt_path": str(CKPT_PATH),
    }
    with open(OUT_DIR / "summary_droh.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== MV-Consistency Skeleton DrOh =====")
    print("Rows:", len(df))
    print("Test acc:", acc)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
