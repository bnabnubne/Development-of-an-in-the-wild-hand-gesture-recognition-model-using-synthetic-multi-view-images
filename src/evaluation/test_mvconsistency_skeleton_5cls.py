import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


APRIL_ROOT = Path(".")
MODEL_ROOT = Path(".")

SALUX_ORIG_CSV = APRIL_ROOT / "metadata" / "salux_baseline.csv"
SALUX_MV_CSV = APRIL_ROOT / "metadata" / "salux_multiview3d_8cam_front.csv"
DROH_CSV = MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv"

CKPT_PATH = (
    APRIL_ROOT
    / "results"
    / "mv_consistency_anchor_8cam_model_lambda0.3"
    / "best_mv_consistency_anchor.pt"
)
OUT_DIR = MODEL_ROOT / "results" / "mvconsistency_skeleton_5cls"

CLASSES_5 = ["ok", "paper", "rock", "scissors", "the-finger"]
BATCH_SIZE = 64
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class SingleViewSkeletonDataset(Dataset):
    def __init__(self, df, path_col, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row[self.path_col]).astype(np.float32)
        x = torch.tensor(x.reshape(21, 3), dtype=torch.float32)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y


class MultiViewSkeletonDataset(Dataset):
    def __init__(self, df, cam_cols, label_to_idx):
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
            views.append(torch.tensor(x.reshape(21, 3), dtype=torch.float32))

        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return views, y


def collate_multiview(batch):
    views_list, y_list = zip(*batch)
    y = torch.stack(y_list, dim=0)
    num_views = len(views_list[0])
    views = []
    for i in range(num_views):
        views.append(torch.stack([sample_views[i] for sample_views in views_list], dim=0))
    return views, y


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


def evaluate_single(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits, _ = model(x)
            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def evaluate_multiview_avg(model, loader, num_views):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for views, y in loader:
            y = y.to(DEVICE)
            logits_sum = None

            for i in range(num_views):
                x = views[i].to(DEVICE)
                logits, _ = model(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            pred = torch.argmax(logits_sum / num_views, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(y_true, y_pred, true_label_ids, pred_label_ids, true_names, pred_names, out_path):
    import matplotlib.pyplot as plt

    cm_full = confusion_matrix(y_true, y_pred, labels=pred_label_ids)
    cm = cm_full[true_label_ids, :]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(pred_names)))
    ax.set_yticks(np.arange(len(true_names)))
    ax.set_xticklabels(pred_names, rotation=45, ha="right")
    ax.set_yticklabels(true_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_ylim(len(true_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(name, acc, y_true, y_pred, class_names_all, label_to_idx_all, rows):
    true_label_ids = [label_to_idx_all[name] for name in CLASSES_5]
    pred_label_ids = list(range(len(class_names_all)))
    report = classification_report(
        y_true,
        y_pred,
        labels=true_label_ids,
        target_names=CLASSES_5,
        digits=4,
        zero_division=0,
    )
    (OUT_DIR / f"report_{name}.txt").write_text(report, encoding="utf-8")
    save_confmat(
        y_true,
        y_pred,
        true_label_ids,
        pred_label_ids,
        CLASSES_5,
        class_names_all,
        OUT_DIR / f"confmat_{name}.png",
    )

    summary = {
        "test_acc": acc,
        "rows": rows,
        "classes": label_to_idx_all,
        "evaluated_classes": CLASSES_5,
        "ckpt_path": str(CKPT_PATH),
    }
    with open(OUT_DIR / f"summary_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    label_to_idx_all = ckpt["label_to_idx"]
    cfg = ckpt["config"]

    label_to_idx = {name: label_to_idx_all[name] for name in CLASSES_5}
    idx_to_label_all = {v: k for k, v in label_to_idx_all.items()}
    class_names_all = [idx_to_label_all[i] for i in range(len(idx_to_label_all))]

    model = SingleViewGRU3D(
        input_dim=cfg.get("input_dim", 3),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=len(label_to_idx_all),
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    salux_orig_df = pd.read_csv(SALUX_ORIG_CSV)
    salux_orig_df = salux_orig_df[
        (salux_orig_df["split"] == "test") & salux_orig_df["action"].isin(CLASSES_5)
    ].copy()

    salux_mv_df = pd.read_csv(SALUX_MV_CSV)
    salux_mv_df = salux_mv_df[
        (salux_mv_df["split"] == "test") & salux_mv_df["action"].isin(CLASSES_5)
    ].copy()
    cam_cols = sorted(
        [c for c in salux_mv_df.columns if c.startswith("cam_") and c.endswith("_path")],
        key=lambda x: int(x.split("_")[1]),
    )

    droh_df = pd.read_csv(DROH_CSV)
    droh_df = droh_df[droh_df["action"].isin(CLASSES_5)].copy()

    salux_orig_loader = DataLoader(
        SingleViewSkeletonDataset(salux_orig_df, "input_path", label_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    salux_mv_loader = DataLoader(
        MultiViewSkeletonDataset(salux_mv_df, cam_cols, label_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_multiview,
    )
    droh_loader = DataLoader(
        SingleViewSkeletonDataset(droh_df, "skeleton_path", label_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    salux_orig_acc, salux_orig_true, salux_orig_pred = evaluate_single(model, salux_orig_loader)
    salux_avg8_acc, salux_avg8_true, salux_avg8_pred = evaluate_multiview_avg(
        model,
        salux_mv_loader,
        len(cam_cols),
    )
    droh_acc, droh_true, droh_pred = evaluate_single(model, droh_loader)

    summaries = {
        "salux_original_5cls": write_report(
            "salux_original_5cls",
            salux_orig_acc,
            salux_orig_true,
            salux_orig_pred,
            class_names_all,
            label_to_idx_all,
            len(salux_orig_df),
        ),
        "salux_avg8_5cls": write_report(
            "salux_avg8_5cls",
            salux_avg8_acc,
            salux_avg8_true,
            salux_avg8_pred,
            class_names_all,
            label_to_idx_all,
            len(salux_mv_df),
        ),
        "droh_5cls": write_report(
            "droh_5cls",
            droh_acc,
            droh_true,
            droh_pred,
            class_names_all,
            label_to_idx_all,
            len(droh_df),
        ),
    }

    summary_all = {
        "device": DEVICE,
        "classes_5": CLASSES_5,
        "ckpt_path": str(CKPT_PATH),
        "salux_original_5cls_acc": salux_orig_acc,
        "salux_avg8_5cls_acc": salux_avg8_acc,
        "droh_5cls_acc": droh_acc,
        "rows": {
            "salux_original_5cls": len(salux_orig_df),
            "salux_avg8_5cls": len(salux_mv_df),
            "droh_5cls": len(droh_df),
        },
        "details": summaries,
    }
    with open(OUT_DIR / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)

    print("===== MV-Consistency Skeleton 5-Class Eval =====")
    print("Device:", DEVICE)
    print("Salux original rows:", len(salux_orig_df), "acc:", salux_orig_acc)
    print("Salux avg8 rows    :", len(salux_mv_df), "acc:", salux_avg8_acc)
    print("DrOh rows          :", len(droh_df), "acc:", droh_acc)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
