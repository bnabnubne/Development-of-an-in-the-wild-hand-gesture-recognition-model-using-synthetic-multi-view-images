import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = Path("./metadata/rgb_skeleton_multiview.csv")
CKPT_PATH = Path("./results/rgb_only/best_rgb_only.pt")
OUT_DIR = Path("./results/rgb_only_eval")

BATCH_SIZE = 32
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# "single_cam" or "avg_all_cams"
EVAL_MODE = "avg_all_cams"
SINGLE_CAM_ID = 0


class RGBEvalDataset(Dataset):
    def __init__(self, df, cam_cols, label_to_idx, image_size):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.label_to_idx = label_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        for col in self.cam_cols:
            image = Image.open(row[col]).convert("RGB")
            images.append(self.transform(image))

        images = torch.stack(images, dim=0)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return images, label


class RGBOnlyNet(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, image):
        return self.classifier(self.backbone(image))


def evaluate(model, loader, eval_mode, single_cam_id):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, label in loader:
            images = images.to(DEVICE)
            label = label.to(DEVICE)

            if eval_mode == "single_cam":
                logits = model(images[:, single_cam_id])
            elif eval_mode == "avg_all_cams":
                logits_sum = None
                for cam_idx in range(images.shape[1]):
                    logits_cam = model(images[:, cam_idx])
                    logits_sum = logits_cam if logits_sum is None else logits_sum + logits_cam
                logits = logits_sum / images.shape[1]
            else:
                raise ValueError(f"Unknown EVAL_MODE: {eval_mode}")

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(cm, class_names, out_path):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100.0

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

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"].copy()
    cam_cols = cfg["cam_cols"]

    ds = RGBEvalDataset(test_df, cam_cols, label_to_idx, cfg["image_size"])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = RGBOnlyNet(num_classes=len(label_to_idx)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, y_true, y_pred = evaluate(model, loader, EVAL_MODE, SINGLE_CAM_ID)
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

    suffix = EVAL_MODE if EVAL_MODE == "avg_all_cams" else f"cam{SINGLE_CAM_ID}"
    (OUT_DIR / f"report_{suffix}.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, OUT_DIR / f"confmat_{suffix}.png")

    summary = {
        "eval_mode": EVAL_MODE,
        "single_cam_id": SINGLE_CAM_ID,
        "test_acc": acc,
        "classes": label_to_idx,
        "csv_path": str(CSV_PATH),
        "ckpt_path": str(CKPT_PATH),
    }
    with open(OUT_DIR / f"summary_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== RGB-ONLY TEST SUMMARY =====")
    print("Eval mode:", EVAL_MODE)
    print("Test acc :", acc)
    print("Saved to :", OUT_DIR)


if __name__ == "__main__":
    main()
