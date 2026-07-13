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
DROH_CSV = Path("./metadata/droh_rgb_skeleton_5cls.csv")
FUSION_CKPT = Path(
    "./results/rgb_mvconsistency_fusion/"
    "best_rgb_mvconsistency_fusion.pt"
)
OUT_DIR = Path("./results/droh_mvconsistency_fusion")

BATCH_SIZE = 32
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class DrOhDataset(Dataset):
    def __init__(self, df, label_to_idx, image_size):
        self.df = df.reset_index(drop=True)
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
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        skeleton = np.load(row["skeleton_path"]).astype(np.float32)
        skeleton = torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, skeleton, label


class RGBEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        return self.proj(self.backbone(image))


class MVConsistencySkeletonEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, skeleton):
        out, _ = self.gru(skeleton)
        return out[:, -1, :]


class FusionNet(nn.Module):
    def __init__(self, num_classes, rgb_feature_dim, skeleton_hidden_dim, fusion_dim, dropout):
        super().__init__()
        self.rgb_encoder = RGBEncoder(rgb_feature_dim)
        self.skeleton_encoder = MVConsistencySkeletonEncoder(skeleton_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feature_dim + skeleton_hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image, skeleton):
        rgb_feat = self.rgb_encoder(image)
        skel_feat = self.skeleton_encoder(skeleton)
        return self.classifier(torch.cat([rgb_feat, skel_feat], dim=1))


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for image, skeleton, label in loader:
            image = image.to(DEVICE)
            skeleton = skeleton.to(DEVICE)
            label = label.to(DEVICE)
            logits = model(image, skeleton)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())
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
    ckpt = torch.load(FUSION_CKPT, map_location=DEVICE)
    cfg = ckpt["config"]
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    df = pd.read_csv(DROH_CSV)
    df = df[df["action"].isin(label_to_idx)].copy()
    ds = DrOhDataset(df, label_to_idx, cfg["image_size"])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = FusionNet(
        num_classes=len(label_to_idx),
        rgb_feature_dim=cfg["rgb_feature_dim"],
        skeleton_hidden_dim=cfg["skeleton_hidden_dim"],
        fusion_dim=cfg["fusion_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, y_true, y_pred = evaluate(model, loader)
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
        "fusion_ckpt": str(FUSION_CKPT),
        "skeleton_input_mode": cfg["skeleton_input_mode"],
        "freeze_skeleton_encoder": cfg["freeze_skeleton_encoder"],
        "mvconsistency_ckpt": cfg["mvconsistency_ckpt"],
    }
    with open(OUT_DIR / "summary_droh.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== DrOh RGB + MV-Consistency Fusion Summary =====")
    print("Rows:", len(df))
    print("Test acc:", acc)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
