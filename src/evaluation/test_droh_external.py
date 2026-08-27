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


DROH_CSV = Path("./metadata/droh_rgb_skeleton_5cls.csv")
OUT_DIR = Path("./results/droh_external")

RGB_CKPT = Path("./results/rgb_only/best_rgb_only.pt")
FUSION_CKPT = Path("./results/rgb_skeleton_fusion/best_rgb_skeleton_fusion.pt")

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


class SkeletonEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, skeleton):
        out, _ = self.gru(skeleton)
        return out[:, -1, :]


class FusionNet(nn.Module):
    def __init__(self, num_classes, rgb_feature_dim, skel_hidden_dim, fusion_dim, dropout):
        super().__init__()
        self.rgb_encoder = RGBEncoder(rgb_feature_dim)
        self.skel_encoder = SkeletonEncoder(3, skel_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feature_dim + skel_hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image, skeleton):
        rgb_feat = self.rgb_encoder(image)
        skel_feat = self.skel_encoder(skeleton)
        return self.classifier(torch.cat([rgb_feat, skel_feat], dim=1))


def evaluate(model, loader, mode):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for image, skeleton, label in loader:
            image = image.to(DEVICE)
            skeleton = skeleton.to(DEVICE)
            label = label.to(DEVICE)

            if mode == "rgb":
                logits = model(image)
            elif mode == "fusion":
                logits = model(image, skeleton)
            else:
                raise ValueError(mode)

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


def write_result(name, acc, y_true, y_pred, class_names, label_to_idx):
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
    (OUT_DIR / f"report_{name}.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, OUT_DIR / f"confmat_{name}.png")
    summary = {
        "model": name,
        "test_acc": acc,
        "classes": label_to_idx,
        "csv_path": str(DROH_CSV),
    }
    with open(OUT_DIR / f"summary_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DROH_CSV)

    rgb_ckpt = torch.load(RGB_CKPT, map_location=DEVICE)
    fusion_ckpt = torch.load(FUSION_CKPT, map_location=DEVICE)

    label_to_idx = rgb_ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    df = df[df["action"].isin(label_to_idx)].copy()

    image_size = rgb_ckpt["config"]["image_size"]
    ds = DrOhDataset(df, label_to_idx, image_size)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    rgb_model = RGBOnlyNet(num_classes=len(label_to_idx)).to(DEVICE)
    rgb_model.load_state_dict(rgb_ckpt["model_state_dict"])
    rgb_acc, rgb_true, rgb_pred = evaluate(rgb_model, loader, "rgb")
    write_result("rgb_only_real_rgb", rgb_acc, rgb_true, rgb_pred, class_names, label_to_idx)

    fu_cfg = fusion_ckpt["config"]
    fusion_model = FusionNet(
        num_classes=len(label_to_idx),
        rgb_feature_dim=fu_cfg["rgb_feature_dim"],
        skel_hidden_dim=fu_cfg["skel_hidden_dim"],
        fusion_dim=fu_cfg["fusion_dim"],
        dropout=fu_cfg["dropout"],
    ).to(DEVICE)
    fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
    fu_acc, fu_true, fu_pred = evaluate(fusion_model, loader, "fusion")
    write_result("fusion_real_rgb_real_skeleton", fu_acc, fu_true, fu_pred, class_names, label_to_idx)

    combined = {
        "rows": len(df),
        "classes": label_to_idx,
        "rgb_only_real_rgb": rgb_acc,
        "fusion_real_rgb_real_skeleton": fu_acc,
        "note": (
            "Skeleton-only baseline is evaluated with MV-Consistency in "
            "test_droh_mvconsistency_skeleton.py."
        ),
    }
    with open(OUT_DIR / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("===== DrOh External Summary =====")
    print("Rows:", len(df))
    print("RGB-only:", rgb_acc)
    print("Fusion:", fu_acc)
    print("Skeleton-only baseline: use test_droh_mvconsistency_skeleton.py")
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
