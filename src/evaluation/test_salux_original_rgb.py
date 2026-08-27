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


APRIL_ROOT = Path(".")
MODEL_ROOT = Path(".")
SALUX_IMAGE_ROOT = Path("./data/raw/FullLabelled")

SALUX_BASELINE_CSV = APRIL_ROOT / "metadata" / "salux_baseline.csv"
OUT_METADATA = MODEL_ROOT / "metadata" / "salux_original_rgb_5cls.csv"
OUT_DIR = MODEL_ROOT / "results" / "salux_original_rgb"

RGB_CKPT = MODEL_ROOT / "results" / "rgb_only" / "best_rgb_only.pt"
FUSION_CKPT = MODEL_ROOT / "results" / "rgb_mvconsistency_fusion" / "best_rgb_mvconsistency_fusion.pt"

CLASSES_5 = ["ok", "paper", "rock", "scissors", "the-finger"]
BATCH_SIZE = 32
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def find_image(action, sample_id):
    class_dir = SALUX_IMAGE_ROOT / action
    for suffix in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        path = class_dir / f"{sample_id}{suffix}"
        if path.exists():
            return path
    return None


def build_metadata():
    df = pd.read_csv(SALUX_BASELINE_CSV)
    df = df[(df["split"] == "test") & df["action"].isin(CLASSES_5)].copy()

    rows = []
    missing_images = 0
    missing_skeleton = 0

    for _, row in df.iterrows():
        image_path = find_image(row["action"], row["sample_id"])
        skeleton_path = Path(row["input_path"])

        if image_path is None:
            missing_images += 1
            continue
        if not skeleton_path.exists():
            missing_skeleton += 1
            continue

        rows.append({
            "action": row["action"],
            "sample_id": row["sample_id"],
            "split": "test",
            "image_path": str(image_path),
            "skeleton_path": str(skeleton_path),
        })

    out_df = pd.DataFrame(rows)
    OUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_METADATA, index=False)

    return out_df, {
        "source_rows": len(df),
        "rows": len(out_df),
        "missing_images": missing_images,
        "missing_skeleton": missing_skeleton,
        "metadata_path": str(OUT_METADATA),
    }


class SaluxOriginalRGBDataset(Dataset):
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


class MVConsistencySkeletonEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
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


def evaluate_rgb(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for image, _, label in loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            logits = model(image)
            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())
    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def evaluate_fusion(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for image, skeleton, label in loader:
            image = image.to(DEVICE)
            skeleton = skeleton.to(DEVICE)
            label = label.to(DEVICE)
            logits = model(image, skeleton)
            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
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


def write_result(name, acc, y_true, y_pred, class_names, label_to_idx, metadata_info, ckpt_path):
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
        "metadata": metadata_info,
        "ckpt_path": str(ckpt_path),
    }
    with open(OUT_DIR / f"summary_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata_info = build_metadata()
    if len(df) == 0:
        raise RuntimeError("No Salux original RGB samples found.")

    rgb_ckpt = torch.load(RGB_CKPT, map_location=DEVICE)
    fusion_ckpt = torch.load(FUSION_CKPT, map_location=DEVICE)

    label_to_idx = rgb_ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    df = df[df["action"].isin(label_to_idx)].copy()

    image_size = rgb_ckpt["config"]["image_size"]
    loader = DataLoader(
        SaluxOriginalRGBDataset(df, label_to_idx, image_size),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    rgb_model = RGBOnlyNet(num_classes=len(label_to_idx)).to(DEVICE)
    rgb_model.load_state_dict(rgb_ckpt["model_state_dict"])
    rgb_acc, rgb_true, rgb_pred = evaluate_rgb(rgb_model, loader)
    rgb_summary = write_result(
        "rgb_only",
        rgb_acc,
        rgb_true,
        rgb_pred,
        class_names,
        label_to_idx,
        metadata_info,
        RGB_CKPT,
    )

    fu_cfg = fusion_ckpt["config"]
    fusion_model = FusionNet(
        num_classes=len(label_to_idx),
        rgb_feature_dim=fu_cfg["rgb_feature_dim"],
        skeleton_hidden_dim=fu_cfg["skeleton_hidden_dim"],
        fusion_dim=fu_cfg["fusion_dim"],
        dropout=fu_cfg["dropout"],
    ).to(DEVICE)
    fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
    fusion_acc, fusion_true, fusion_pred = evaluate_fusion(fusion_model, loader)
    fusion_summary = write_result(
        "rgb_mvconsistency_fusion",
        fusion_acc,
        fusion_true,
        fusion_pred,
        class_names,
        label_to_idx,
        metadata_info,
        FUSION_CKPT,
    )

    combined = {
        "dataset": "Salux original RGB test split",
        "rows": len(df),
        "classes": label_to_idx,
        "metadata": metadata_info,
        "rgb_only": rgb_acc,
        "rgb_mvconsistency_fusion": fusion_acc,
        "details": {
            "rgb_only": rgb_summary,
            "rgb_mvconsistency_fusion": fusion_summary,
        },
    }
    with open(OUT_DIR / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("===== Salux Original RGB External Eval =====")
    print("Rows:", len(df))
    print("RGB-only:", rgb_acc)
    print("RGB + MV-Consistency fusion:", fusion_acc)
    print("Saved metadata:", OUT_METADATA)
    print("Saved results:", OUT_DIR)


if __name__ == "__main__":
    main()
