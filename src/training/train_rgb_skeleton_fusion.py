import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CSV_PATH = Path("./metadata/rgb_skeleton_multiview.csv")
OUT_DIR = Path("./results/rgb_skeleton_fusion")

MERGE_THUMB = False

BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
MIN_DELTA = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

IMAGE_SIZE = 160
RGB_FEATURE_DIM = 256
SKEL_HIDDEN_DIM = 128
FUSION_DIM = 256
DROPOUT = 0.2

TRAIN_RGB_MODE = "random_cam"

EVAL_CAM_ID = 0

SKELETON_MODE = "view_aligned"

USE_IMAGENET_WEIGHTS = False
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RGBSkeletonDataset(Dataset):
    def __init__(
        self,
        df,
        cam_cols,
        label_to_idx=None,
        mode="eval",
        train_rgb_mode="random_cam",
        eval_cam_id=0,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.mode = mode
        self.train_rgb_mode = train_rgb_mode
        self.eval_cam_id = eval_cam_id
        self.transform = transform
        self.skeleton_mode = SKELETON_MODE

        if label_to_idx is None:
            labels = sorted(self.df["action"].unique().tolist())
            self.label_to_idx = {label: i for i, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.items = []
        if mode == "train" and train_rgb_mode == "all_cams":
            for row_idx in range(len(self.df)):
                for cam_idx in range(len(self.cam_cols)):
                    self.items.append((row_idx, cam_idx))
        else:
            for row_idx in range(len(self.df)):
                self.items.append((row_idx, None))

    def __len__(self):
        return len(self.items)

    def choose_cam_idx(self, fixed_cam):
        if fixed_cam is not None:
            return fixed_cam
        if self.mode == "train" and self.train_rgb_mode == "random_cam":
            return random.randrange(len(self.cam_cols))
        return self.eval_cam_id

    def __getitem__(self, idx):
        row_idx, fixed_cam = self.items[idx]
        row = self.df.iloc[row_idx]
        cam_idx = self.choose_cam_idx(fixed_cam)
        rgb_path = row[self.cam_cols[cam_idx]]

        image = Image.open(rgb_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.skeleton_mode == "view_aligned":
            skeleton_path = row[f"skeleton_cam_{cam_idx}_path"]
        elif self.skeleton_mode == "original":
            skeleton_path = row["skeleton_orig_path"] if "skeleton_orig_path" in row else row["skeleton_path"]
        else:
            raise ValueError(f"Unknown SKELETON_MODE: {self.skeleton_mode}")

        skeleton = np.load(skeleton_path).astype(np.float32)
        skeleton = torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32)

        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, skeleton, label


class SkeletonEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]


class RGBEncoder(nn.Module):
    def __init__(self, out_dim=256, use_imagenet_weights=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        backbone = models.resnet18(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class RGBSkeletonFusionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_encoder = RGBEncoder(RGB_FEATURE_DIM, USE_IMAGENET_WEIGHTS)
        self.skel_encoder = SkeletonEncoder(3, SKEL_HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(RGB_FEATURE_DIM + SKEL_HIDDEN_DIM, FUSION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_DIM, num_classes),
        )

    def forward(self, image, skeleton):
        rgb_feat = self.rgb_encoder(image)
        skel_feat = self.skel_encoder(skeleton)
        fused = torch.cat([rgb_feat, skel_feat], dim=1)
        return self.classifier(fused)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


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

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            if value == 0:
                continue
            ax.text(j, i, f"{value:.1f}", ha="center", va="center")

    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    cam_cols = sorted(
        [c for c in df.columns if c.startswith("rgb_cam_") and c.endswith("_path")],
        key=lambda x: int(x.split("_")[2]),
    )

    if MERGE_THUMB:
        df["action"] = df["action"].replace({"thumbup": "thumb", "thumbdown": "thumb"})

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_tf, eval_tf = build_transforms()
    train_ds = RGBSkeletonDataset(
        train_df,
        cam_cols,
        mode="train",
        train_rgb_mode=TRAIN_RGB_MODE,
        eval_cam_id=EVAL_CAM_ID,
        transform=train_tf,
    )
    label_to_idx = train_ds.label_to_idx
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    val_ds = RGBSkeletonDataset(
        val_df,
        cam_cols,
        label_to_idx=label_to_idx,
        mode="eval",
        eval_cam_id=EVAL_CAM_ID,
        transform=eval_tf,
    )
    test_ds = RGBSkeletonDataset(
        test_df,
        cam_cols,
        label_to_idx=label_to_idx,
        mode="eval",
        eval_cam_id=EVAL_CAM_ID,
        transform=eval_tf,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("DEVICE:", DEVICE, flush=True)
    print("CSV:", CSV_PATH, flush=True)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds), flush=True)
    print("Classes:", label_to_idx, flush=True)
    print("Cam cols:", cam_cols, flush=True)

    model = RGBSkeletonFusionNet(num_classes=len(label_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_path = OUT_DIR / "best_rgb_skeleton_fusion.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for image, skeleton, label in train_loader:
            image = image.to(DEVICE)
            skeleton = skeleton.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            logits = model(image, skeleton)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc, _, _ = evaluate(model, val_loader)

        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "rgb_feature_dim": RGB_FEATURE_DIM,
                    "skel_hidden_dim": SKEL_HIDDEN_DIM,
                    "fusion_dim": FUSION_DIM,
                    "dropout": DROPOUT,
                    "cam_cols": cam_cols,
                    "eval_cam_id": EVAL_CAM_ID,
                    "merge_thumb": MERGE_THUMB,
                    "train_rgb_mode": TRAIN_RGB_MODE,
                    "skeleton_mode": SKELETON_MODE,
                },
            }, best_path)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | loss={running_loss:.4f} | "
            f"val_acc={val_acc:.4f} | best={best_val_acc:.4f}",
            flush=True,
        )

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    test_acc, y_true, y_pred = evaluate(model, test_loader)
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

    (OUT_DIR / "report_test.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, OUT_DIR / "confmat_test.png")

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "classes": label_to_idx,
        "csv_path": str(CSV_PATH),
        "train_rgb_mode": TRAIN_RGB_MODE,
        "eval_cam_id": EVAL_CAM_ID,
        "skeleton_mode": SKELETON_MODE,
    }
    with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== TRAIN SUMMARY =====", flush=True)
    print("Best val acc:", best_val_acc, flush=True)
    print("Test acc    :", test_acc, flush=True)
    print("Saved model :", best_path, flush=True)
    print("Saved to    :", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
