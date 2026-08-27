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
MVCONSISTENCY_CKPT = Path(
    "./results/mv_consistency_anchor_8cam_model_lambda0.3/"
    "best_mv_consistency_anchor.pt"
)
OUT_DIR = Path("./results/rgb_mvconsistency_fusion")

BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
MIN_DELTA = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

IMAGE_SIZE = 160
RGB_FEATURE_DIM = 256
FUSION_DIM = 256
DROPOUT = 0.2

TRAIN_RGB_MODE = "random_cam"

SKELETON_INPUT_MODE = "original"
FREEZE_SKELETON_ENCODER = True

EVAL_CAM_ID = 0
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
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.mode = mode
        self.transform = transform

        if label_to_idx is None:
            labels = sorted(self.df["action"].unique().tolist())
            self.label_to_idx = {label: i for i, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def choose_cam_idx(self):
        if self.mode == "train" and TRAIN_RGB_MODE == "random_cam":
            return random.randrange(len(self.cam_cols))
        return EVAL_CAM_ID

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cam_idx = self.choose_cam_idx()

        image = Image.open(row[self.cam_cols[cam_idx]]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if SKELETON_INPUT_MODE == "original":
            skeleton_path = row["skeleton_orig_path"]
        elif SKELETON_INPUT_MODE == "view_aligned":
            skeleton_path = row[f"skeleton_cam_{cam_idx}_path"]
        else:
            raise ValueError(f"Unknown SKELETON_INPUT_MODE: {SKELETON_INPUT_MODE}")

        skeleton = np.load(skeleton_path).astype(np.float32)
        skeleton = torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, skeleton, label


class RGBSkeletonEvalDataset(Dataset):
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
        skeletons = []

        for cam_idx, col in enumerate(self.cam_cols):
            image = Image.open(row[col]).convert("RGB")
            images.append(self.transform(image))

            if SKELETON_INPUT_MODE == "original":
                skeleton_path = row["skeleton_orig_path"]
            elif SKELETON_INPUT_MODE == "view_aligned":
                skeleton_path = row[f"skeleton_cam_{cam_idx}_path"]
            else:
                raise ValueError(f"Unknown SKELETON_INPUT_MODE: {SKELETON_INPUT_MODE}")

            skeleton = np.load(skeleton_path).astype(np.float32)
            skeletons.append(torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32))

        images = torch.stack(images, dim=0)
        skeletons = torch.stack(skeletons, dim=0)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return images, skeletons, label


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

    def forward(self, image):
        return self.proj(self.backbone(image))


class MVConsistencySkeletonEncoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        self.hidden_dim = int(cfg["hidden_dim"])
        self.num_layers = int(cfg["num_layers"])
        self.dropout = float(cfg["dropout"])
        self.source_label_to_idx = ckpt["label_to_idx"]

        self.gru = nn.GRU(
            input_size=int(cfg.get("input_dim", 3)),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        gru_state = {
            key.replace("gru.", ""): value
            for key, value in ckpt["model_state_dict"].items()
            if key.startswith("gru.")
        }
        self.gru.load_state_dict(gru_state)

    def forward(self, skeleton):
        out, _ = self.gru(skeleton)
        return out[:, -1, :]


class FusionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_encoder = RGBEncoder(RGB_FEATURE_DIM, USE_IMAGENET_WEIGHTS)
        self.skeleton_encoder = MVConsistencySkeletonEncoder(MVCONSISTENCY_CKPT)

        if FREEZE_SKELETON_ENCODER:
            for param in self.skeleton_encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(RGB_FEATURE_DIM + self.skeleton_encoder.hidden_dim, FUSION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_DIM, num_classes),
        )

    def forward(self, image, skeleton):
        rgb_feat = self.rgb_encoder(image)
        skel_feat = self.skeleton_encoder(skeleton)
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


def evaluate_single_cam(model, loader):
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


def evaluate_avg_cams(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, skeletons, label in loader:
            images = images.to(DEVICE)
            skeletons = skeletons.to(DEVICE)
            label = label.to(DEVICE)
            logits_sum = None
            for cam_idx in range(images.shape[1]):
                logits_cam = model(images[:, cam_idx], skeletons[:, cam_idx])
                logits_sum = logits_cam if logits_sum is None else logits_sum + logits_cam
            logits = logits_sum / images.shape[1]
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


def write_report(out_dir, name, acc, y_true, y_pred, class_names, label_to_idx):
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
    (out_dir / f"report_{name}.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, out_dir / f"confmat_{name}.png")
    return {
        "eval_name": name,
        "test_acc": acc,
        "classes": label_to_idx,
    }


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    cam_cols = sorted(
        [c for c in df.columns if c.startswith("rgb_cam_") and c.endswith("_path")],
        key=lambda x: int(x.split("_")[2]),
    )

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_tf, eval_tf = build_transforms()
    train_ds = RGBSkeletonDataset(train_df, cam_cols, mode="train", transform=train_tf)
    label_to_idx = train_ds.label_to_idx
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    val_ds = RGBSkeletonDataset(val_df, cam_cols, label_to_idx=label_to_idx, mode="eval", transform=eval_tf)
    test_single_ds = RGBSkeletonDataset(test_df, cam_cols, label_to_idx=label_to_idx, mode="eval", transform=eval_tf)
    test_avg_ds = RGBSkeletonEvalDataset(test_df, cam_cols, label_to_idx, IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_single_loader = DataLoader(test_single_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_avg_loader = DataLoader(test_avg_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("DEVICE:", DEVICE, flush=True)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_df), flush=True)
    print("Classes:", label_to_idx, flush=True)
    print("Skeleton input mode:", SKELETON_INPUT_MODE, flush=True)
    print("Freeze skeleton encoder:", FREEZE_SKELETON_ENCODER, flush=True)
    print("MV checkpoint:", MVCONSISTENCY_CKPT, flush=True)

    model = FusionNet(num_classes=len(label_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    best_path = OUT_DIR / "best_rgb_mvconsistency_fusion.pt"

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

        val_acc, _, _ = evaluate_single_cam(model, val_loader)
        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "rgb_feature_dim": RGB_FEATURE_DIM,
                    "fusion_dim": FUSION_DIM,
                    "dropout": DROPOUT,
                    "cam_cols": cam_cols,
                    "train_rgb_mode": TRAIN_RGB_MODE,
                    "eval_cam_id": EVAL_CAM_ID,
                    "skeleton_input_mode": SKELETON_INPUT_MODE,
                    "freeze_skeleton_encoder": FREEZE_SKELETON_ENCODER,
                    "mvconsistency_ckpt": str(MVCONSISTENCY_CKPT),
                    "skeleton_hidden_dim": model.skeleton_encoder.hidden_dim,
                },
            }, best_path)
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | loss={running_loss:.4f} | "
            f"val_acc={val_acc:.4f} | best={best_val_acc:.4f}",
            flush=True,
        )
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    single_acc, y_true_single, y_pred_single = evaluate_single_cam(model, test_single_loader)
    avg_acc, y_true_avg, y_pred_avg = evaluate_avg_cams(model, test_avg_loader)
    single_summary = write_report(OUT_DIR, "cam0", single_acc, y_true_single, y_pred_single, class_names, label_to_idx)
    avg_summary = write_report(OUT_DIR, "avg_all_cams", avg_acc, y_true_avg, y_pred_avg, class_names, label_to_idx)

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_cam0_acc": single_acc,
        "test_avg_all_cams_acc": avg_acc,
        "classes": label_to_idx,
        "csv_path": str(CSV_PATH),
        "skeleton_input_mode": SKELETON_INPUT_MODE,
        "freeze_skeleton_encoder": FREEZE_SKELETON_ENCODER,
        "mvconsistency_ckpt": str(MVCONSISTENCY_CKPT),
        "single_summary": single_summary,
        "avg_summary": avg_summary,
    }
    with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== RGB + MV-CONSISTENCY FUSION SUMMARY =====", flush=True)
    print("Best val acc:", best_val_acc, flush=True)
    print("Test cam0 acc:", single_acc, flush=True)
    print("Test avg all cams acc:", avg_acc, flush=True)
    print("Saved to:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
