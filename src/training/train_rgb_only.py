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


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = Path("./metadata/rgb_skeleton_multiview.csv")
OUT_DIR = Path("./results/rgb_only")

BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
IMAGE_SIZE = 160
DROPOUT = 0.2
TRAIN_RGB_MODE = "random_cam"  # "random_cam" or "all_cams"
EVAL_CAM_ID = 0
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


class RGBDataset(Dataset):
    def __init__(self, df, cam_cols, label_to_idx=None, mode="eval", transform=None):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.mode = mode
        self.transform = transform

        if label_to_idx is None:
            labels = sorted(self.df["action"].unique().tolist())
            self.label_to_idx = {label: i for i, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.items = []
        if mode == "train" and TRAIN_RGB_MODE == "all_cams":
            for row_idx in range(len(self.df)):
                for cam_idx in range(len(self.cam_cols)):
                    self.items.append((row_idx, cam_idx))
        else:
            for row_idx in range(len(self.df)):
                self.items.append((row_idx, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row_idx, fixed_cam = self.items[idx]
        row = self.df.iloc[row_idx]
        if fixed_cam is None:
            cam_idx = random.randrange(len(self.cam_cols)) if self.mode == "train" else EVAL_CAM_ID
        else:
            cam_idx = fixed_cam

        image = Image.open(row[self.cam_cols[cam_idx]]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, label


class RGBOnlyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, image):
        return self.classifier(self.backbone(image))


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
        for image, label in loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            logits = model(image)
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

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_tf, eval_tf = build_transforms()
    train_ds = RGBDataset(train_df, cam_cols, mode="train", transform=train_tf)
    label_to_idx = train_ds.label_to_idx
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    val_ds = RGBDataset(val_df, cam_cols, label_to_idx=label_to_idx, mode="eval", transform=eval_tf)
    test_ds = RGBDataset(test_df, cam_cols, label_to_idx=label_to_idx, mode="eval", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("DEVICE:", DEVICE, flush=True)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds), flush=True)
    print("Classes:", label_to_idx, flush=True)

    model = RGBOnlyNet(num_classes=len(label_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    best_path = OUT_DIR / "best_rgb_only.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for image, label in train_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, _, _ = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "cam_cols": cam_cols,
                    "eval_cam_id": EVAL_CAM_ID,
                    "train_rgb_mode": TRAIN_RGB_MODE,
                },
            }, best_path)
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | loss={running_loss:.4f} | val_acc={val_acc:.4f}", flush=True)
        if no_improve >= PATIENCE:
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
    }
    with open(OUT_DIR / "summary_train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== RGB-ONLY SUMMARY =====", flush=True)
    print("Best val acc:", best_val_acc, flush=True)
    print("Test acc    :", test_acc, flush=True)
    print("Saved to    :", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
