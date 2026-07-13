import argparse
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


MODEL_ROOT = Path(__file__).resolve().parent
CSV_PATH = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview.csv"
SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 4
NUM_WORKERS = 0
EVAL_CAM_ID = 0

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
    def __init__(self, df, cam_cols, label_to_idx=None, train=False, transform=None):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.train = train
        self.transform = transform
        labels = sorted(self.df["action"].unique())
        self.label_to_idx = label_to_idx or {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cam_idx = random.randrange(len(self.cam_cols)) if self.train else EVAL_CAM_ID
        image = Image.open(row[self.cam_cols[cam_idx]]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, label


def build_transforms(variant):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if variant == "pretrained_full":
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),
            transforms.ToTensor(),
            normalize,
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == "pretrained_handcentric":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.42, 0.9), ratio=(0.72, 1.35), antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.35),
            transforms.ColorJitter(0.35, 0.35, 0.25, 0.05),
            transforms.RandomGrayscale(p=0.08),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.4, 2.5)),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(192, antialias=True),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return train_tf, eval_tf


class RGBOnlyNet(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, image):
        return self.classifier(self.backbone(image))


def evaluate(model, loader):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for image, label in loader:
            logits = model(image.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(label.tolist())
    return accuracy_score(targets, preds), targets, preds


def save_results(out_dir, name, acc, y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    (out_dir / f"report_{name}.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / f"confmat_{name}.npy", cm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["pretrained_full", "pretrained_handcentric"], required=True
    )
    args = parser.parse_args()
    set_seed(SEED)

    out_dir = MODEL_ROOT / "results" / f"rgb_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    cam_cols = sorted(
        [c for c in df if c.startswith("rgb_cam_") and c.endswith("_path")],
        key=lambda c: int(c.split("_")[2]),
    )
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    train_tf, eval_tf = build_transforms(args.variant)
    train_ds = RGBDataset(train_df, cam_cols, train=True, transform=train_tf)
    labels = train_ds.label_to_idx
    val_ds = RGBDataset(val_df, cam_cols, labels, transform=eval_tf)
    test_ds = RGBDataset(test_df, cam_cols, labels, transform=eval_tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = RGBOnlyNet(len(labels)).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc, best_epoch, stale = -1.0, -1, 0
    best_path = out_dir / "best.pt"

    print(f"variant={args.variant} device={DEVICE}", flush=True)
    print(f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for image, label in train_loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(image), label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": labels,
                "config": {
                    "variant": args.variant,
                    "image_size": IMAGE_SIZE,
                    "pretrained": "IMAGENET1K_V1",
                    "eval_cam_id": EVAL_CAM_ID,
                    "cam_cols": cam_cols,
                },
            }, best_path)
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} loss={loss_sum/len(train_loader):.4f} "
            f"val_acc={val_acc:.4f} best={best_acc:.4f}", flush=True
        )
        if stale >= PATIENCE:
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, y_true, y_pred = evaluate(model, test_loader)
    class_names = [label for label, _ in sorted(labels.items(), key=lambda x: x[1])]
    save_results(out_dir, "render_test_cam0", test_acc, y_true, y_pred, class_names)
    summary = {
        "variant": args.variant,
        "device": DEVICE,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "render_test_cam0_acc": test_acc,
        "classes": labels,
        "checkpoint": str(best_path),
    }
    (out_dir / "summary_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
