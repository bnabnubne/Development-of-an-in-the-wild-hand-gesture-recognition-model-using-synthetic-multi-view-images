import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MODEL_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview.csv"
TRAIN_BBOX_CSV = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview_mphand_bbox.csv"
EXTERNAL_DATASETS = {
    "salux": MODEL_ROOT / "metadata" / "salux_original_rgb_5cls.csv",
    "droh": MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv",
}
EXTERNAL_BBOX_DATASETS = {
    "salux": MODEL_ROOT / "metadata" / "salux_original_rgb_5cls_mphand_bbox.csv",
    "droh": MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls_mphand_bbox.csv",
}

SEED = 42
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 15
DEFAULT_WARMUP_HEAD_EPOCHS = 2
PATIENCE = 4
NUM_WORKERS = 0
EVAL_CAM_ID = 0
DROPOUT = 0.35

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


def class_names_from_mapping(label_to_idx):
    return [label for label, _ in sorted(label_to_idx.items(), key=lambda item: item[1])]


def discover_cam_cols(df):
    return sorted(
        [c for c in df.columns if c.startswith("rgb_cam_") and c.endswith("_path")],
        key=lambda c: int(c.split("_")[2]),
    )


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_trainable_params(model, train_mode):
    set_requires_grad(model.model.features, False)
    set_requires_grad(model.model.avgpool, False)
    set_requires_grad(model.model.classifier, True)

    trainable_groups = [{"params": model.model.classifier.parameters(), "name": "classifier"}]
    if train_mode == "head":
        return trainable_groups
    if train_mode == "last_stage":
        set_requires_grad(model.model.features[7], True)
        trainable_groups.insert(0, {"params": model.model.features[7].parameters(), "name": "features_7"})
        return trainable_groups
    if train_mode == "last_two_stages":
        for stage_idx in [5, 6, 7]:
            set_requires_grad(model.model.features[stage_idx], True)
        trainable_groups.insert(
            0,
            {
                "params": [
                    param
                    for stage_idx in [5, 6, 7]
                    for param in model.model.features[stage_idx].parameters()
                ],
                "name": "features_5_6_7",
            },
        )
        return trainable_groups
    if train_mode == "full":
        set_requires_grad(model.model.features, True)
        set_requires_grad(model.model.avgpool, True)
        return [
            {"params": model.model.features.parameters(), "name": "features"},
            {"params": model.model.avgpool.parameters(), "name": "avgpool"},
            {"params": model.model.classifier.parameters(), "name": "classifier"},
        ]
    raise ValueError(f"Unknown train mode: {train_mode}")


def build_optimizer(model, train_mode, lr_backbone, lr_head, weight_decay):
    groups = configure_trainable_params(model, train_mode)
    optimizer_groups = []
    for group in groups:
        lr = lr_head if group["name"] == "classifier" else lr_backbone
        optimizer_groups.append({"params": group["params"], "lr": lr})
    return torch.optim.AdamW(optimizer_groups, weight_decay=weight_decay)


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def crop_image_from_normalized_bbox(image, bbox, margin):
    if bbox is None or any(pd.isna(v) for v in bbox):
        return image

    width, height = image.size
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0.0, 1.0)) * width
    x2 = float(np.clip(x2, 0.0, 1.0)) * width
    y1 = float(np.clip(y1, 0.0, 1.0)) * height
    y2 = float(np.clip(y2, 0.0, 1.0)) * height
    if x2 <= x1 or y2 <= y1:
        return image

    side = max(x2 - x1, y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = side * (1.0 + margin) / 2.0
    left = max(0, int(round(cx - half)))
    top = max(0, int(round(cy - half)))
    right = min(width, int(round(cx + half)))
    bottom = min(height, int(round(cy + half)))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


class RenderRGBDataset(Dataset):
    def __init__(
        self,
        df,
        cam_cols,
        label_to_idx=None,
        train=False,
        transform=None,
        crop_mode="none",
        crop_margin=0.45,
    ):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.train = train
        self.transform = transform
        self.crop_mode = crop_mode
        self.crop_margin = crop_margin
        labels = sorted(self.df["action"].unique())
        self.label_to_idx = label_to_idx or {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cam_idx = random.randrange(len(self.cam_cols)) if self.train else EVAL_CAM_ID
        image = Image.open(row[self.cam_cols[cam_idx]]).convert("RGB")
        if self.crop_mode == "mphand_bbox":
            bbox = [
                row.get(f"mphand_bbox_cam_{cam_idx}_x1", np.nan),
                row.get(f"mphand_bbox_cam_{cam_idx}_y1", np.nan),
                row.get(f"mphand_bbox_cam_{cam_idx}_x2", np.nan),
                row.get(f"mphand_bbox_cam_{cam_idx}_y2", np.nan),
            ]
            image = crop_image_from_normalized_bbox(image, bbox, self.crop_margin)
        image = self.transform(image)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, label


class RenderMultiCamDataset(Dataset):
    def __init__(
        self,
        df,
        cam_cols,
        label_to_idx,
        transform,
        crop_mode="none",
        crop_margin=0.45,
    ):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.crop_mode = crop_mode
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        for cam_idx, col in enumerate(self.cam_cols):
            image = Image.open(row[col]).convert("RGB")
            if self.crop_mode == "mphand_bbox":
                bbox = [
                    row.get(f"mphand_bbox_cam_{cam_idx}_x1", np.nan),
                    row.get(f"mphand_bbox_cam_{cam_idx}_y1", np.nan),
                    row.get(f"mphand_bbox_cam_{cam_idx}_x2", np.nan),
                    row.get(f"mphand_bbox_cam_{cam_idx}_y2", np.nan),
                ]
                image = crop_image_from_normalized_bbox(image, bbox, self.crop_margin)
            images.append(self.transform(image))
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return torch.stack(images, dim=0), label


class ExternalRGBDataset(Dataset):
    def __init__(self, df, label_to_idx, transform, crop_mode="none", crop_margin=0.45):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.crop_mode = crop_mode
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.crop_mode == "keypoints":
            image = crop_image_from_row_keypoints(image, row, self.crop_margin)
        elif self.crop_mode == "mphand_bbox":
            bbox = [
                row.get("mphand_bbox_x1", np.nan),
                row.get("mphand_bbox_y1", np.nan),
                row.get("mphand_bbox_x2", np.nan),
                row.get("mphand_bbox_y2", np.nan),
            ]
            image = crop_image_from_normalized_bbox(image, bbox, self.crop_margin)
        normal = self.transform(image)
        flipped = self.transform(ImageOps.mirror(image))
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return normal, flipped, label


def raw_keypoint_path_for_row(row):
    if "raw_path" in row and isinstance(row["raw_path"], str) and row["raw_path"]:
        path = Path(row["raw_path"])
        if path.exists():
            return path

    sample_id = str(row["sample_id"])
    action = str(row["action"])
    candidates = [
        Path("./dataset/skeleton_raw") / action / f"{sample_id}.npy",
        Path("./dataset/skeleton_raw") / action.replace("_", "-") / f"{sample_id}.npy",
        Path("./dataset/skeleton_raw") / action.replace("-", "_") / f"{sample_id}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def crop_image_from_row_keypoints(image, row, margin):
    raw_path = raw_keypoint_path_for_row(row)
    if raw_path is None:
        return image

    points = np.load(raw_path, allow_pickle=False).astype(np.float32).reshape(21, 3)[:, :2]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) == 0:
        return image

    width, height = image.size
    xs = np.clip(points[:, 0] * width, 0, width - 1)
    ys = np.clip(points[:, 1] * height, 0, height - 1)
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    side = max(x2 - x1, y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = side * (1.0 + margin) / 2.0
    left = max(0, int(round(cx - half)))
    top = max(0, int(round(cy - half)))
    right = min(width, int(round(cx + half)))
    bottom = min(height, int(round(cy + half)))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def build_transforms(variant):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if variant == "full":
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == "handcentric":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.45, 0.95), ratio=(0.72, 1.35), antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.35),
            transforms.ColorJitter(0.35, 0.35, 0.25, 0.05),
            transforms.RandomGrayscale(p=0.08),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.4, 2.5)),
        ])
    else:
        raise ValueError(f"Unknown variant: {variant}")

    eval_tf = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class ConvNeXtTinyRGB(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.model = models.convnext_tiny(weights=weights)
        in_dim = self.model.classifier[2].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],
            self.model.classifier[1],
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, image):
        return self.model(image)


def evaluate_single(model, loader):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for image, label in loader:
            logits = model(image.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(label.tolist())
    return accuracy_score(targets, preds), targets, preds


def evaluate_avg_cams(model, loader):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for images, label in loader:
            images = images.to(DEVICE)
            logits_sum = None
            for cam_idx in range(images.shape[1]):
                logits_cam = model(images[:, cam_idx])
                logits_sum = logits_cam if logits_sum is None else logits_sum + logits_cam
            logits = logits_sum / images.shape[1]
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(label.tolist())
    return accuracy_score(targets, preds), targets, preds


def evaluate_external(model, loader, use_flip_tta):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for normal, flipped, label in loader:
            logits = model(normal.to(DEVICE))
            if use_flip_tta:
                logits = (logits + model(flipped.to(DEVICE))) / 2
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(label.tolist())
    return accuracy_score(targets, preds), targets, preds


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
    np.save(out_dir / f"confmat_{name}.npy", cm)
    return {
        "name": name,
        "accuracy": acc,
        "classes": label_to_idx,
    }


def evaluate_external_datasets(
    model,
    out_dir,
    eval_tf,
    label_to_idx,
    class_names,
    batch_size,
    selected_datasets,
    use_flip_tta,
    external_crop_mode,
    external_crop_margin,
):
    summaries = {}
    dataset_paths = EXTERNAL_BBOX_DATASETS if external_crop_mode == "mphand_bbox" else EXTERNAL_DATASETS
    for dataset_name, csv_path in dataset_paths.items():
        if dataset_name not in selected_datasets:
            continue
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df["action"].isin(label_to_idx)].copy()
        if len(df) == 0:
            continue
        loader = DataLoader(
            ExternalRGBDataset(
                df,
                label_to_idx,
                eval_tf,
                crop_mode=external_crop_mode,
                crop_margin=external_crop_margin,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        dataset_summary = {"rows": len(df)}
        eval_modes = [("plain", False)]
        if use_flip_tta:
            eval_modes.append(("flip_tta", True))
        for eval_name, tta in eval_modes:
            acc, y_true, y_pred = evaluate_external(model, loader, tta)
            crop_suffix = ""
            if external_crop_mode != "none":
                margin_tag = str(external_crop_margin).replace(".", "p")
                crop_suffix = f"_{external_crop_mode}_m{margin_tag}"
            key = f"{dataset_name}_{eval_name}{crop_suffix}"
            dataset_summary[eval_name] = write_report(
                out_dir, key, acc, y_true, y_pred, class_names, label_to_idx
            )
        dataset_summary["crop_mode"] = external_crop_mode
        dataset_summary["crop_margin"] = external_crop_margin
        summaries[dataset_name] = dataset_summary
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "handcentric"], default="handcentric")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-head-epochs", type=int, default=DEFAULT_WARMUP_HEAD_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the ConvNeXt classifier head for a fast linear-probe baseline.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["head", "last_stage", "last_two_stages", "full"],
        default="last_stage",
        help="Which ConvNeXt layers to train after optional classifier-head warmup.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the existing best.pt in the output directory.",
    )
    parser.add_argument(
        "--external-datasets",
        default="salux,droh",
        help="Comma-separated external datasets to evaluate: salux,droh, or none.",
    )
    parser.add_argument(
        "--no-flip-tta",
        action="store_true",
        help="Skip external horizontal-flip TTA to make CPU eval faster.",
    )
    parser.add_argument(
        "--render-crop-mode",
        choices=["none", "mphand_bbox"],
        default="none",
        help="Crop render training/eval images using cached MediaPipe hand bboxes.",
    )
    parser.add_argument(
        "--render-crop-margin",
        type=float,
        default=0.45,
        help="Relative margin around MediaPipe render bbox.",
    )
    parser.add_argument(
        "--external-crop-mode",
        choices=["none", "keypoints", "mphand_bbox"],
        default="none",
        help="Crop external real RGB around raw keypoints or cached MediaPipe bboxes.",
    )
    parser.add_argument(
        "--external-crop-margin",
        type=float,
        default=0.45,
        help="Relative margin around keypoint bbox for external hand-centric crop.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    if args.freeze_backbone:
        args.train_mode = "head"
        args.warmup_head_epochs = 0
    crop_tag = f"_{args.render_crop_mode}" if args.render_crop_mode != "none" else ""
    out_dir = MODEL_ROOT / "results" / f"rgb_convnext_tiny_{args.variant}_{args.train_mode}{crop_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = TRAIN_BBOX_CSV if args.render_crop_mode == "mphand_bbox" else TRAIN_CSV
    df = pd.read_csv(train_csv)
    cam_cols = discover_cam_cols(df)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_tf, eval_tf = build_transforms(args.variant)
    train_ds = RenderRGBDataset(
        train_df,
        cam_cols,
        train=True,
        transform=train_tf,
        crop_mode=args.render_crop_mode,
        crop_margin=args.render_crop_margin,
    )
    label_to_idx = train_ds.label_to_idx
    class_names = class_names_from_mapping(label_to_idx)
    val_ds = RenderRGBDataset(
        val_df,
        cam_cols,
        label_to_idx,
        transform=eval_tf,
        crop_mode=args.render_crop_mode,
        crop_margin=args.render_crop_margin,
    )
    test_cam0_ds = RenderRGBDataset(
        test_df,
        cam_cols,
        label_to_idx,
        transform=eval_tf,
        crop_mode=args.render_crop_mode,
        crop_margin=args.render_crop_margin,
    )
    test_avg_ds = RenderMultiCamDataset(
        test_df,
        cam_cols,
        label_to_idx,
        eval_tf,
        crop_mode=args.render_crop_mode,
        crop_margin=args.render_crop_margin,
    )

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_cam0_loader = DataLoader(
        test_cam0_ds, args.batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    test_avg_loader = DataLoader(
        test_avg_ds, args.batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    model = ConvNeXtTinyRGB(len(label_to_idx)).to(DEVICE)
    active_train_mode = "head" if args.warmup_head_epochs > 0 else args.train_mode
    optimizer = build_optimizer(
        model, active_train_mode, args.lr_backbone, args.lr_head, args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_acc = -1.0
    best_epoch = -1
    stale = 0
    best_path = out_dir / "best.pt"

    print(f"variant={args.variant} device={DEVICE}", flush=True)
    print(
        f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_cam0_ds)}",
        flush=True,
    )
    print(f"classes={label_to_idx}", flush=True)
    print(
        f"train_mode={args.train_mode} warmup_head_epochs={args.warmup_head_epochs} "
        f"active={active_train_mode} trainable_params={count_trainable_params(model)}",
        flush=True,
    )

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            if epoch == args.warmup_head_epochs + 1 and active_train_mode != args.train_mode:
                active_train_mode = args.train_mode
                optimizer = build_optimizer(
                    model, active_train_mode, args.lr_backbone, args.lr_head, args.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.3, patience=2
                )
                print(
                    f"switched_train_mode={active_train_mode} "
                    f"trainable_params={count_trainable_params(model)}",
                    flush=True,
                )

            model.train()
            loss_sum = 0.0
            for image, label in train_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(image), label)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            val_acc, _, _ = evaluate_single(model, val_loader)
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "label_to_idx": label_to_idx,
                        "config": {
                            "architecture": "convnext_tiny",
                            "weights": "IMAGENET1K_V1",
                            "variant": args.variant,
                            "image_size": IMAGE_SIZE,
                            "cam_cols": cam_cols,
                            "eval_cam_id": EVAL_CAM_ID,
                            "dropout": DROPOUT,
                            "lr_backbone": args.lr_backbone,
                            "lr_head": args.lr_head,
                            "weight_decay": args.weight_decay,
                            "label_smoothing": args.label_smoothing,
                            "train_mode": args.train_mode,
                            "warmup_head_epochs": args.warmup_head_epochs,
                            "render_crop_mode": args.render_crop_mode,
                            "render_crop_margin": args.render_crop_margin,
                        },
                        "best_val_acc": best_val_acc,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )
            else:
                stale += 1

            print(
                f"epoch={epoch:02d} loss={loss_sum / len(train_loader):.4f} "
                f"val_acc={val_acc:.4f} best={best_val_acc:.4f} mode={active_train_mode}",
                flush=True,
            )
            if stale >= PATIENCE:
                break
    elif not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for --eval-only: {best_path}")

    ckpt = torch.load(best_path, map_location=DEVICE)
    best_val_acc = ckpt.get("best_val_acc", best_val_acc)
    best_epoch = ckpt.get("best_epoch", best_epoch)
    model.load_state_dict(ckpt["model_state_dict"])
    label_to_idx = ckpt["label_to_idx"]
    class_names = class_names_from_mapping(label_to_idx)

    test_cam0_acc, y_true, y_pred = evaluate_single(model, test_cam0_loader)
    cam0_summary = write_report(
        out_dir, "render_test_cam0", test_cam0_acc, y_true, y_pred, class_names, label_to_idx
    )
    test_avg_acc, y_true, y_pred = evaluate_avg_cams(model, test_avg_loader)
    avg_summary = write_report(
        out_dir, "render_test_avg_all_cams", test_avg_acc, y_true, y_pred,
        class_names, label_to_idx
    )
    summary_path = out_dir / "summary_train.json"
    existing_external = {}
    if summary_path.exists():
        try:
            existing_external = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "external", {}
            )
        except json.JSONDecodeError:
            existing_external = {}

    summary = {
        "variant": args.variant,
        "device": DEVICE,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "render_test_cam0": cam0_summary,
        "render_test_avg_all_cams": avg_summary,
        "checkpoint": str(best_path),
        "train_csv": str(train_csv),
        "train_mode": args.train_mode,
        "warmup_head_epochs": args.warmup_head_epochs,
        "render_crop_mode": args.render_crop_mode,
        "render_crop_margin": args.render_crop_margin,
    }
    if existing_external:
        summary["external"] = existing_external
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    selected_external = {
        name.strip()
        for name in args.external_datasets.split(",")
        if name.strip() and name.strip() != "none"
    }
    if selected_external:
        external_summary = evaluate_external_datasets(
            model,
            out_dir,
            eval_tf,
            label_to_idx,
            class_names,
            args.batch_size,
            selected_external,
            not args.no_flip_tta,
            args.external_crop_mode,
            args.external_crop_margin,
        )
        merged_external = {**existing_external, **external_summary}
        summary["external"] = merged_external
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
