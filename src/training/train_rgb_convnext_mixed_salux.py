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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_rgb_convnext import (
    DEVICE,
    DROPOUT,
    EVAL_CAM_ID,
    IMAGE_SIZE,
    NUM_WORKERS,
    ConvNeXtTinyRGB,
    build_optimizer,
    build_transforms,
    class_names_from_mapping,
    count_trainable_params,
    discover_cam_cols,
)


MODEL_ROOT = Path(__file__).resolve().parent
RENDER_CSV = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview.csv"
SALUX_ALL_CSV = MODEL_ROOT / "metadata" / "salux_original_rgb_all_5cls.csv"
DROH_CSV = MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv"
OUT_DIR = MODEL_ROOT / "results" / "rgb_convnext_tiny_mixed_salux_real"

SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_WARMUP_HEAD_EPOCHS = 2
PATIENCE = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MixedRGBDataset(Dataset):
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
        if row["source"] == "render":
            cam_idx = random.randrange(len(self.cam_cols)) if self.train else EVAL_CAM_ID
            image_path = row[self.cam_cols[cam_idx]]
        else:
            image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return image, label


def make_sampler(df, label_to_idx):
    labels = df["action"].map(label_to_idx).to_numpy()
    counts = np.bincount(labels, minlength=len(label_to_idx)).astype(np.float32)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def evaluate(model, loader):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for image, label in loader:
            logits = model(image.to(DEVICE))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "handcentric"], default="full")
    parser.add_argument("--train-mode", choices=["head", "last_stage", "last_two_stages", "full"], default="last_stage")
    parser.add_argument("--warmup-head-epochs", type=int, default=DEFAULT_WARMUP_HEAD_EPOCHS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--render-weight", type=float, default=0.35)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    render_df = pd.read_csv(RENDER_CSV)
    cam_cols = discover_cam_cols(render_df)
    render_df = render_df[render_df["split"] == "train"].copy()
    render_df["source"] = "render"

    salux_df = pd.read_csv(SALUX_ALL_CSV)
    salux_train = salux_df[salux_df["split"] == "train"].copy()
    salux_val = salux_df[salux_df["split"] == "val"].copy()
    salux_test = salux_df[salux_df["split"] == "test"].copy()
    for frame in [salux_train, salux_val, salux_test]:
        frame["source"] = "salux"

    if args.render_weight <= 0:
        train_df = salux_train.copy()
    else:
        n_render = min(len(render_df), int(round(len(salux_train) * args.render_weight)))
        render_sample = render_df.sample(n=n_render, replace=n_render > len(render_df), random_state=SEED)
        train_df = pd.concat([salux_train, render_sample], ignore_index=True)

    train_tf, eval_tf = build_transforms(args.variant)
    train_ds = MixedRGBDataset(train_df, cam_cols, train=True, transform=train_tf)
    label_to_idx = train_ds.label_to_idx
    class_names = class_names_from_mapping(label_to_idx)
    val_ds = MixedRGBDataset(salux_val, cam_cols, label_to_idx, transform=eval_tf)
    salux_test_ds = MixedRGBDataset(salux_test, cam_cols, label_to_idx, transform=eval_tf)

    droh_df = pd.read_csv(DROH_CSV)
    droh_df = droh_df[droh_df["action"].isin(label_to_idx)].copy()
    droh_df["source"] = "droh"
    droh_ds = MixedRGBDataset(droh_df, cam_cols, label_to_idx, transform=eval_tf)

    sampler = make_sampler(train_df, label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    salux_test_loader = DataLoader(salux_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    droh_loader = DataLoader(droh_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = ConvNeXtTinyRGB(len(label_to_idx)).to(DEVICE)
    active_train_mode = "head" if args.warmup_head_epochs > 0 else args.train_mode
    optimizer = build_optimizer(model, active_train_mode, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_acc = -1.0
    best_epoch = -1
    stale = 0
    best_path = OUT_DIR / "best.pt"

    print(f"variant={args.variant} device={DEVICE}", flush=True)
    print(f"train={len(train_ds)} salux_val={len(val_ds)} salux_test={len(salux_test_ds)} droh={len(droh_ds)}", flush=True)
    print(f"classes={label_to_idx}", flush=True)
    print(f"mode={args.train_mode} active={active_train_mode} trainable={count_trainable_params(model)}", flush=True)

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            if epoch == args.warmup_head_epochs + 1 and active_train_mode != args.train_mode:
                active_train_mode = args.train_mode
                optimizer = build_optimizer(model, active_train_mode, args.lr_backbone, args.lr_head, args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
                print(f"switched_mode={active_train_mode} trainable={count_trainable_params(model)}", flush=True)

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

            val_acc, _, _ = evaluate(model, val_loader)
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                stale = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "config": {
                        "architecture": "convnext_tiny",
                        "weights": "IMAGENET1K_V1",
                        "variant": args.variant,
                        "image_size": IMAGE_SIZE,
                        "train_mode": args.train_mode,
                        "warmup_head_epochs": args.warmup_head_epochs,
                        "lr_backbone": args.lr_backbone,
                        "lr_head": args.lr_head,
                        "render_weight": args.render_weight,
                    },
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                }, best_path)
            else:
                stale += 1
            print(
                f"epoch={epoch:02d} loss={loss_sum / len(train_loader):.4f} "
                f"salux_val_acc={val_acc:.4f} best={best_val_acc:.4f} mode={active_train_mode}",
                flush=True,
            )
            if stale >= PATIENCE:
                break
    elif not best_path.exists():
        raise FileNotFoundError(best_path)

    ckpt = torch.load(best_path, map_location=DEVICE)
    best_val_acc = ckpt.get("best_val_acc", best_val_acc)
    best_epoch = ckpt.get("best_epoch", best_epoch)
    model.load_state_dict(ckpt["model_state_dict"])
    salux_acc, y_true, y_pred = evaluate(model, salux_test_loader)
    salux_summary = write_report(OUT_DIR, "salux_test", salux_acc, y_true, y_pred, class_names, label_to_idx)
    droh_acc, y_true, y_pred = evaluate(model, droh_loader)
    droh_summary = write_report(OUT_DIR, "droh_external", droh_acc, y_true, y_pred, class_names, label_to_idx)

    summary = {
        "device": DEVICE,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "salux_test": salux_summary,
        "droh_external": droh_summary,
        "train_rows": len(train_df),
        "salux_train_rows": len(salux_train),
        "render_train_rows": int((train_df["source"] == "render").sum()),
        "checkpoint": str(best_path),
    }
    (OUT_DIR / "summary_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
