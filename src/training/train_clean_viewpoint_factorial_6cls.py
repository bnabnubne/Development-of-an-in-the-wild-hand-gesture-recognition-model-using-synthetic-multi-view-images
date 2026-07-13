"""Clean 2x2 test of canonicalization and skeleton-space multiview augmentation.

The four cells differ only by palm-axis canonicalization and random Blender8
training augmentation. Every run has the same samples, optimizer steps, model,
and Salux-only checkpoint selection. DrOh is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
APRIL = Path(".")
OUT_ROOT = ROOT / "results/clean_viewpoint_factorial_v2_6cls"
SALUX_SPLIT = APRIL / "metadata/salux_baseline.csv"
SALUX_RAW_META = APRIL / "dataset/skeleton_final_metadata.csv"
DROH_SPLIT = APRIL / "metadata/droh_baseline.csv"
DROH_RAW_META = APRIL / "test/mediapipe_metadata.csv"

CLASSES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
LABELS = {name: index for index, name in enumerate(CLASSES)}
ANGLES = [-45, 0, 30, 45, 60, 90, 120, 135]
CAMERA_RADIUS, CAMERA_HEIGHT = 3.0, 0.8
CONFIGS = ["camera_single", "camera_mv", "canonical_single", "canonical_mv"]
SEEDS = [0, 1, 7, 21, 42]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def label(action):
    return "thumb" if action in {"thumbup", "thumbdown"} else action


def build_manifests():
    salux = pd.read_csv(SALUX_SPLIT)
    sm = pd.read_csv(SALUX_RAW_META).rename(columns={"class": "action"})
    sm["sample_id"] = sm.raw_path.map(lambda value: Path(value).stem)
    salux = salux.merge(sm[["action", "sample_id", "raw_path", "handedness"]], on=["action", "sample_id"], how="left", validate="one_to_one")

    droh = pd.read_csv(DROH_SPLIT)
    dm = pd.read_csv(DROH_RAW_META)
    dm = dm[dm.status == "ok"].rename(columns={"class": "action"}).copy()
    dm["sample_id"] = dm.raw_path.map(lambda value: Path(value).stem)
    droh = droh.merge(dm[["action", "sample_id", "raw_path", "handedness"]], on=["action", "sample_id"], how="left", validate="one_to_one")

    for name, frame, expected in [("salux", salux, 6547), ("droh", droh, 675)]:
        if len(frame) != expected or frame[["raw_path", "handedness"]].isna().any().any():
            raise RuntimeError(f"Incomplete {name} raw manifest: rows={len(frame)}")
        missing = [path for path in frame.raw_path if not Path(path).exists()]
        if missing:
            raise RuntimeError(f"Missing {name} raw skeletons: {len(missing)}")
    return salux, droh


def minimal_preprocess(raw, handedness):
    points = np.asarray(raw, dtype=np.float32).reshape(21, 3).copy()
    if str(handedness).lower() == "left":
        points[:, 0] = 1.0 - points[:, 0]
        points[:, 2] *= -1.0
    points -= points[0:1]
    scale = float(np.linalg.norm(points[9]))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("Degenerate wrist-middle scale")
    return points / scale


def palm_canonicalize(points):
    points = np.asarray(points, dtype=np.float32).copy()
    y = points[9] - points[0]
    y /= np.linalg.norm(y) + 1e-6
    x = points[17] - points[5]
    x /= np.linalg.norm(x) + 1e-6
    z = np.cross(x, y)
    z /= np.linalg.norm(z) + 1e-6
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-6
    result = points @ np.stack([x, y, z], axis=1)
    x2 = result[17] - result[5]
    y2 = result[9] - result[0]
    if np.cross(x2 / (np.linalg.norm(x2) + 1e-6), y2 / (np.linalg.norm(y2) + 1e-6))[2] < 0:
        result[:, 0] *= -1
        result[:, 2] *= -1
    return result.astype(np.float32)


def camera_view(points, angle_deg):
    angle = np.deg2rad(angle_deg)
    camera = np.array([CAMERA_RADIUS * np.sin(angle), -CAMERA_RADIUS * np.cos(angle), CAMERA_HEIGHT], dtype=np.float32)
    z_axis = camera / (np.linalg.norm(camera) + 1e-8)
    x_axis = np.cross(np.array([0, 0, 1], dtype=np.float32), z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    world = points - points.mean(axis=0, keepdims=True)
    view = (rotation @ (world - camera).T).T
    view -= view[0:1]
    view /= np.linalg.norm(view[9]) + 1e-8
    return view.astype(np.float32)


def transform_sample(raw, handedness, canonical, with_views):
    anchor = minimal_preprocess(raw, handedness)
    transformed_anchor = palm_canonicalize(anchor) if canonical else anchor
    if not with_views:
        return transformed_anchor
    views = [transformed_anchor]
    for angle in ANGLES:
        view = camera_view(anchor, angle)
        views.append(palm_canonicalize(view) if canonical else view)
    return np.stack(views).astype(np.float32)


def raw_palm_angle(raw):
    across = raw[17] - raw[5]
    along = raw[9] - raw[0]
    normal = np.cross(across, along)
    normal /= np.linalg.norm(normal) + 1e-8
    return float(np.degrees(np.arctan2(abs(normal[0]), abs(normal[2]) + 1e-8)))


class SkeletonDataset(Dataset):
    def __init__(self, frame, canonical, with_views=False):
        self.frame = frame.reset_index(drop=True)
        values, angles = [], []
        for row in self.frame.itertuples(index=False):
            raw = np.load(row.raw_path, allow_pickle=False).astype(np.float32)
            values.append(transform_sample(raw, row.handedness, canonical, with_views))
            angles.append(raw_palm_angle(raw))
        self.x = torch.from_numpy(np.stack(values))
        self.y = torch.tensor([LABELS[label(value)] for value in self.frame.action], dtype=torch.long)
        self.angles = np.asarray(angles, dtype=np.float32)
        self.with_views = with_views

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(3, 128, batch_first=True)
        self.classifier = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        sequence, _ = self.encoder(x)
        return self.classifier(sequence[:, -1])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "rows": len(y),
    }


def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    truth, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # Evaluation always uses exactly one observed anchor.
            if x.ndim == 4:
                x = x[:, 0]
            probabilities.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
            truth.extend(y.numpy().tolist())
    probabilities = np.concatenate(probabilities)
    return np.asarray(truth), probabilities


def save_predictions(path, dataset, truth, probs):
    pred = probs.argmax(1)
    frame = dataset.frame[["action", "sample_id"]].copy()
    frame["true_class"] = [CLASSES[i] for i in truth]
    frame["predicted_class"] = [CLASSES[i] for i in pred]
    frame["correct"] = truth == pred
    frame["confidence"] = probs.max(1)
    frame["raw_palm_angle"] = dataset.angles
    for i, name in enumerate(CLASSES):
        frame[f"prob_{name}"] = probs[:, i]
    frame.to_csv(path, index=False)


def audit_preprocessing(salux, droh):
    errors = {}
    for name, frame in [("salux", salux), ("droh", droh)]:
        maxima = []
        for row in frame.itertuples(index=False):
            generated = palm_canonicalize(minimal_preprocess(np.load(row.raw_path), row.handedness))
            saved = np.load(row.input_path)
            maxima.append(float(np.max(np.abs(generated - saved))))
        errors[name] = {"max_abs": max(maxima), "mean_max_abs": float(np.mean(maxima))}
    return errors


def run(config, seed, salux, droh, force=False):
    out = OUT_ROOT / f"{config}_seed{seed}"
    summary_path = out / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {config} seed={seed}", flush=True); return
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    canonical = config.startswith("canonical")
    multiview = config.endswith("mv")
    parts = {split: salux[salux.split == split].copy() for split in ["train", "val", "test"]}
    train_ds = SkeletonDataset(parts["train"], canonical, with_views=multiview)
    val_ds = SkeletonDataset(parts["val"], canonical, with_views=False)
    salux_test = SkeletonDataset(parts["test"], canonical, with_views=False)
    droh_test = SkeletonDataset(droh, canonical, with_views=False)
    shuffle_generator = torch.Generator().manual_seed(seed)
    view_generator = torch.Generator().manual_seed(seed + 10000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=shuffle_generator)
    model = GRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale = -1.0, -1, 0
    history, start = [], time.perf_counter()
    checkpoint = out / "best.pt"
    for epoch in range(1, 151):
        model.train(); total = 0.0
        for x, y in train_loader:
            if multiview:
                # One input per anchor/update, sampled from identity + Blender8.
                indices = torch.randint(0, x.shape[1], (x.shape[0],), generator=view_generator)
                x = x[torch.arange(x.shape[0]), indices]
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
            total += loss.item()
        yt, probability = evaluate(model, val_ds)
        accuracy = accuracy_score(yt, probability.argmax(1))
        history.append({"epoch": epoch, "loss": total / len(train_loader), "val_accuracy": accuracy})
        if accuracy > best:
            best, best_epoch, stale = accuracy, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "config": config, "seed": seed}, checkpoint)
        else:
            stale += 1
        print(f"{config} seed={seed} epoch={epoch:03d} loss={total/len(train_loader):.5f} val={accuracy:.5f} best={best:.5f}", flush=True)
        if stale >= 20:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=False)["model_state_dict"])
    evaluations = {}
    for name, dataset in [("salux", salux_test), ("droh", droh_test)]:
        truth, probs = evaluate(model, dataset); pred = probs.argmax(1)
        evaluations[name] = metrics(truth, pred)
        np.save(out / f"confmat_{name}.npy", confusion_matrix(truth, pred, labels=range(6)))
        save_predictions(out / f"predictions_{name}.csv", dataset, truth, probs)
    summary = {
        "status": "complete", "experiment": "clean_viewpoint_factorial_6cls",
        "config": config, "seed": seed, "canonicalization": canonical,
        "multiview_training": multiview, "training_views": ["identity"] + (ANGLES if multiview else []),
        "sampling": "one input per anchor per optimizer update; shuffle RNG separated from view RNG",
        "loss": "cross_entropy_only", "lambda": 0.0, "best_epoch": best_epoch,
        "best_salux_val_accuracy": best, "evaluations": evaluations,
        "elapsed_seconds": time.perf_counter() - start, "device": DEVICE,
        "droh_policy": "evaluation only; single observed view; no TTA; no fitting",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS + ["all"], default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    salux, droh = build_manifests()
    audit = {
        "salux_rows": len(salux), "droh_rows": len(droh),
        "salux_splits": salux.split.value_counts().to_dict(),
        "classes": CLASSES, "preprocessing_match": audit_preprocessing(salux, droh),
    }
    (OUT_ROOT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    configs = CONFIGS if args.config == "all" else [args.config]
    seeds = SEEDS if args.seed is None else [args.seed]
    for config in configs:
        for seed in seeds:
            run(config, seed, salux, droh, force=args.force)


if __name__ == "__main__":
    main()
