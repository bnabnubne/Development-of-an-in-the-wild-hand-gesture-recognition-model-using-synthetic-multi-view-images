"""Controlled post-submission ablations for the six-class raw-skeleton pipeline.

All checkpoint selection uses Salux validation only. DrOh is evaluated once after
training and is never used for model or hyperparameter selection.
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
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset


MODEL_ROOT = Path(__file__).resolve().parent
APRIL_ROOT = Path(".")
SALUX_CSV = APRIL_ROOT / "metadata/salux_baseline.csv"
MV_CSV = APRIL_ROOT / "metadata/salux_multiview3d_8cam_front.csv"
DROH_CSV = APRIL_ROOT / "metadata/droh_baseline.csv"
OUT_ROOT = MODEL_ROOT / "results/defense_experiments_6cls"

CLASS_NAMES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
LABELS = {name: index for index, name in enumerate(CLASS_NAMES)}
CAMERA_ANGLES_DEG = [-45, 0, 30, 45, 60, 90, 120, 135]
VIEW_INDICES = {
    0: [],
    2: [1, 4],               # 0 and 60 degrees
    4: [0, 1, 4, 6],         # -45, 0, 60 and 120 degrees
    8: list(range(8)),
}

BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 20
HIDDEN_DIM = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-views", type=int, choices=VIEW_INDICES, default=0)
    parser.add_argument("--lambda-weight", type=float, default=0.0)
    parser.add_argument("--normalization", choices=["wrist_middle", "palm_robust"], default="wrist_middle")
    parser.add_argument("--model", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def action_label(action: str) -> str:
    return "thumb" if action in {"thumbup", "thumbdown"} else action


def normalize_skeleton(value, method: str):
    x = np.asarray(value, dtype=np.float32).reshape(21, 3).copy()
    if method == "wrist_middle":
        center = x[0].copy()
        x -= center
        scale = float(np.linalg.norm(x[9]))
    else:
        palm_ids = np.array([0, 5, 9, 13, 17])
        center = x[palm_ids].mean(axis=0)
        distances = np.array([
            np.linalg.norm(x[0] - x[5]),
            np.linalg.norm(x[0] - x[9]),
            np.linalg.norm(x[0] - x[13]),
            np.linalg.norm(x[0] - x[17]),
            np.linalg.norm(x[5] - x[17]),
        ], dtype=np.float32)
        x -= center
        scale = float(np.median(distances))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError(f"Invalid skeleton scale: {scale}")
    return x / scale


class SingleDataset(Dataset):
    def __init__(self, frame, path_col: str, normalization: str):
        self.frame = frame.reset_index(drop=True)
        self.x = torch.from_numpy(np.stack([
            normalize_skeleton(np.load(path, allow_pickle=False), normalization)
            for path in self.frame[path_col]
        ]))
        self.y = torch.tensor([
            LABELS[action_label(action)] for action in self.frame.action
        ], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MultiViewDataset(Dataset):
    def __init__(self, frame, view_indices, normalization: str):
        self.frame = frame.reset_index(drop=True)
        self.anchor = torch.from_numpy(np.stack([
            normalize_skeleton(np.load(path, allow_pickle=False), normalization)
            for path in self.frame.orig_input_path
        ]))
        self.views = torch.from_numpy(np.stack([
            np.stack([
                normalize_skeleton(
                    np.load(getattr(row, f"cam_{index}_path"), allow_pickle=False),
                    normalization,
                )
                for index in view_indices
            ])
            for row in self.frame.itertuples(index=False)
        ]))
        self.y = torch.tensor([
            LABELS[action_label(action)] for action in self.frame.action
        ], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.anchor[index], self.views[index], self.y[index]


class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(3, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, len(CLASS_NAMES))

    def forward(self, x):
        sequence, _ = self.encoder(x)
        feature = sequence[:, -1]
        return self.classifier(feature), feature


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(HIDDEN_DIM, len(CLASS_NAMES))

    def forward(self, x):
        feature = self.encoder(x)
        return self.classifier(feature), feature


def make_model(name: str):
    return GRUClassifier() if name == "gru" else MLPClassifier()


def evaluate(model, loader):
    model.eval()
    targets, predictions, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[-1]
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            probabilities.extend(probs.tolist())
            predictions.extend(probs.argmax(dim=1).tolist())
            targets.extend(y.tolist())
    return targets, predictions, np.asarray(probabilities, dtype=np.float32)


def compute_metrics(targets, predictions):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
        "weighted_f1": f1_score(targets, predictions, average="weighted"),
        "rows": len(targets),
    }


def save_evaluation(out_dir, name, frame, targets, predictions, probabilities):
    result = frame[["action", "sample_id"]].copy().reset_index(drop=True)
    result["true_class"] = [CLASS_NAMES[index] for index in targets]
    result["predicted_class"] = [CLASS_NAMES[index] for index in predictions]
    result["correct"] = np.asarray(targets) == np.asarray(predictions)
    result["confidence"] = probabilities.max(axis=1)
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"prob_{class_name}"] = probabilities[:, index]
    result.to_csv(out_dir / f"predictions_{name}.csv", index=False)
    np.save(
        out_dir / f"confmat_{name}.npy",
        confusion_matrix(targets, predictions, labels=range(len(CLASS_NAMES))),
    )
    (out_dir / f"report_{name}.txt").write_text(
        classification_report(
            targets,
            predictions,
            labels=range(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            digits=6,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    return compute_metrics(targets, predictions)


def experiment_tag(args):
    weight = str(args.lambda_weight).replace(".", "p")
    return (
        f"{args.model}_{args.normalization}_views{args.num_views}_"
        f"lambda{weight}_seed{args.seed}"
    )


def main():
    args = parse_args()
    if args.num_views == 0 and args.lambda_weight != 0:
        raise ValueError("lambda-weight must be zero when num-views is zero")
    if args.lambda_weight < 0:
        raise ValueError("lambda-weight must be non-negative")

    out_dir = OUT_ROOT / experiment_tag(args)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[skip] completed: {summary_path}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    start_time = time.perf_counter()

    salux = pd.read_csv(SALUX_CSV)
    droh = pd.read_csv(DROH_CSV)
    salux_parts = {
        split: salux[salux.split == split].copy()
        for split in ["train", "val", "test"]
    }
    use_views = args.num_views > 0
    selected_indices = VIEW_INDICES[args.num_views]

    if use_views:
        mv = pd.read_csv(MV_CSV)
        mv_parts = {
            split: mv[mv.split == split].copy()
            for split in ["train", "val", "test"]
        }
        train_loader = DataLoader(
            MultiViewDataset(mv_parts["train"], selected_indices, args.normalization),
            BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            MultiViewDataset(mv_parts["val"], selected_indices, args.normalization),
            BATCH_SIZE,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            SingleDataset(salux_parts["train"], "input_path", args.normalization),
            BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            SingleDataset(salux_parts["val"], "input_path", args.normalization),
            BATCH_SIZE,
            shuffle=False,
        )

    model = make_model(args.model).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_epoch = -1
    stale = 0
    history = []
    checkpoint_path = out_dir / "best.pt"

    print(
        f"device={DEVICE} tag={experiment_tag(args)} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = total_cls = total_consistency = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_views:
                anchor, views, y = [value.to(DEVICE) for value in batch]
                anchor_logits, anchor_feature = model(anchor)
                batch_size, num_views, joints, channels = views.shape
                view_logits, view_features = model(
                    views.reshape(batch_size * num_views, joints, channels)
                )
                view_logits = view_logits.reshape(batch_size, num_views, -1)
                view_features = view_features.reshape(batch_size, num_views, -1)
                classification = (
                    criterion(anchor_logits, y)
                    + sum(criterion(view_logits[:, index], y) for index in range(num_views))
                ) / (num_views + 1)
                consistency = (
                    1
                    - F.cosine_similarity(
                        anchor_feature[:, None, :], view_features, dim=2
                    )
                ).mean()
                loss = classification + args.lambda_weight * consistency
            else:
                x, y = [value.to(DEVICE) for value in batch]
                classification = criterion(model(x)[0], y)
                consistency = torch.zeros((), device=DEVICE)
                loss = classification

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_cls += classification.item()
            total_consistency += consistency.item()

        targets, predictions, _ = evaluate(model, val_loader)
        val_accuracy = accuracy_score(targets, predictions)
        row = {
            "epoch": epoch,
            "loss": total_loss / len(train_loader),
            "classification_loss": total_cls / len(train_loader),
            "consistency_loss": total_consistency / len(train_loader),
            "val_anchor_accuracy": val_accuracy,
        }
        history.append(row)
        if val_accuracy > best_val + 1e-4:
            best_val = val_accuracy
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": LABELS,
                    "config": vars(args) | {
                        "device": DEVICE,
                        "selected_view_indices": selected_indices,
                        "camera_angles_deg": [CAMERA_ANGLES_DEG[i] for i in selected_indices],
                        "selection_dataset": "Salux validation",
                        "selection_metric": "anchor accuracy",
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} loss={row['loss']:.5f} "
            f"cls={row['classification_loss']:.5f} "
            f"con={row['consistency_loss']:.5f} "
            f"val={val_accuracy:.5f} best={best_val:.5f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluations = {}
    for name, frame in [("salux_raw", salux_parts["test"]), ("droh_raw", droh)]:
        loader = DataLoader(
            SingleDataset(frame, "input_path", args.normalization),
            BATCH_SIZE,
            shuffle=False,
        )
        targets, predictions, probabilities = evaluate(model, loader)
        evaluations[name] = save_evaluation(
            out_dir, name, frame, targets, predictions, probabilities
        )

    elapsed = time.perf_counter() - start_time
    summary = {
        "status": "complete",
        "post_submission_experiment": True,
        "tag": experiment_tag(args),
        "best_epoch": best_epoch,
        "best_salux_val_accuracy": best_val,
        "evaluations": evaluations,
        "settings": checkpoint["config"],
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint_path),
        "external_test_policy": "DrOh used only once after Salux-validation checkpoint selection",
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
