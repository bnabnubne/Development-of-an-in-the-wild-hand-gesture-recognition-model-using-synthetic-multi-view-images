"""Reproducible camera-ready experiments for the MAPR hand-gesture paper.

The external HandinWildSet is evaluated only after model selection on the
CanonicalSet validation split.  Synthetic views are training-only inputs;
validation and both test sets use one original skeleton per sample.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
ORIG_CSV = ROOT / "metadata/salux_baseline.csv"
EXTERNAL_CSV = ROOT / "metadata/droh_baseline.csv"
MV4_CSV = ROOT / "metadata/salux_multiview3d.csv"
MV8_CSV = ROOT / "metadata/salux_multiview3d_8cam_front.csv"
CLASS_NAMES = ["ok", "paper", "rock", "scissors", "the-finger", "thumbdown", "thumbup"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SkeletonDataset(Dataset):
    def __init__(self, frame, path_col="input_path", view_cols=None):
        # Skeleton files are tiny, but opening tens of thousands of .npy files
        # every epoch dominates CPU training.  Load each split once; this does
        # not alter samples, ordering, augmentation, or the training protocol.
        self.originals = torch.stack([self.load(path) for path in frame[path_col]], dim=0)
        self.views = None
        if view_cols:
            self.views = torch.stack([
                torch.stack([self.load(path) for path in row], dim=0)
                for row in frame[view_cols].values.tolist()
            ], dim=0)
        self.labels = [LABEL_TO_IDX[x] for x in frame["action"]]

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load(path):
        value = np.load(path).astype(np.float32).reshape(21, 3)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite skeleton: {path}")
        return torch.from_numpy(value)

    def __getitem__(self, idx):
        original = self.originals[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.views is None:
            return original, label
        return original, self.views[idx], label


class GRUEncoder(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(3, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        sequence, _ = self.gru(x)
        embedding = sequence[:, -1, :]
        return self.fc(embedding), embedding


def hand_adjacency():
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    adjacency = torch.eye(21)
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    degree = adjacency.sum(dim=1)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adjacency):
        x = torch.einsum("ij,bjf->bif", adjacency, x)
        return F.relu(self.norm(self.linear(x)))


class HandGCN(nn.Module):
    """Static spatial GCN using the 21-joint MediaPipe hand topology."""

    def __init__(self, num_classes=7, hidden_dim=128):
        super().__init__()
        self.register_buffer("adjacency", hand_adjacency())
        self.gcn1 = GraphConv(3, 64)
        self.gcn2 = GraphConv(64, 128)
        self.gcn3 = GraphConv(128, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.gcn1(x, self.adjacency)
        x = self.gcn2(x, self.adjacency)
        x = self.gcn3(x, self.adjacency)
        embedding = x.mean(dim=1)
        return self.fc(embedding), embedding


def load_frames(num_views):
    original = pd.read_csv(ORIG_CSV)
    external = pd.read_csv(EXTERNAL_CSV)
    for frame, name in [(original, "CanonicalSet"), (external, "HandinWildSet")]:
        labels = set(frame.action.unique())
        if labels != set(CLASS_NAMES):
            raise ValueError(f"Unexpected {name} classes: {sorted(labels)}")

    train = original[original.split == "train"].copy()
    val = original[original.split == "val"].copy()
    test = original[original.split == "test"].copy()
    view_cols = []
    if num_views:
        multiview = pd.read_csv(MV4_CSV if num_views == 4 else MV8_CSV)
        view_cols = [f"cam_{idx}_path" for idx in range(num_views)]
        required = ["action", "sample_id", "split", *view_cols]
        if any(column not in multiview for column in required):
            raise ValueError("Multiview metadata does not contain the expected camera columns")
        train = train.merge(
            multiview[required], on=["action", "sample_id", "split"], how="inner",
            validate="one_to_one"
        )
        if len(train) != (original.split == "train").sum():
            raise ValueError("Training samples were lost while joining synthetic views")
    return train, val, test, external, view_cols


def single_view_loader(frame, shuffle=False):
    return DataLoader(SkeletonDataset(frame), batch_size=64, shuffle=shuffle, num_workers=0)


def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for skeleton, label in loader:
            logits, _ = model(skeleton.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(label.tolist())
    return accuracy_score(targets, predictions), targets, predictions


def train(args):
    seed_everything(args.seed)
    device = select_device()
    train_frame, val_frame, test_frame, external_frame, view_cols = load_frames(args.views)
    if args.views:
        train_loader = DataLoader(
            SkeletonDataset(train_frame, view_cols=view_cols), batch_size=64,
            shuffle=True, num_workers=0
        )
    else:
        train_loader = single_view_loader(train_frame, shuffle=True)
    val_loader = single_view_loader(val_frame)
    test_loader = single_view_loader(test_frame)
    external_loader = single_view_loader(external_frame)

    model = (GRUEncoder() if args.backbone == "gru" else HandGCN()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_state, best_val, best_epoch = None, -1.0, -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if args.views:
                original, views, label = batch
                original, views, label = original.to(device), views.to(device), label.to(device)
                logits_original, embedding_original = model(original)
                classification_loss = criterion(logits_original, label)
                consistency_terms = []
                for view_index in range(args.views):
                    logits_view, embedding_view = model(views[:, view_index])
                    classification_loss = classification_loss + criterion(logits_view, label)
                    consistency_terms.append(
                        1.0 - F.cosine_similarity(embedding_original, embedding_view, dim=1).mean()
                    )
                classification_loss = classification_loss / (args.views + 1)
                consistency_loss = torch.stack(consistency_terms).mean()
                loss = classification_loss + args.consistency_weight * consistency_loss
            else:
                original, label = batch
                logits, _ = model(original.to(device))
                loss = criterion(logits, label.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_accuracy, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss_sum": total_loss, "val_accuracy": val_accuracy})
        if val_accuracy > best_val + 1e-4:
            best_val, best_epoch = val_accuracy, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"{args.backbone} views={args.views} lambda={args.consistency_weight} "
            f"seed={args.seed} epoch={epoch:03d} loss={total_loss:.4f} val={val_accuracy:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= 20:
            break

    model.load_state_dict(best_state)
    canonical_accuracy, canonical_targets, canonical_predictions = evaluate(model, test_loader, device)
    external_accuracy, external_targets, external_predictions = evaluate(model, external_loader, device)

    experiment_name = (
        f"{args.backbone}_original" if not args.views else
        f"{args.backbone}_mv{args.views}_lambda{args.consistency_weight:g}"
    )
    output_dir = ROOT / "results" / "camera_ready" / experiment_name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "label_to_idx": LABEL_TO_IDX,
        "config": vars(args),
    }, output_dir / "best_model.pt")
    summary = {
        "experiment": experiment_name,
        "backbone": args.backbone,
        "seed": args.seed,
        "num_views": args.views,
        "consistency_weight": args.consistency_weight,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "max_epochs": 150,
        "patience": 20,
        "selection_set": "CanonicalSet validation only",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "canonical_test_accuracy": canonical_accuracy,
        "handinwild_test_accuracy": external_accuracy,
        "split_sizes": {
            "train": len(train_frame), "validation": len(val_frame),
            "canonical_test": len(test_frame), "handinwild_test": len(external_frame),
        },
        "class_names": CLASS_NAMES,
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "canonical_report.txt").write_text(
        classification_report(canonical_targets, canonical_predictions, target_names=CLASS_NAMES, digits=4),
        encoding="utf-8",
    )
    (output_dir / "handinwild_report.txt").write_text(
        classification_report(external_targets, external_predictions, target_names=CLASS_NAMES, digits=4),
        encoding="utf-8",
    )
    np.save(output_dir / "canonical_confusion.npy", confusion_matrix(canonical_targets, canonical_predictions))
    np.save(output_dir / "handinwild_confusion.npy", confusion_matrix(external_targets, external_predictions))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["gru", "gcn"], required=True)
    parser.add_argument("--views", type=int, choices=[0, 4, 8], default=0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.views == 0 and arguments.consistency_weight != 0:
        parser.error("Consistency requires synthetic views")
    train(arguments)
