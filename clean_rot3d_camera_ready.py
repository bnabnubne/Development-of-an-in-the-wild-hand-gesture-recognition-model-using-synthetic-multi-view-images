import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["ok", "paper", "rock", "scissors", "the-finger", "thumbdown", "thumbup"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EagerSkeletonDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.labels = [LABEL_TO_IDX[action] for action in frame["action"].tolist()]
        self.arrays = [
            torch.tensor(np.load(path).astype(np.float32), dtype=torch.float32)
            for path in frame["input_path"].tolist()
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.arrays[index], torch.tensor(self.labels[index], dtype=torch.long)


class GRUEncoder(nn.Module):
    def __init__(self, num_classes: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(3, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


def read_metadata():
    rot = pd.read_csv(ROOT / "metadata" / "salux_rot3d.csv")
    canonical = pd.read_csv(ROOT / "metadata" / "salux_baseline.csv")
    handinwild = pd.read_csv(ROOT / "metadata" / "droh_baseline.csv")

    for name, frame in [("ROT3D", rot), ("CanonicalSet", canonical), ("HandinWildSet", handinwild)]:
        labels = sorted(frame["action"].unique().tolist())
        if labels != CLASS_NAMES:
            raise ValueError(f"{name} labels are not the expected 7 classes: {labels}")

    train = rot[rot["split"] == "train"].copy()
    val = canonical[canonical["split"] == "val"].copy()
    canonical_test = canonical[canonical["split"] == "test"].copy()
    external_test = handinwild.copy()

    return train, val, canonical_test, external_test


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            predictions.extend(pred)
            targets.extend(y.numpy().tolist())
    return accuracy_score(targets, predictions), targets, predictions


def make_loader(frame, *, shuffle):
    return DataLoader(EagerSkeletonDataset(frame), batch_size=64, shuffle=shuffle, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_frame, val_frame, canonical_test_frame, external_test_frame = read_metadata()

    train_loader = make_loader(train_frame, shuffle=True)
    val_loader = make_loader(val_frame, shuffle=False)
    canonical_test_loader = make_loader(canonical_test_frame, shuffle=False)
    external_test_loader = make_loader(external_test_frame, shuffle=False)

    model = GRUEncoder(num_classes=len(CLASS_NAMES), hidden_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state = None
    best_val_accuracy = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_accuracy, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss_sum": total_loss, "val_accuracy": val_accuracy})

        if val_accuracy > best_val_accuracy + 1e-4:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"clean_rot3d seed={args.seed} epoch={epoch:03d} "
            f"loss={total_loss:.4f} val={val_accuracy:.4f}",
            flush=True,
        )

        if epochs_without_improvement >= 20:
            break

    model.load_state_dict(best_state)
    canonical_accuracy, canonical_targets, canonical_predictions = evaluate(
        model, canonical_test_loader, device
    )
    external_accuracy, external_targets, external_predictions = evaluate(
        model, external_test_loader, device
    )

    output_dir = ROOT / "results" / "camera_ready" / "rot3d_clean" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "label_to_idx": LABEL_TO_IDX,
            "config": {"input_dim": 3, "hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
        },
        output_dir / "best_model.pt",
    )
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

    summary = {
        "experiment": "rot3d_clean",
        "backbone": "gru",
        "seed": args.seed,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "training_input": "ROT3D-augmented CanonicalSet training split only",
        "validation_input": "Original CanonicalSet validation split",
        "test_input": "Original CanonicalSet test split and original HandinWildSet",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 150,
        "patience": 20,
        "selection_set": "CanonicalSet validation only",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "canonical_test_accuracy": canonical_accuracy,
        "handinwild_test_accuracy": external_accuracy,
        "split_sizes": {
            "rot3d_train": len(train_frame),
            "canonical_validation": len(val_frame),
            "canonical_test": len(canonical_test_frame),
            "handinwild_test": len(external_test_frame),
        },
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
