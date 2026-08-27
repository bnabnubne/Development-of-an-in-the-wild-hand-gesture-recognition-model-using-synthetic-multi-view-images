import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


MODEL_ROOT = Path(__file__).resolve().parent
CSV_PATH = MODEL_ROOT / "metadata" / "salux_refined_skeleton_6cls.csv"
DROH_CSV = Path("./metadata/droh_baseline.csv")
LABELS = {name: i for i, name in enumerate(["ok", "paper", "rock", "scissors", "the-finger", "thumb"])}
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
HIDDEN_DIM = 128
LR = 1e-3
CONSISTENCY_WEIGHT = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_skeleton(x):
    x = np.asarray(x, dtype=np.float32).reshape(21, 3)
    x = x - x[0:1]
    scale = float(np.linalg.norm(x, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError(f"Invalid skeleton scale: {scale}")
    return x / scale


def align_refined_to_raw(raw, refined):
    u, _, vt = np.linalg.svd(refined.T @ raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return (refined @ rotation).astype(np.float32)


class PairedDataset(Dataset):
    def __init__(self, df, align_refined=False):
        self.df = df.reset_index(drop=True)
        self.align_refined = align_refined

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_np = normalize_skeleton(np.load(row.raw_path, allow_pickle=False))
        refined_np = normalize_skeleton(np.load(row.refined_path, allow_pickle=False))
        if self.align_refined:
            refined_np = align_refined_to_raw(raw_np, refined_np)
        raw = torch.from_numpy(raw_np)
        refined = torch.from_numpy(refined_np)
        return raw, refined, LABELS[row.action]


class SingleDataset(Dataset):
    def __init__(self, df, path_col):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.from_numpy(normalize_skeleton(np.load(row[self.path_col], allow_pickle=False)))
        action = "thumb" if row.action in {"thumbup", "thumbdown"} else row.action
        return x, LABELS[action]


class GRU6(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, len(LABELS))

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.fc(z), z


def evaluate(model, loader):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE))[0].argmax(1).cpu()
            targets.extend(y.tolist())
            preds.extend(pred.tolist())
    return targets, preds


def metrics(targets, preds):
    return {
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
        "weighted_f1": f1_score(targets, preds, average="weighted"),
        "rows": len(targets),
    }


def save_eval(out_dir, name, targets, preds):
    names = list(LABELS)
    (out_dir / f"report_{name}.txt").write_text(
        classification_report(targets, preds, labels=range(6), target_names=names,
                              digits=6, zero_division=0), encoding="utf-8"
    )
    np.save(out_dir / f"confmat_{name}.npy", confusion_matrix(targets, preds, labels=range(6)))
    return metrics(targets, preds)


def train_loss(variant, model, raw, refined, y, criterion):
    if variant == "raw_normalized":
        return criterion(model(raw)[0], y)
    if variant in {"refined_only_raw_eval", "oracle_refined"}:
        return criterion(model(refined)[0], y)
    raw_logits, raw_z = model(raw)
    refined_logits, refined_z = model(refined)
    cls = (criterion(raw_logits, y) + criterion(refined_logits, y)) / 2
    if variant in {"hybrid_raw_refined", "aligned_hybrid"}:
        return cls
    if variant in {"paired_consistency", "aligned_consistency"}:
        cons = (1 - F.cosine_similarity(raw_z, refined_z, dim=1)).mean()
        return cls + CONSISTENCY_WEIGHT * cons
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=[
        "raw_normalized", "refined_only_raw_eval", "hybrid_raw_refined",
        "paired_consistency", "oracle_refined", "aligned_hybrid", "aligned_consistency",
    ])
    args = parser.parse_args()
    set_seed(SEED)
    out_dir = MODEL_ROOT / "results" / f"refined_skeleton_6cls_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    train_df, val_df, test_df = [df[df.split == s].copy() for s in ["train", "val", "test"]]
    use_alignment = args.variant in {"aligned_hybrid", "aligned_consistency"}
    train_loader = DataLoader(PairedDataset(train_df, align_refined=use_alignment), BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_col = "refined_path" if args.variant == "oracle_refined" else "raw_path"
    val_loader = DataLoader(SingleDataset(val_df, val_col), BATCH_SIZE, shuffle=False, num_workers=0)

    model = GRU6().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch = -1.0, -1
    best_path = out_dir / "best.pt"
    history = []
    print(f"variant={args.variant} device={DEVICE} train/val/test={len(train_df)}/{len(val_df)}/{len(test_df)}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for raw, refined, y in train_loader:
            raw, refined, y = raw.to(DEVICE), refined.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = train_loss(args.variant, model, raw, refined, y, criterion)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        yt, yp = evaluate(model, val_loader)
        val_acc = accuracy_score(yt, yp)
        history.append({"epoch": epoch, "loss": loss_sum / len(train_loader), "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": LABELS,
                        "config": {"variant": args.variant, "normalization": "wrist_center_max_radius",
                                   "hidden_dim": HIDDEN_DIM, "seed": SEED,
                                   "consistency_weight": CONSISTENCY_WEIGHT,
                                   "procrustes_align_refined_to_raw": use_alignment}}, best_path)
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} val={val_acc:.5f} best={best_acc:.5f}", flush=True)

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    droh = pd.read_csv(DROH_CSV)
    evaluations = {}
    for name, frame, path_col in [
        ("salux_raw", test_df, "raw_path"),
        ("salux_refined_oracle", test_df, "refined_path"),
        ("droh_raw", droh, "input_path"),
    ]:
        loader = DataLoader(SingleDataset(frame, path_col), BATCH_SIZE, shuffle=False, num_workers=0)
        yt, yp = evaluate(model, loader)
        evaluations[name] = save_eval(out_dir, name, yt, yp)
    summary = {
        "variant": args.variant, "best_epoch": best_epoch, "best_val_acc": best_acc,
        "selection_input": val_col, "evaluations": evaluations, "classes": LABELS,
        "warning": "salux_refined_oracle uses a ground-truth pose-specific template and is not deployable",
        "checkpoint": str(best_path),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
