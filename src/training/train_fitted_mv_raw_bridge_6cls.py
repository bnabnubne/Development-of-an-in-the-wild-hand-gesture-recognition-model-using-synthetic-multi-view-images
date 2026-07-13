"""Fitted-anchor Blender8 pipeline with balanced paired-raw supervision.

DrOh is evaluation-only.  Gamma is selected using Salux validation metrics.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from train_fitted_baseline_6cls import GRU6, LABELS, SkeletonDataset, normalize


ROOT = Path(__file__).resolve().parent
SALUX_CSV = ROOT / "metadata/salux_refined_skeleton_6cls.csv"
MV_CSV = ROOT / "metadata/salux_fitted_multiview_6cls_blender.csv"
DROH_CSV = ROOT / "metadata/droh_refined_skeleton_6cls.csv"
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 64, 150, 20
LR, LAMBDA, SEED = 1e-3, 0.3, 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class PairedFittedMV(Dataset):
    def __init__(self, frame, camera_columns):
        self.raw = torch.from_numpy(np.stack([normalize(np.load(p, allow_pickle=False)) for p in frame.raw_path]))
        self.fitted = torch.from_numpy(np.stack([normalize(np.load(p, allow_pickle=False)) for p in frame.fitted_anchor_path]))
        self.views = torch.from_numpy(np.stack([
            np.stack([normalize(np.load(getattr(row, c), allow_pickle=False)) for c in camera_columns])
            for row in frame.itertuples(index=False)
        ]))
        self.labels = torch.tensor([LABELS[a] for a in frame.action], dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.raw[i], self.fitted[i], self.views[i], self.labels[i]


def evaluate(model, loader):
    model.eval(); truth, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred.extend(model(x.to(DEVICE))[0].argmax(1).cpu().tolist()); truth.extend(y.tolist())
    return truth, pred


def metric_dict(truth, pred):
    return {
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, average="macro"),
        "weighted_f1": f1_score(truth, pred, average="weighted"),
        "rows": len(truth),
    }


def save_evaluation(out, name, truth, pred):
    (out / f"report_{name}.txt").write_text(classification_report(
        truth, pred, labels=range(6), target_names=list(LABELS), digits=6, zero_division=0), encoding="utf-8")
    np.save(out / f"confmat_{name}.npy", confusion_matrix(truth, pred, labels=range(6)))
    return metric_dict(truth, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-weight", type=float, required=True)
    args = parser.parse_args()
    if args.raw_weight < 0: raise ValueError("raw-weight must be non-negative")
    set_seed(SEED)
    tag = str(args.raw_weight).replace(".", "p")
    out = ROOT / "results" / f"fitted_mv_raw_bridge_gamma_{tag}_6cls"; out.mkdir(parents=True, exist_ok=True)

    salux = pd.read_csv(SALUX_CSV)
    mv = pd.read_csv(MV_CSV)
    mv = mv.merge(salux[["sample_id", "raw_path", "refined_path"]], on="sample_id", validate="one_to_one")
    if not np.all(mv.fitted_anchor_path == mv.refined_path): raise ValueError("Fitted-anchor metadata mismatch")
    cameras = [f"cam_{i}_path" for i in range(8)]
    train, val, test = [mv[mv.split == s].copy() for s in ["train", "val", "test"]]
    train_loader = DataLoader(PairedFittedMV(train, cameras), BATCH_SIZE, shuffle=True)
    val_raw = DataLoader(SkeletonDataset(val, "raw_path"), BATCH_SIZE, shuffle=False)
    val_fit = DataLoader(SkeletonDataset(val, "refined_path"), BATCH_SIZE, shuffle=False)

    model = GRU6().to(DEVICE); optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    best, best_epoch, stale, history = -1.0, -1, 0, []
    checkpoint_path = out / "best.pt"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); sums = {"loss": 0.0, "fit_cls": 0.0, "raw_cls": 0.0, "cons": 0.0}
        for raw, fitted, views, y in train_loader:
            raw, fitted, views, y = raw.to(DEVICE), fitted.to(DEVICE), views.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            raw_logits, _ = model(raw); fit_logits, fit_z = model(fitted)
            b, v, j, c = views.shape
            view_logits, view_z = model(views.reshape(b*v, j, c))
            view_logits, view_z = view_logits.reshape(b, v, -1), view_z.reshape(b, v, -1)
            fit_cls = (criterion(fit_logits, y) + sum(criterion(view_logits[:, i], y) for i in range(v))) / (v + 1)
            raw_cls = criterion(raw_logits, y)
            cons = (1 - F.cosine_similarity(fit_z[:, None, :], view_z, dim=2)).mean()
            loss = fit_cls + args.raw_weight * raw_cls + LAMBDA * cons
            loss.backward(); optimizer.step()
            for key, value in [("loss", loss), ("fit_cls", fit_cls), ("raw_cls", raw_cls), ("cons", cons)]: sums[key] += value.item()

        fit_t, fit_p = evaluate(model, val_fit); raw_t, raw_p = evaluate(model, val_raw)
        fit_acc, raw_acc = accuracy_score(fit_t, fit_p), accuracy_score(raw_t, raw_p)
        selection = (fit_acc + raw_acc) / 2
        row = {"epoch": epoch, **{k: v/len(train_loader) for k, v in sums.items()},
               "val_fitted_acc": fit_acc, "val_raw_acc": raw_acc, "selection_mean_acc": selection}
        history.append(row)
        if selection > best + 1e-4:
            best, best_epoch, stale = selection, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": LABELS,
                        "config": {"anchor": "fitted", "views": "Blender8 from fitted", "paired_raw_auxiliary": True,
                                   "raw_weight": args.raw_weight, "consistency_weight": LAMBDA, "hidden_dim": 128,
                                   "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
                                   "selection": "mean Salux validation fitted/raw accuracy"}}, checkpoint_path)
        else: stale += 1
        print(f"gamma={args.raw_weight:g} epoch={epoch:03d} loss={row['loss']:.5f} fit={fit_acc:.5f} raw={raw_acc:.5f} select={selection:.5f} best={best:.5f}", flush=True)
        if stale >= PATIENCE: break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    droh = pd.read_csv(DROH_CSV)
    evaluations = {}
    for name, frame, column in [
        ("salux_fitted", test, "refined_path"), ("salux_raw", test, "raw_path"),
        ("droh_raw", droh, "raw_path"), ("droh_fitted_oracle", droh, "refined_path"),
    ]:
        truth, pred = evaluate(model, DataLoader(SkeletonDataset(frame, column), BATCH_SIZE, shuffle=False))
        evaluations[name] = save_evaluation(out, name, truth, pred)
    summary = {
        "protocol": "fitted anchor + fitted Blender8, with balanced paired Salux raw auxiliary CE",
        "best_epoch": best_epoch, "best_validation_mean_accuracy": best,
        "evaluations": evaluations, "settings": checkpoint["config"],
        "external_policy": "DrOh is evaluation-only and never used for selection",
        "oracle_warning": "DrOh fitted uses its ground-truth class template and is not deployable",
        "checkpoint": str(checkpoint_path),
    }
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
