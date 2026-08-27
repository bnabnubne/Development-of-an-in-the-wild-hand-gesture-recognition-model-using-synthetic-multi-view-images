
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


MODEL_ROOT = Path(__file__).resolve().parent
SALUX_CSV = MODEL_ROOT / "metadata/salux_refined_skeleton_6cls.csv"
DROH_CSV = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls.csv"
OUT_DIR = MODEL_ROOT / "results/fitted_baseline_6cls"
LABELS = {name: i for i, name in enumerate(["ok", "paper", "rock", "scissors", "the-finger", "thumb"])}
BATCH_SIZE, EPOCHS, PATIENCE = 64, 150, 20
HIDDEN_DIM, LR, SEED = 128, 1e-3, 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def normalize(x):
    x = np.asarray(x, dtype=np.float32).reshape(21, 3).copy(); x -= x[0:1]
    scale = float(np.linalg.norm(x[9] - x[0]))
    if not np.isfinite(scale) or scale < 1e-8: raise ValueError(f"invalid scale {scale}")
    return x / scale


class SkeletonDataset(Dataset):
    def __init__(self, df, path_col):
        self.x = torch.from_numpy(np.stack([normalize(np.load(p, allow_pickle=False)) for p in df[path_col]]))
        self.y = torch.tensor([LABELS[a] for a in df.action], dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class GRU6(nn.Module):
    def __init__(self):
        super().__init__(); self.gru = nn.GRU(3, HIDDEN_DIM, batch_first=True); self.fc = nn.Linear(HIDDEN_DIM, 6)
    def forward(self, x):
        out, _ = self.gru(x); z = out[:, -1]; return self.fc(z), z


def evaluate(model, loader):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            yp.extend(model(x.to(DEVICE))[0].argmax(1).cpu().tolist()); yt.extend(y.tolist())
    return yt, yp


def save_eval(name, yt, yp):
    names = list(LABELS)
    (OUT_DIR / f"report_{name}.txt").write_text(classification_report(
        yt, yp, labels=range(6), target_names=names, digits=6, zero_division=0), encoding="utf-8")
    np.save(OUT_DIR / f"confmat_{name}.npy", confusion_matrix(yt, yp, labels=range(6)))
    return {"accuracy": accuracy_score(yt, yp), "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro"), "weighted_f1": f1_score(yt, yp, average="weighted"),
            "rows": len(yt)}


def main():
    set_seed(SEED); OUT_DIR.mkdir(parents=True, exist_ok=True)
    salux, droh = pd.read_csv(SALUX_CSV), pd.read_csv(DROH_CSV)
    train, val, test = [salux[salux.split == split].copy() for split in ["train", "val", "test"]]
    train_loader = DataLoader(SkeletonDataset(train, "refined_path"), BATCH_SIZE, True)
    val_loader = DataLoader(SkeletonDataset(val, "refined_path"), BATCH_SIZE, False)
    model = GRU6().to(DEVICE); optimizer = torch.optim.Adam(model.parameters(), lr=LR); criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale, history = -1.0, -1, 0, []
    best_path = OUT_DIR / "best.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); optimizer.zero_grad()
            loss = criterion(model(x)[0], y); loss.backward(); optimizer.step(); total += loss.item()
        yt, yp = evaluate(model, val_loader); acc = accuracy_score(yt, yp)
        history.append({"epoch": epoch, "loss": total/len(train_loader), "val_fitted_anchor_acc": acc})
        if acc > best + 1e-4:
            best, best_epoch, stale = acc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": LABELS,
                        "config": {"experiment": "fitted_baseline", "normalization": "wrist_middle_mcp",
                                   "hidden_dim": HIDDEN_DIM, "seed": SEED, "num_views": 1}}, best_path)
        else: stale += 1
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} val={acc:.5f} best={best:.5f}", flush=True)
        if stale >= PATIENCE: break
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    evaluations = {}
    for name, frame, col in [("salux_fitted", test, "refined_path"), ("salux_raw", test, "raw_path"),
                             ("droh_raw", droh, "raw_path"), ("droh_fitted_oracle", droh, "refined_path")]:
        yt, yp = evaluate(model, DataLoader(SkeletonDataset(frame, col), BATCH_SIZE, False)); evaluations[name] = save_eval(name, yt, yp)
    summary = {"protocol": "fitted single-view baseline; fair control for fitted+Blender8",
               "best_epoch": best_epoch, "best_val_acc": best, "evaluations": evaluations,
               "settings": {"normalization": "wrist-middle-MCP", "batch_size": BATCH_SIZE,
                            "optimizer": "Adam", "lr": LR, "hidden_dim": HIDDEN_DIM, "seed": SEED,
                            "max_epochs": EPOCHS, "patience": PATIENCE},
               "warning": "DrOh fitted is a ground-truth class-template oracle diagnostic.", "checkpoint": str(best_path)}
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
