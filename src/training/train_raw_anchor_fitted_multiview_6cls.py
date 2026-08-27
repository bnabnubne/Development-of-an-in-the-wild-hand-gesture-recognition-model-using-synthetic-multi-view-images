
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

from train_fitted_baseline_6cls import GRU6, LABELS, SkeletonDataset, normalize


MODEL_ROOT = Path(__file__).resolve().parent
SALUX_CSV = MODEL_ROOT / "metadata/salux_refined_skeleton_6cls.csv"
MV_CSV = MODEL_ROOT / "metadata/salux_fitted_multiview_6cls_blender.csv"
DROH_CSV = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls.csv"
OUT_DIR = MODEL_ROOT / "results/raw_anchor_fitted_multiview_6cls"
BATCH_SIZE, EPOCHS, PATIENCE = 64, 150, 20
LR, CONS_WEIGHT, SEED = 1e-3, 0.3, 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class RawAnchorFittedViews(Dataset):
    def __init__(self, df, cam_cols):
        self.anchor = torch.from_numpy(np.stack([normalize(np.load(p, allow_pickle=False)) for p in df.raw_path]))
        self.views = torch.from_numpy(np.stack([
            np.stack([np.load(getattr(row, col), allow_pickle=False).astype(np.float32) for col in cam_cols])
            for row in df.itertuples(index=False)
        ]))
        self.y = torch.tensor([LABELS[a] for a in df.action], dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.anchor[idx], self.views[idx], self.y[idx]


def evaluate(model, loader):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            yp.extend(model(x.to(DEVICE))[0].argmax(1).cpu().tolist()); yt.extend(y.tolist())
    return yt, yp


def save_eval(name, yt, yp):
    (OUT_DIR / f"report_{name}.txt").write_text(classification_report(
        yt, yp, labels=range(6), target_names=list(LABELS), digits=6, zero_division=0), encoding="utf-8")
    np.save(OUT_DIR / f"confmat_{name}.npy", confusion_matrix(yt, yp, labels=range(6)))
    return {"accuracy": accuracy_score(yt, yp), "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro"), "weighted_f1": f1_score(yt, yp, average="weighted"),
            "rows": len(yt)}


def main():
    set_seed(SEED); OUT_DIR.mkdir(parents=True, exist_ok=True)
    salux, mv, droh = pd.read_csv(SALUX_CSV), pd.read_csv(MV_CSV), pd.read_csv(DROH_CSV)
    mv = mv.merge(salux[["sample_id", "raw_path"]], on="sample_id", validate="one_to_one")
    cam_cols = [f"cam_{i}_path" for i in range(8)]
    train, val, test = [mv[mv.split == split].copy() for split in ["train", "val", "test"]]
    train_loader = DataLoader(RawAnchorFittedViews(train, cam_cols), BATCH_SIZE, True)
    val_loader = DataLoader(SkeletonDataset(val, "raw_path"), BATCH_SIZE, False)
    model = GRU6().to(DEVICE); optimizer = torch.optim.Adam(model.parameters(), lr=LR); criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale, history = -1.0, -1, 0, []; best_path = OUT_DIR / "best.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = cls_total = con_total = 0.0
        for anchor, views, y in train_loader:
            anchor, views, y = anchor.to(DEVICE), views.to(DEVICE), y.to(DEVICE); optimizer.zero_grad()
            logits_a, z_a = model(anchor); b, v, j, c = views.shape
            logits_v, z_v = model(views.reshape(b*v, j, c)); logits_v = logits_v.reshape(b, v, -1); z_v = z_v.reshape(b, v, -1)
            cls = (criterion(logits_a, y) + sum(criterion(logits_v[:, i], y) for i in range(v))) / (v + 1)
            con = (1 - F.cosine_similarity(z_a[:, None, :], z_v, dim=2)).mean()
            loss = cls + CONS_WEIGHT * con; loss.backward(); optimizer.step()
            total += loss.item(); cls_total += cls.item(); con_total += con.item()
        yt, yp = evaluate(model, val_loader); acc = accuracy_score(yt, yp)
        history.append({"epoch": epoch, "loss": total/len(train_loader), "cls_loss": cls_total/len(train_loader),
                        "cons_loss": con_total/len(train_loader), "val_raw_anchor_acc": acc})
        if acc > best + 1e-4:
            best, best_epoch, stale = acc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": LABELS,
                        "config": {"anchor": "raw", "views": "Blender8 generated from fitted",
                                   "normalization": "wrist-middle-MCP", "consistency_weight": CONS_WEIGHT,
                                   "hidden_dim": 128, "seed": SEED}}, best_path)
        else: stale += 1
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} cls={history[-1]['cls_loss']:.5f} con={history[-1]['cons_loss']:.5f} val={acc:.5f} best={best:.5f}", flush=True)
        if stale >= PATIENCE: break
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    evaluations = {}
    for name, frame, col in [("salux_raw", test, "raw_path"), ("salux_fitted", test, "fitted_anchor_path"),
                             ("droh_raw", droh, "raw_path"), ("droh_fitted_oracle", droh, "refined_path")]:
        yt, yp = evaluate(model, DataLoader(SkeletonDataset(frame, col), BATCH_SIZE, False)); evaluations[name] = save_eval(name, yt, yp)
    summary = {"protocol": "raw anchor + Blender8 views generated from fitted skeleton",
               "best_epoch": best_epoch, "best_val_acc": best, "evaluations": evaluations,
               "settings": {"normalization": "wrist-middle-MCP", "batch_size": BATCH_SIZE, "optimizer": "Adam",
                            "lr": LR, "consistency_weight": CONS_WEIGHT, "hidden_dim": 128, "seed": SEED,
                            "max_epochs": EPOCHS, "patience": PATIENCE},
               "inference": "single-view; raw is the deployment input",
               "warning": "DrOh fitted is a ground-truth class-template oracle diagnostic.", "checkpoint": str(best_path)}
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
