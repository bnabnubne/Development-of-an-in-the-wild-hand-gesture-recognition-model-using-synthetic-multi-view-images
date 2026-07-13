"""Full-stack 6-class experiment: fitted anchor + eight fitted camera views."""

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
MV_CSV = MODEL_ROOT / "metadata/salux_fitted_multiview_6cls_blender.csv"
DROH_CSV = Path("./metadata/droh_baseline.csv")
OUT_DIR = MODEL_ROOT / "results/fitted_anchor_multiview_6cls"
LABELS = {name: i for i, name in enumerate(["ok", "paper", "rock", "scissors", "the-finger", "thumb"])}
BATCH_SIZE, EPOCHS, PATIENCE = 64, 150, 20
HIDDEN_DIM, LR, CONS_WEIGHT, SEED = 128, 1e-3, 0.3, 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def paper_normalize(x):
    x = np.asarray(x, dtype=np.float32).reshape(21, 3).copy()
    x -= x[0:1]
    scale = float(np.linalg.norm(x[9] - x[0]))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError(f"invalid wrist-to-middle-MCP scale: {scale}")
    return x / scale


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class FittedMVDataset(Dataset):
    def __init__(self, df, cam_cols, include_views):
        self.df, self.cam_cols, self.include_views = df.reset_index(drop=True), cam_cols, include_views
        self.anchors = torch.from_numpy(np.stack([
            paper_normalize(np.load(row.fitted_anchor_path, allow_pickle=False))
            for row in self.df.itertuples(index=False)
        ]).astype(np.float32))
        self.labels = torch.tensor([LABELS[action] for action in self.df.action], dtype=torch.long)
        self.views = None
        if include_views:
            self.views = torch.from_numpy(np.stack([
                np.stack([np.load(getattr(row, c), allow_pickle=False).astype(np.float32) for c in cam_cols])
                for row in self.df.itertuples(index=False)
            ]).astype(np.float32))
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        if not self.include_views:
            return self.anchors[idx], self.labels[idx]
        return self.anchors[idx], self.views[idx], self.labels[idx]


class SingleRawDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.skeletons = torch.from_numpy(np.stack([
            paper_normalize(np.load(row.input_path, allow_pickle=False))
            for row in self.df.itertuples(index=False)
        ]).astype(np.float32))
        self.labels = torch.tensor([
            LABELS["thumb" if action in {"thumbup", "thumbdown"} else action]
            for action in self.df.action
        ], dtype=torch.long)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return self.skeletons[idx], self.labels[idx]


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
            "macro_f1": f1_score(yt, yp, average="macro"),
            "weighted_f1": f1_score(yt, yp, average="weighted"), "rows": len(yt)}


def main():
    set_seed(SEED); OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MV_CSV)
    cam_cols = [f"cam_{i}_path" for i in range(8)]
    if len(df) != 6547 or any(c not in df for c in cam_cols):
        raise ValueError("Incomplete fitted 8-camera metadata")
    train, val, test = [df[df.split == s].copy() for s in ["train", "val", "test"]]
    train_loader = DataLoader(FittedMVDataset(train, cam_cols, True), BATCH_SIZE, True, num_workers=0)
    val_loader = DataLoader(FittedMVDataset(val, cam_cols, False), BATCH_SIZE, False, num_workers=0)
    model = GRU6().to(DEVICE); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_epoch, stale, history = -1.0, -1, 0, []
    best_path = OUT_DIR / "best.pt"
    print(f"device={DEVICE} train/val/test={len(train)}/{len(val)}/{len(test)} protocol=fitted_anchor+8_fitted_camera_views", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = cls_total = con_total = 0.0
        for anchor, views, y in train_loader:
            anchor, views, y = anchor.to(DEVICE), views.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(); logits_a, z_a = model(anchor)
            b, v, j, c = views.shape; logits_v, z_v = model(views.reshape(b*v, j, c))
            logits_v, z_v = logits_v.reshape(b, v, -1), z_v.reshape(b, v, -1)
            cls = (criterion(logits_a, y) + sum(criterion(logits_v[:, i], y) for i in range(v))) / (v + 1)
            con = (1 - F.cosine_similarity(z_a[:, None, :], z_v, dim=2)).mean()
            loss = cls + CONS_WEIGHT * con; loss.backward(); optimizer.step()
            total += loss.item(); cls_total += cls.item(); con_total += con.item()
        yt, yp = evaluate(model, val_loader); val_acc = accuracy_score(yt, yp)
        history.append({"epoch": epoch, "loss": total/len(train_loader), "cls_loss": cls_total/len(train_loader),
                        "cons_loss": con_total/len(train_loader), "val_fitted_anchor_acc": val_acc})
        if val_acc > best + 1e-4:
            best, best_epoch, stale = val_acc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": LABELS,
                        "config": {"anchor": "fitted", "views": "8 camera-space views generated from fitted anchor",
                                   "camera_azimuths_deg": [-45,0,30,45,60,90,120,135], "hidden_dim": HIDDEN_DIM,
                                   "consistency_weight": CONS_WEIGHT, "seed": SEED}}, best_path)
        else: stale += 1
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} cls={history[-1]['cls_loss']:.5f} con={history[-1]['cons_loss']:.5f} val={val_acc:.5f} best={best:.5f}", flush=True)
        if stale >= PATIENCE: break
    checkpoint = torch.load(best_path, map_location=DEVICE); model.load_state_dict(checkpoint["model_state_dict"])
    evaluations = {}
    yt, yp = evaluate(model, DataLoader(FittedMVDataset(test, cam_cols, False), BATCH_SIZE, False)); evaluations["salux_fitted_anchor"] = save_eval("salux_fitted_anchor", yt, yp)
    droh = pd.read_csv(DROH_CSV)
    yt, yp = evaluate(model, DataLoader(SingleRawDataset(droh), BATCH_SIZE, False)); evaluations["droh_raw_singleview"] = save_eval("droh_raw_singleview", yt, yp)
    summary = {"valid_for_requested_protocol": True, "protocol": "fitted Salux anchor + 8 camera-space views generated from fitted anchor",
               "best_epoch": best_epoch, "best_val_acc": best, "evaluations": evaluations,
               "external_protocol": "DrOh raw single-view only; no fitting and no multiview", "classes": LABELS,
               "checkpoint": str(best_path)}
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
