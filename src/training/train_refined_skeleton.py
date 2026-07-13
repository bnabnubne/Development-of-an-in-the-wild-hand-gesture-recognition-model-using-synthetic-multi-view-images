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


MODEL_ROOT = Path(__file__).resolve().parent
CSV_PATH = MODEL_ROOT / "metadata" / "salux_refined_skeleton_5cls.csv"
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
HIDDEN_DIM = 128
CONS_WEIGHT = 0.3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Recovered from April/salux_multiview3d_8cam_front with rigid Procrustes.
# Each matrix maps a column-vector world point into the original camera space.
CAMERA_ROTATIONS = np.asarray([
    [[.707106773, -.707106789, 0.], [.182195065, .182194999, .966234930], [-.683231271, -.683231272, .257662685]],
    [[1., 0., 0.], [0., .257662698, .966234927], [0., -.966234927, .257662698]],
    [[.866025368, .500000062, 0.], [-.128831339, .223142448, .966234927], [.483117530, -.836783954, .257662698]],
    [[.707106773, .707106790, 0.], [-.182195064, .182194999, .966234931], [.683231272, -.683231272, .257662685]],
    [[.500000068, .866025364, 0.], [-.223142465, .128831316, .966234926], [.836783946, -.483117543, .257662702]],
    [[0., 1., 0.], [-.257662697, 0., .966234927], [.966234927, 0., .257662697]],
    [[-.499999850, .866025490, 0.], [-.223142256, -.128831206, .966234989], [.836784132, .483117346, .257662465]],
    [[-.707106777, .707106785, 0.], [-.182195006, -.182195058, .966234930], [.683231283, .683231260, .257662685]],
], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess(x, normalization):
    x = np.asarray(x, dtype=np.float32).reshape(21, 3).copy()
    x -= x[0:1]
    if normalization == "scale":
        scale = float(np.linalg.norm(x[9] - x[0]))
        x /= max(scale, 1e-6)
    elif normalization != "raw":
        raise ValueError(normalization)
    return x


class SkeletonDataset(Dataset):
    def __init__(self, df, path_col, labels, normalization, multiview):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.labels = labels
        self.normalization = normalization
        self.multiview = multiview

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = preprocess(np.load(row[self.path_col]), self.normalization)
        y = self.labels[row["action"]]
        if not self.multiview:
            return torch.from_numpy(x), y
        views = np.stack([(r @ x.T).T for r in CAMERA_ROTATIONS]).astype(np.float32)
        views -= views[:, 0:1, :]
        return torch.from_numpy(x), torch.from_numpy(views), y


class GRUClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gru = nn.GRU(3, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.fc(z), z


def evaluate(model, loader, multiview=False, average_views=False):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for batch in loader:
            if multiview:
                anchor, views, y = batch
                if average_views:
                    b, v, j, c = views.shape
                    logits, _ = model(views.reshape(b * v, j, c).to(DEVICE))
                    logits = logits.reshape(b, v, -1).mean(1)
                else:
                    logits, _ = model(anchor.to(DEVICE))
            else:
                x, y = batch
                logits, _ = model(x.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    return accuracy_score(targets, preds), targets, preds


def save_eval(out_dir, name, acc, y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                   target_names=class_names, digits=4, zero_division=0)
    (out_dir / f"report_{name}.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / f"confmat_{name}.npy",
            confusion_matrix(y_true, y_pred, labels=range(len(class_names))))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mediapipe", "refined"], required=True)
    parser.add_argument("--normalization", choices=["raw", "scale"], required=True)
    parser.add_argument("--experiment", choices=["baseline", "mv_consistency"], required=True)
    args = parser.parse_args()
    set_seed(SEED)
    multiview = args.experiment == "mv_consistency"
    path_col = f"{args.source}_path"
    out_dir = MODEL_ROOT / "results" / f"skeleton_{args.source}_{args.normalization}_{args.experiment}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    labels = {name: i for i, name in enumerate(sorted(df["action"].unique()))}
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    loaders = {}
    for name, part, shuffle in [("train", train_df, True), ("val", val_df, False), ("test", test_df, False)]:
        ds = SkeletonDataset(part, path_col, labels, args.normalization, multiview)
        loaders[name] = DataLoader(ds, BATCH_SIZE, shuffle=shuffle, num_workers=0)

    model = GRUClassifier(len(labels)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, stale = -1., -1, 0
    best_path = out_dir / "best.pt"
    print(f"source={args.source} normalization={args.normalization} experiment={args.experiment} device={DEVICE}", flush=True)
    print(f"train/val/test={len(train_df)}/{len(val_df)}/{len(test_df)} labels={labels}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.
        for batch in loaders["train"]:
            optimizer.zero_grad()
            if multiview:
                anchor, views, y = batch
                anchor, views, y = anchor.to(DEVICE), views.to(DEVICE), y.to(DEVICE)
                logits_anchor, z_anchor = model(anchor)
                b, v, j, c = views.shape
                logits_v, z_v = model(views.reshape(b * v, j, c))
                logits_v = logits_v.reshape(b, v, -1)
                z_v = z_v.reshape(b, v, -1)
                cls_loss = (criterion(logits_anchor, y) +
                            sum(criterion(logits_v[:, i], y) for i in range(v))) / (v + 1)
                cons_loss = (1 - F.cosine_similarity(z_anchor[:, None, :], z_v, dim=2)).mean()
                loss = cls_loss + CONS_WEIGHT * cons_loss
            else:
                x, y = batch
                logits, _ = model(x.to(DEVICE))
                loss = criterion(logits, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_acc, _, _ = evaluate(model, loaders["val"], multiview)
        if val_acc > best_acc + 1e-4:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "label_to_idx": labels,
                "config": {"source": args.source, "normalization": args.normalization,
                           "experiment": args.experiment, "hidden_dim": HIDDEN_DIM,
                           "cons_weight": CONS_WEIGHT if multiview else 0., "num_views": 8 if multiview else 1},
            }, best_path)
        else:
            stale += 1
        print(f"epoch={epoch:03d} loss={loss_sum/len(loaders['train']):.4f} val_acc={val_acc:.4f} best={best_acc:.4f}", flush=True)
        if stale >= PATIENCE:
            break

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = [x for x, _ in sorted(labels.items(), key=lambda z: z[1])]
    test_acc, yt, yp = evaluate(model, loaders["test"], multiview)
    save_eval(out_dir, "test_anchor", test_acc, yt, yp, class_names)
    summary = {"source": args.source, "normalization": args.normalization,
               "experiment": args.experiment, "best_val_acc": best_acc,
               "best_epoch": best_epoch, "test_anchor_acc": test_acc,
               "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
               "classes": labels}
    if multiview:
        avg_acc, yt, yp = evaluate(model, loaders["test"], True, True)
        summary["test_avg8_acc"] = save_eval(out_dir, "test_avg8", avg_acc, yt, yp, class_names)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
