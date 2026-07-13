import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from test_mvconsistency_skeleton_5cls import SingleViewGRU3D
from train_refined_skeleton import preprocess


MODEL_ROOT = Path(__file__).resolve().parent
CSV_PATH = MODEL_ROOT / "metadata" / "salux_refined_skeleton_5cls.csv"
OLD_CKPT = (
    MODEL_ROOT.parent.parent / "April" / "results"
    / "mv_consistency_anchor_8cam_model_lambda0.3" / "best_mv_consistency_anchor.pt"
)
OUT_PATH = MODEL_ROOT / "results" / "refined_skeleton_analysis.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TransferDataset(Dataset):
    def __init__(self, df, normalization, labels):
        self.df = df.reset_index(drop=True)
        self.normalization = normalization
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = preprocess(np.load(row["refined_path"]), self.normalization)
        return torch.from_numpy(x), self.labels[row["action"]]


def old_checkpoint_transfer(df, normalization):
    ckpt = torch.load(OLD_CKPT, map_location=DEVICE)
    cfg, labels7 = ckpt["config"], ckpt["label_to_idx"]
    labels5 = {name: labels7[name] for name in ["ok", "paper", "rock", "scissors", "the-finger"]}
    model = SingleViewGRU3D(
        input_dim=int(cfg.get("input_dim", 3)), hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]), num_classes=len(labels7), dropout=float(cfg["dropout"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loader = DataLoader(TransferDataset(df, normalization, labels5), batch_size=64)
    true, pred_full, pred_5 = [], [], []
    keep = torch.tensor([labels7[x] for x in labels5], device=DEVICE)
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(DEVICE))
            true.extend(y.tolist())
            pred_full.extend(logits.argmax(1).cpu().tolist())
            restricted = logits.index_select(1, keep).argmax(1)
            pred_5.extend(keep[restricted].cpu().tolist())
    return {
        "full_7class_head_acc": accuracy_score(true, pred_full),
        "restricted_5class_head_acc": accuracy_score(true, pred_5),
    }


def nearest_centroid(df, normalization):
    labels = sorted(df["action"].unique())
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    centroids = {}
    within = {}
    for label in labels:
        xs = np.stack([preprocess(np.load(p), normalization).reshape(-1)
                       for p in train[train["action"] == label]["refined_path"]])
        centroids[label] = xs.mean(0)
        within[label] = float(np.linalg.norm(xs - centroids[label], axis=1).mean())
    true, pred = [], []
    for row in test.itertuples(index=False):
        x = preprocess(np.load(row.refined_path), normalization).reshape(-1)
        pred.append(min(labels, key=lambda label: np.linalg.norm(x - centroids[label])))
        true.append(row.action)
    inter = [np.linalg.norm(centroids[a] - centroids[b]) for i, a in enumerate(labels)
             for b in labels[i + 1:]]
    return {
        "test_acc": accuracy_score(true, pred),
        "mean_within_class_distance": float(np.mean(list(within.values()))),
        "min_between_centroid_distance": float(np.min(inter)),
        "within_by_class": within,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    results = {}
    for norm in ["raw", "scale"]:
        results[norm] = {
            "old_mediapipe_checkpoint_zero_shot": old_checkpoint_transfer(df[df["split"] == "test"], norm),
            "nearest_refined_train_centroid": nearest_centroid(df, norm),
        }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
