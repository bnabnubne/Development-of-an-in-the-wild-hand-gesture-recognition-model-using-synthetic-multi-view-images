import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from test_mvconsistency_skeleton_5cls import SingleViewGRU3D
from train_refined_skeleton import GRUClassifier, SkeletonDataset, evaluate, preprocess, save_eval


MODEL_ROOT = Path(__file__).resolve().parent
CSV_PATH = MODEL_ROOT / "metadata" / "droh_refined_skeleton_5cls.csv"
OUT_DIR = MODEL_ROOT / "results" / "droh_refined_skeleton"
SALUX_CSV = MODEL_ROOT / "metadata" / "salux_refined_skeleton_5cls.csv"
OLD_CKPT = (
    MODEL_ROOT.parent.parent / "April" / "results"
    / "mv_consistency_anchor_8cam_model_lambda0.3" / "best_mv_consistency_anchor.pt"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = {
    "mediapipe_scale_baseline": {
        "checkpoint": "skeleton_mediapipe_scale_baseline/best.pt",
        "source": "mediapipe", "normalization": "scale", "multiview": False,
    },
    "mediapipe_scale_mv_consistency": {
        "checkpoint": "skeleton_mediapipe_scale_mv_consistency/best.pt",
        "source": "mediapipe", "normalization": "scale", "multiview": True,
    },
    "refined_raw_baseline": {
        "checkpoint": "skeleton_refined_raw_baseline/best.pt",
        "source": "refined", "normalization": "raw", "multiview": False,
    },
    "refined_scale_baseline": {
        "checkpoint": "skeleton_refined_scale_baseline/best.pt",
        "source": "refined", "normalization": "scale", "multiview": False,
    },
    "refined_scale_mv_consistency": {
        "checkpoint": "skeleton_refined_scale_mv_consistency/best.pt",
        "source": "refined", "normalization": "scale", "multiview": True,
    },
}


def nearest_salux_centroid(droh_df, normalization):
    salux = pd.read_csv(SALUX_CSV)
    salux = salux[salux["split"] == "train"]
    labels = sorted(salux["action"].unique())
    centroids = {}
    for label in labels:
        xs = [preprocess(np.load(p), normalization).reshape(-1)
              for p in salux[salux["action"] == label]["refined_path"]]
        centroids[label] = np.mean(xs, axis=0)
    true, pred = [], []
    for row in droh_df.itertuples(index=False):
        x = preprocess(np.load(row.refined_path), normalization).reshape(-1)
        pred.append(min(labels, key=lambda label: np.linalg.norm(x - centroids[label])))
        true.append(row.action)
    return accuracy_score(true, pred)


def old_checkpoint_on_refined(droh_df, normalization):
    ckpt = torch.load(OLD_CKPT, map_location=DEVICE)
    cfg, labels7 = ckpt["config"], ckpt["label_to_idx"]
    labels5 = {name: labels7[name] for name in ["ok", "paper", "rock", "scissors", "the-finger"]}
    model = SingleViewGRU3D(
        input_dim=int(cfg.get("input_dim", 3)), hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]), num_classes=len(labels7), dropout=float(cfg["dropout"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ds = SkeletonDataset(droh_df, "refined_path", labels5, normalization, False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    keep = torch.tensor([labels7[x] for x in labels5], device=DEVICE)
    true, full, restricted = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(DEVICE))
            true.extend(y.tolist())
            full.extend(logits.argmax(1).cpu().tolist())
            idx = logits.index_select(1, keep).argmax(1)
            restricted.extend(keep[idx].cpu().tolist())
    return {"full_7class_head_acc": accuracy_score(true, full),
            "restricted_5class_head_acc": accuracy_score(true, restricted)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    results = {}
    for name, spec in EXPERIMENTS.items():
        ckpt_path = MODEL_ROOT / "results" / spec["checkpoint"]
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        labels = ckpt["label_to_idx"]
        model = GRUClassifier(len(labels)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        ds = SkeletonDataset(
            df, f"{spec['source']}_path", labels,
            spec["normalization"], spec["multiview"],
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        class_names = [x for x, _ in sorted(labels.items(), key=lambda z: z[1])]
        acc, yt, yp = evaluate(model, loader, spec["multiview"])
        save_eval(OUT_DIR, f"{name}_anchor", acc, yt, yp, class_names)
        row = {"source": spec["source"], "normalization": spec["normalization"],
               "checkpoint": str(ckpt_path), "anchor_acc": acc}
        if spec["multiview"]:
            avg_acc, yt, yp = evaluate(model, loader, True, True)
            save_eval(OUT_DIR, f"{name}_avg8", avg_acc, yt, yp, class_names)
            row["avg8_acc"] = avg_acc
        results[name] = row
        print(name, row, flush=True)
    summary = {
        "dataset": "DrOh refined template skeleton 5-class",
        "rows": len(df),
        "classes": sorted(df["action"].unique().tolist()),
        "results": results,
        "analysis": {
            "nearest_salux_refined_centroid_raw_acc": nearest_salux_centroid(df, "raw"),
            "nearest_salux_refined_centroid_scale_acc": nearest_salux_centroid(df, "scale"),
            "old_mediapipe_checkpoint_on_refined_raw": old_checkpoint_on_refined(df, "raw"),
            "old_mediapipe_checkpoint_on_refined_scale": old_checkpoint_on_refined(df, "scale"),
        },
        "caveat": "Refinement uses a class-selected template; results include class-template prior.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
