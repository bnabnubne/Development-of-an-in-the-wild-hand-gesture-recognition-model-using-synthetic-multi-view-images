"""Complete the four-domain evaluation of the already-trained fitted+Blender8 checkpoint."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from train_fitted_baseline_6cls import BATCH_SIZE, DEVICE, GRU6, LABELS, SkeletonDataset


MODEL_ROOT = Path(__file__).resolve().parent
SALUX_CSV = MODEL_ROOT / "metadata/salux_refined_skeleton_6cls.csv"
DROH_CSV = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls.csv"
OUT_DIR = MODEL_ROOT / "results/fitted_anchor_multiview_6cls"
CKPT = OUT_DIR / "best.pt"


def main():
    model = GRU6().to(DEVICE)
    checkpoint = torch.load(CKPT, map_location=DEVICE, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    salux, droh = pd.read_csv(SALUX_CSV), pd.read_csv(DROH_CSV); test = salux[salux.split == "test"].copy()
    evaluations = {}
    for name, frame, col in [("salux_fitted", test, "refined_path"), ("salux_raw", test, "raw_path"),
                             ("droh_raw", droh, "raw_path"), ("droh_fitted_oracle", droh, "refined_path")]:
        yt, yp = [], []
        with torch.no_grad():
            for x, y in DataLoader(SkeletonDataset(frame, col), BATCH_SIZE, False):
                yp.extend(model(x.to(DEVICE))[0].argmax(1).cpu().tolist()); yt.extend(y.tolist())
        (OUT_DIR / f"report_{name}.txt").write_text(classification_report(
            yt, yp, labels=range(6), target_names=list(LABELS), digits=6, zero_division=0), encoding="utf-8")
        np.save(OUT_DIR / f"confmat_{name}.npy", confusion_matrix(yt, yp, labels=range(6)))
        evaluations[name] = {"accuracy": accuracy_score(yt, yp), "balanced_accuracy": balanced_accuracy_score(yt, yp),
                             "macro_f1": f1_score(yt, yp, average="macro"),
                             "weighted_f1": f1_score(yt, yp, average="weighted"), "rows": len(yt)}
    summary_path = OUT_DIR / "summary.json"; summary = json.loads(summary_path.read_text())
    summary["evaluations"] = evaluations
    summary["warning"] = "DrOh fitted is a ground-truth class-template oracle diagnostic."
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
