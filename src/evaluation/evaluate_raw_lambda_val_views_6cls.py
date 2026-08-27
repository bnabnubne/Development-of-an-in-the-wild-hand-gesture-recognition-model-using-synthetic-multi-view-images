
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_defense_ablation_6cls as train


ROOT = Path(__file__).resolve().parent / "results/defense_experiments_6cls"
OUT = ROOT / "raw_lambda_sweep"
WEIGHTS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
SEEDS = [0, 1, 42]


def tag(weight, seed):
    return f"gru_wrist_middle_views8_lambda{str(weight).replace('.', 'p')}_seed{seed}"


def main():
    mv = pd.read_csv(train.MV_CSV)
    val = mv[mv.split == "val"].copy()
    dataset = train.MultiViewDataset(val, list(range(8)), "wrist_middle")
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    rows = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            model = train.make_model("gru").to(train.DEVICE)
            checkpoint = torch.load(ROOT / tag(weight, seed) / "best.pt", map_location=train.DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
            anchor_correct = 0; view_correct = np.zeros(8, dtype=np.int64); rows_seen = 0
            cosine_sum = 0.0
            with torch.no_grad():
                for anchor, views, y in loader:
                    anchor, views, y = anchor.to(train.DEVICE), views.to(train.DEVICE), y.to(train.DEVICE)
                    anchor_logits, anchor_feature = model(anchor)
                    b, v, j, c = views.shape
                    logits, features = model(views.reshape(b*v, j, c))
                    logits = logits.reshape(b, v, -1); features = features.reshape(b, v, -1)
                    anchor_correct += int((anchor_logits.argmax(1) == y).sum())
                    view_correct += (logits.argmax(2) == y[:, None]).sum(0).cpu().numpy()
                    cosine_sum += float((1-torch.nn.functional.cosine_similarity(anchor_feature[:,None,:],features,dim=2)).sum())
                    rows_seen += b
            anchor_acc = anchor_correct / rows_seen
            view_acc = view_correct / rows_seen
            rows.append({
                "lambda": weight, "seed": seed, "rows": rows_seen,
                "anchor_accuracy": anchor_acc, "mean_view_accuracy": float(view_acc.mean()),
                "all9_accuracy": float((anchor_acc + view_acc.sum()) / 9),
                "worst_view_accuracy": float(view_acc.min()),
                "mean_feature_inconsistency": cosine_sum / (rows_seen * 8),
                **{f"view_{i}_accuracy": float(value) for i, value in enumerate(view_acc)},
            })
    runs = pd.DataFrame(rows)
    aggregate = runs.groupby("lambda").agg(
        anchor_mean=("anchor_accuracy","mean"), anchor_std=("anchor_accuracy","std"),
        view_mean=("mean_view_accuracy","mean"), view_std=("mean_view_accuracy","std"),
        all9_mean=("all9_accuracy","mean"), all9_std=("all9_accuracy","std"),
        worst_view_mean=("worst_view_accuracy","mean"),
        feature_inconsistency_mean=("mean_feature_inconsistency","mean"),
    ).reset_index()
    selected = float(aggregate.loc[aggregate.all9_mean.idxmax(),"lambda"])
    runs.to_csv(OUT/"salux_val_view_metrics_all_runs.csv",index=False)
    aggregate.to_csv(OUT/"salux_val_view_metrics_aggregate.csv",index=False)
    (OUT/"salux_val_view_selection.json").write_text(json.dumps({
        "selection_metric":"mean accuracy across Salux validation anchor + eight held-out synthetic views",
        "selected_lambda":selected,"rows":len(val),"seeds":SEEDS,
    },indent=2))
    print(aggregate.to_string(index=False,float_format=lambda x:f"{100*x:.3f}"))
    print("selected_lambda_all9",selected)


if __name__ == "__main__": main()
