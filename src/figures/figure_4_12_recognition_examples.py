from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from figure_utils import CLASS_NAMES, OUT_DIR, plot_skeleton, save_figure, set_equal_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_paper_controlled_raw_6cls import DEVICE, GRU6, LABELS, normalize


def action_label(action):
    return "thumb" if action in {"thumbup", "thumbdown"} else action


def run_inference():
    ckpt = ROOT / "results" / "paper_controlled_raw_mv_6cls" / "best.pt"
    droh_csv = Path("./metadata/droh_baseline.csv")
    df = pd.read_csv(droh_csv).copy()
    model = GRU6().to(DEVICE)
    checkpoint = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for row in df.itertuples(index=False):
            x_np = np.load(row.input_path, allow_pickle=False).astype(np.float32)
            x = torch.from_numpy(normalize(x_np)[None]).to(DEVICE)
            probs = F.softmax(model(x)[0], dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            gt = action_label(row.action)
            pred = CLASS_NAMES[pred_idx]
            rows.append({
                "sample_id": row.sample_id,
                "raw_path": row.input_path,
                "ground_truth": gt,
                "prediction": pred,
                "confidence": float(probs[pred_idx]),
                "correct": gt == pred,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "figure_4_12_raw_mv_inference_examples.csv", index=False)
    return out


def choose_examples(df, correct=True, k=4):
    part = df[df.correct == correct].copy()
    selected = []
    used_gt = set()
    for row in part.sort_values("confidence", ascending=False).itertuples(index=False):
        if row.ground_truth in used_gt and len(used_gt) < k:
            continue
        selected.append(row)
        used_gt.add(row.ground_truth)
        if len(selected) == k:
            break
    if len(selected) < k:
        for row in part.sort_values("confidence", ascending=False).itertuples(index=False):
            if row.sample_id not in {x.sample_id for x in selected}:
                selected.append(row)
            if len(selected) == k:
                break
    return selected[:k]


def main():
    df = run_inference()
    correct = choose_examples(df, True, 4)
    failures = choose_examples(df, False, 4)
    examples = [correct, failures]
    row_titles = ["Correct predictions", "Failure cases"]

    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.0))
    for r in range(2):
        for c, ex in enumerate(examples[r]):
            ax = axes[r, c]
            raw = np.load(ex.raw_path, allow_pickle=False).astype(np.float32)
            plot_skeleton(ax, raw[:, :2], color="#54A24B" if ex.correct else "#E45756", linewidth=2.2)
            set_equal_limits(ax, raw[:, :2])
            ax.set_title(
                f"GT: {ex.ground_truth}\nPred: {ex.prediction}\nConf: {100*ex.confidence:.1f}%",
                fontsize=10,
            )
        axes[r, 0].text(
            -0.12, 0.5, row_titles[r],
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    fig.suptitle("Recognition Examples and Failure Cases", y=0.98, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.95])
    png, pdf = save_figure(fig, "figure_4_12_recognition_examples_failure_cases")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
