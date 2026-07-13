"""Generate deterministic defense-demo screenshots from real DrOh images."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inference_core import CONNECTIONS, GestureRecognizer, draw_result


MODEL_ROOT = Path(__file__).resolve().parents[1]
METADATA = MODEL_ROOT / "metadata/droh_rgb_skeleton_7cls.csv"
OUT_DIR = MODEL_ROOT / "results/defense_demo_examples"


def merged_label(value):
    return "thumb" if value in {"thumbup", "thumbdown"} else value


def skeleton_overlay(image, raw):
    output = image.copy()
    height, width = output.shape[:2]
    pixels = np.column_stack((raw[:, 0] * width, raw[:, 1] * height)).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(output, tuple(pixels[a]), tuple(pixels[b]), (40, 205, 255), 4, cv2.LINE_AA)
    for x, y in pixels:
        cv2.circle(output, (int(x), int(y)), 6, (35, 55, 250), -1, cv2.LINE_AA)
    return output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = GestureRecognizer()
    metadata = pd.read_csv(METADATA)
    rows = []
    for row in metadata.itertuples(index=False):
        skeleton = np.load(row.skeleton_path, allow_pickle=False)
        prediction = recognizer.predict_skeleton(skeleton)
        rows.append({
            "action": row.action,
            "true_class": merged_label(row.action),
            "sample_id": row.sample_id,
            "image_path": row.image_path,
            "raw_path": row.raw_path,
            "skeleton_path": row.skeleton_path,
            "predicted_class": prediction["class"],
            "confidence": prediction["confidence"],
            "correct": prediction["class"] == merged_label(row.action),
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "all_droh_predictions.csv", index=False)

    correct = (
        results[results.correct]
        .sort_values("confidence", ascending=False)
        .groupby("true_class", sort=False)
        .head(1)
        .head(3)
    )
    failures = (
        results[~results.correct]
        .sort_values("confidence", ascending=False)
        .groupby("true_class", sort=False)
        .head(1)
        .head(3)
    )
    chosen = pd.concat([correct, failures], ignore_index=True)
    chosen.to_csv(OUT_DIR / "selected_gallery_examples.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for index, (ax, row) in enumerate(zip(axes.flat, chosen.itertuples(index=False))):
        image = cv2.imread(row.image_path)
        raw = np.load(row.raw_path, allow_pickle=False)
        skeleton = np.load(row.skeleton_path, allow_pickle=False)
        prediction = recognizer.predict_skeleton(skeleton)
        extraction = {
            "raw_landmarks": raw,
            "skeleton": skeleton,
            "handedness": "",
            "handedness_score": 0.0,
        }
        annotated = draw_result(image, extraction, prediction)
        output_path = OUT_DIR / f"{'correct' if row.correct else 'failure'}_{row.sample_id}.jpg"
        cv2.imwrite(str(output_path), annotated)
        gallery_image = skeleton_overlay(image, raw)
        ax.imshow(cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"GT: {row.true_class} | Pred: {row.predicted_class}\nConfidence: {row.confidence * 100:.1f}%",
            color="#176D2B" if row.correct else "#A51D20",
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")
        if index == 0:
            ax.text(-0.08, 0.5, "Correct", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=13, fontweight="bold")
        if index == 3:
            ax.text(-0.08, 0.5, "Failure", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "defense_demo_gallery.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT_DIR / "defense_demo_gallery.pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUT_DIR / "defense_demo_gallery.png")


if __name__ == "__main__":
    main()
