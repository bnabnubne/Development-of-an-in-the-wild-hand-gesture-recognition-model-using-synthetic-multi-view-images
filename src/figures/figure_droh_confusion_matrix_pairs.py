from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_utils import OUT_DIR, save_figure


CLASS_LABELS = ["OK", "Paper", "Rock", "Scissors", "The-finger", "Thumb"]
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def row_normalize(confusion: np.ndarray) -> np.ndarray:
    confusion = confusion.astype(np.float64)
    totals = confusion.sum(axis=1, keepdims=True)
    return np.divide(
        confusion,
        totals,
        out=np.zeros_like(confusion),
        where=totals != 0,
    ) * 100.0


def draw_matrix(ax, path: Path, panel_title: str):
    matrix = row_normalize(np.load(path, allow_pickle=False))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="equal")

    ticks = np.arange(len(CLASS_LABELS))
    ax.set_xticks(ticks, CLASS_LABELS, rotation=35, ha="right")
    ax.set_yticks(ticks, CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(panel_title, pad=12, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", length=0)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if value >= 50 else "#222222",
            )

    return image


def make_pair(specs, output_stem: str):
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.25))
    image = None
    for ax, (relative_path, panel_title) in zip(axes, specs):
        image = draw_matrix(ax, RESULTS_ROOT / relative_path, panel_title)

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.18, top=0.90, wspace=0.30)
    colorbar_ax = fig.add_axes([0.925, 0.19, 0.018, 0.675])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Percentage (%)")
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])

    png, pdf = save_figure(fig, output_stem)
    plt.close(fig)
    print(png)
    print(pdf)


def main():
    make_pair(
        [
            (
                "paper_controlled_raw_baseline_6cls/confmat_droh_raw.npy",
                "(a) Raw baseline – DrOh raw",
            ),
            (
                "paper_controlled_raw_mv_6cls/confmat_droh_raw.npy",
                "(b) Raw + 8-view MV – DrOh raw",
            ),
        ],
        "droh_confusion_matrix_raw_baseline_vs_multiview",
    )

    make_pair(
        [
            (
                "fitted_anchor_multiview_6cls/confmat_droh_raw.npy",
                "(c) Fitted + 8-view MV – DrOh raw",
            ),
            (
                "fitted_anchor_multiview_6cls/confmat_droh_fitted_oracle.npy",
                "(d) Fitted + 8-view MV – DrOh oracle-fitted",
            ),
        ],
        "droh_confusion_matrix_fitted_raw_vs_oracle",
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()
