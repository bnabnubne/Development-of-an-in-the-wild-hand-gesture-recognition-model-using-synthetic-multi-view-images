from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUT_DIR = ROOT / "results" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))
(OUT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

CLASS_NAMES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_figure(fig, stem: str):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def skeleton_2d(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (21, 3) and arr.shape != (21, 2):
        arr = arr.reshape(21, -1)
    return arr[:, :2]


def plot_skeleton(ax, points, *, color="#1f77b4", label=None, linewidth=2.2,
                  alpha=1.0, linestyle="-", joint_size=22):
    pts = skeleton_2d(points)
    for a, b in CONNECTIONS:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )
    ax.scatter(pts[:, 0], pts[:, 1], s=joint_size, color=color, alpha=alpha, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")


def set_equal_limits(ax, *point_sets, pad=0.08):
    pts = [skeleton_2d(p) for p in point_sets if p is not None]
    if not pts:
        return
    xy = np.concatenate(pts, axis=0)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    dx, dy = x1 - x0, y1 - y0
    span = max(float(dx), float(dy), 1e-6)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = span * (0.5 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)


def project_template_from_log(log_path: str | Path):
    """Return observed U, initial projected template, refined projected template, and log."""
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import batch_fit_ok as fitmod

    log_path = Path(log_path)
    log = load_json(log_path)
    raw = np.load(log["raw_path"], allow_pickle=False).astype(np.float32)
    observed = fitmod.extract_2d_from_raw(raw)
    template = np.load(log["template_path"], allow_pickle=False).astype(np.float32)
    template_centered = template - template[0]
    view = log["best_view"]
    R = np.asarray(view["R"], dtype=np.float32)
    scale = float(view["scale"])
    translation = np.asarray(view["translation"], dtype=np.float32)
    initial = fitmod.project(template_centered, R, scale, translation)
    fitted_path = Path(log.get("output_npy", ""))
    if not fitted_path.exists():
        fitted_path = log_path.parent.parent / "fitted" / f"{log['sample_id']}.npy"
    refined_3d = np.load(fitted_path, allow_pickle=False).astype(np.float32)
    refined = fitmod.project(refined_3d, R, scale, translation)
    return observed, initial, refined, log


def find_fitting_logs(limit=80):
    roots = [
        ROOT / "data" / "droh_refined",
        ROOT / "data" / "salux_refined_6cls",
        PROJECT_ROOT / "batch_ok",
        PROJECT_ROOT / "batch_paper",
        PROJECT_ROOT / "batch_rock",
        PROJECT_ROOT / "batch_scissors",
        PROJECT_ROOT / "batch_thefinger",
    ]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("**/logs/*.json")))
    return paths[:limit]
