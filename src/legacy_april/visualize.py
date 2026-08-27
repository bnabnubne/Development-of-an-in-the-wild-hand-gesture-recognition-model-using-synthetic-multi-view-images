from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SALUX_ORIG_CSV = "./metadata/salux_baseline.csv"

OUT_DIR = Path("./debug_raw_skeleton_spherical_8cam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = "test"
ACTIONS_TO_SHOW = ["thumbup", "thumbdown"]
NUM_SAMPLES_PER_CLASS = 3
SELECT_SAMPLE_IDS = []

SEED = 42

CAMERA_CONFIGS = [
    {"az":   0, "el":   0, "name": "front"},
    {"az":  45, "el":   0, "name": "front_right"},
    {"az":  90, "el":   0, "name": "side"},
    {"az": -45, "el":   0, "name": "front_left"},

    {"az":   0, "el":  60, "name": "top"},
    {"az":  45, "el":  60, "name": "top_right"},

    {"az":   0, "el": -60, "name": "bottom"},
    {"az": -45, "el": -60, "name": "bottom_left"},
]

CAMERA_RADIUS = 3.0

CENTER_BY_MEAN_BEFORE_CAMERA = True
CENTER_ON_WRIST_AFTER_CAMERA = True

PLOT_MODE = "3d"


BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17)                # palm
]


def normalize_input_skeleton(kp: np.ndarray) -> np.ndarray:
    kp = kp.astype(np.float32)

    if kp.shape != (21, 3):
        raise ValueError(f"Expected skeleton shape (21, 3), got {kp.shape}")

    if CENTER_BY_MEAN_BEFORE_CAMERA:
        kp = kp - kp.mean(axis=0, keepdims=True)

    return kp


def camera_position_from_spherical(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    x = radius * math.cos(el) * math.sin(az)
    y = -radius * math.cos(el) * math.cos(az)
    z = radius * math.sin(el)

    return np.array([x, y, z], dtype=np.float32)


def look_at_rotation_world_to_camera(cam_pos: np.ndarray, target: np.ndarray = np.zeros(3, dtype=np.float32)) -> np.ndarray:
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    z_cam_world = -forward

    world_up = np.array([0, 0, 1], dtype=np.float32)

    if abs(np.dot(z_cam_world, world_up)) > 0.98:
        world_up = np.array([0, 1, 0], dtype=np.float32)

    x_cam_world = np.cross(world_up, z_cam_world)
    x_cam_world = x_cam_world / (np.linalg.norm(x_cam_world) + 1e-8)

    y_cam_world = np.cross(z_cam_world, x_cam_world)
    y_cam_world = y_cam_world / (np.linalg.norm(y_cam_world) + 1e-8)

    R_wc = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=0).astype(np.float32)

    return R_wc


def transform_to_camera_space(points_world: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    cam_pos = camera_position_from_spherical(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        radius=CAMERA_RADIUS
    )

    R_wc = look_at_rotation_world_to_camera(cam_pos, target=np.zeros(3, dtype=np.float32))

    points_cam = (R_wc @ (points_world - cam_pos).T).T.astype(np.float32)

    if CENTER_ON_WRIST_AFTER_CAMERA:
        points_cam = points_cam - points_cam[0:1]

    return points_cam


def load_skeleton(path: str) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.shape != (21, 3):
        raise ValueError(f"Expected skeleton shape (21,3), got {x.shape} from {path}")
    return x


def get_axis_limits(skeletons):
    all_pts = np.concatenate(skeletons, axis=0)
    center = all_pts.mean(axis=0)
    max_range = np.max(np.ptp(all_pts, axis=0))
    if max_range < 1e-6:
        max_range = 1.0
    radius = max_range * 0.65
    return center, radius


def set_equal_3d_axis(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_skeleton_3d(ax, pts, title, center, radius):
    for i, j in BONES:
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            linewidth=2
        )

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18)

    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=55, marker="s")

    ax.scatter(pts[4, 0], pts[4, 1], pts[4, 2], s=75, marker="*")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    set_equal_3d_axis(ax, center, radius)

    ax.view_init(elev=20, azim=-70)


def plot_skeleton_2d(ax, pts, title, mode="xy"):
    if mode == "xy":
        a, b = 0, 1
        xlabel, ylabel = "x", "y"
    elif mode == "xz":
        a, b = 0, 2
        xlabel, ylabel = "x", "z"
    elif mode == "yz":
        a, b = 1, 2
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError(f"Unsupported plot mode: {mode}")

    for i, j in BONES:
        ax.plot([pts[i, a], pts[j, a]], [pts[i, b], pts[j, b]], linewidth=2)

    ax.scatter(pts[:, a], pts[:, b], s=18)
    ax.scatter(pts[0, a], pts[0, b], s=55, marker="s")
    ax.scatter(pts[4, a], pts[4, b], s=75, marker="*")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)


def visualize_one_sample(row):
    action = str(row["action"])
    sample_id = str(row["sample_id"])
    input_path = str(row["input_path"])

    original = load_skeleton(input_path)
    world = normalize_input_skeleton(original)

    views = [world]
    titles = ["Original\nmean-centered"]

    for i, cfg in enumerate(CAMERA_CONFIGS):
        cam_view = transform_to_camera_space(
            world,
            azimuth_deg=cfg["az"],
            elevation_deg=cfg["el"]
        )
        views.append(cam_view)
        titles.append(f"Cam {i}\n{cfg['name']}\naz={cfg['az']} el={cfg['el']}")

    if PLOT_MODE == "3d":
        fig = plt.figure(figsize=(18, 6))
        center, radius = get_axis_limits(views)

        for idx, pts in enumerate(views):
            ax = fig.add_subplot(1, 9, idx + 1, projection="3d")
            plot_skeleton_3d(ax, pts, titles[idx], center, radius)

    else:
        fig, axes = plt.subplots(1, 9, figsize=(18, 4))
        for idx, pts in enumerate(views):
            plot_skeleton_2d(axes[idx], pts, titles[idx], mode=PLOT_MODE)

    fig.suptitle(
        f"Spherical 8-camera raw skeleton preview | action={action} | sample_id={sample_id} | split={row.get('split', '')}",
        fontsize=13
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"spherical8_{action}_{sample_id}_{PLOT_MODE}.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_path}")


def main():
    random.seed(SEED)

    df = pd.read_csv(SALUX_ORIG_CSV)

    required = {"action", "sample_id", "input_path", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in baseline CSV: {missing}")

    if SPLIT is not None:
        df = df[df["split"] == SPLIT].copy()

    if SELECT_SAMPLE_IDS:
        ids = [str(x) for x in SELECT_SAMPLE_IDS]
        df = df[df["sample_id"].astype(str).isin(ids)].copy()
    else:
        df = df[df["action"].isin(ACTIONS_TO_SHOW)].copy()

    if len(df) == 0:
        raise ValueError("No samples found. Check SPLIT, ACTIONS_TO_SHOW, or SELECT_SAMPLE_IDS.")

    selected_rows = []

    if SELECT_SAMPLE_IDS:
        selected_rows = [row for _, row in df.iterrows()]
    else:
        for action in ACTIONS_TO_SHOW:
            sub = df[df["action"] == action].copy()
            if len(sub) == 0:
                print(f"[WARN] No samples for action={action}")
                continue

            n = min(NUM_SAMPLES_PER_CLASS, len(sub))
            sub = sub.sample(n=n, random_state=SEED)
            selected_rows.extend([row for _, row in sub.iterrows()])

    print("CSV:", SALUX_ORIG_CSV)
    print("Selected samples:", len(selected_rows))
    print("Output dir:", OUT_DIR)
    print("Camera configs:")
    for i, cfg in enumerate(CAMERA_CONFIGS):
        print(f"  Cam {i}: {cfg}")

    for row in selected_rows:
        visualize_one_sample(row)

    print("\nDONE")


if __name__ == "__main__":
    main()