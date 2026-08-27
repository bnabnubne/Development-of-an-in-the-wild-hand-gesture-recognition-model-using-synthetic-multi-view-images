import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math

TEMPLATE_PATH = "./templates/ok_template_joints.npy"
SALUX_2D_PATH = "./2d/ok/ok_6990.npy"
BEST_VIEW_JSON = "./debug/ok_6990_best_view.json"

OUT_DIR = Path("./debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "ok_6990_fitted_template_joints.npy"
OUT_JSON = OUT_DIR / "ok_6990_finger_refine.json"
OUT_PNG = OUT_DIR / "ok_6990_finger_refine_overlay.png"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

FINGERTIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIVOTS = {
    "thumb": 1,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

REFINE_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

ANGLE_RANGE = range(-35, 36, 5)


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def make_rotation(rx, ry, rz):
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def project_points(points3d_centered, R, scale, translation):
    pts_rot = (R @ points3d_centered.T).T
    pts_2d = pts_rot[:, :2]
    pts_proj = scale * pts_2d + translation
    return pts_proj


def mean_err(P, U, ids):
    return float(np.mean(np.linalg.norm(P[ids] - U[ids], axis=1)))


def rotate_finger_chain(T, finger_name, axis_name, angle_deg):
    T_new = T.copy()

    if axis_name == "x":
        Rloc = rot_x(angle_deg)
    elif axis_name == "y":
        Rloc = rot_y(angle_deg)
    elif axis_name == "z":
        Rloc = rot_z(angle_deg)
    else:
        raise ValueError(axis_name)

    pivot_id = PIVOTS[finger_name]
    pivot = T[pivot_id].copy()

    chain = FINGERS[finger_name]

    for j in chain:
        v = T[j] - pivot
        T_new[j] = pivot + (Rloc @ v)

    return T_new


def refine_one_finger(T_current, U, R, scale, translation, finger_name):
    tip_id = FINGERTIPS[finger_name]
    chain_ids = FINGERS[finger_name]

    P0 = project_points(T_current, R, scale, translation)
    base_tip_err = np.linalg.norm(P0[tip_id] - U[tip_id])
    base_chain_err = mean_err(P0, U, chain_ids)
    base_score = float(base_tip_err + 0.35 * base_chain_err)

    best = {
        "finger": finger_name,
        "axis": "none",
        "angle": 0,
        "score": base_score,
        "tip_err": float(base_tip_err),
        "chain_err": float(base_chain_err),
        "improved": False,
    }
    best_T = T_current.copy()

    for axis in ["x", "y", "z"]:
        for angle in ANGLE_RANGE:
            if angle == 0:
                continue

            T_candidate = rotate_finger_chain(
                T_current,
                finger_name=finger_name,
                axis_name=axis,
                angle_deg=angle
            )

            P = project_points(T_candidate, R, scale, translation)

            tip_err = np.linalg.norm(P[tip_id] - U[tip_id])
            chain_err = mean_err(P, U, chain_ids)

            score = float(tip_err + 0.35 * chain_err)

            if score < best["score"]:
                best = {
                    "finger": finger_name,
                    "axis": axis,
                    "angle": int(angle),
                    "score": score,
                    "tip_err": float(tip_err),
                    "chain_err": float(chain_err),
                    "improved": True,
                }
                best_T = T_candidate.copy()

    return best_T, best


def draw_overlay(U, P_before, P_after, title, out_path):
    plt.figure(figsize=(8, 8))

    for a, b in CONNECTIONS:
        plt.plot(
            [U[a,0], U[b,0]],
            [U[a,1], U[b,1]],
            color="red",
            linewidth=2,
            alpha=0.65
        )
    plt.scatter(U[:,0], U[:,1], color="red", s=35, label="Salux 2D")

    for a, b in CONNECTIONS:
        plt.plot(
            [P_before[a,0], P_before[b,0]],
            [P_before[a,1], P_before[b,1]],
            color="gray",
            linewidth=1.5,
            alpha=0.45
        )
    plt.scatter(P_before[:,0], P_before[:,1], color="gray", s=20, label="Before refine")

    for a, b in CONNECTIONS:
        plt.plot(
            [P_after[a,0], P_after[b,0]],
            [P_after[a,1], P_after[b,1]],
            color="blue",
            linewidth=2,
            alpha=0.85
        )
    plt.scatter(P_after[:,0], P_after[:,1], color="blue", s=35, label="After refine")

    for i in range(21):
        plt.text(U[i,0], U[i,1], str(i), color="red", fontsize=8)
        plt.text(P_after[i,0], P_after[i,1], str(i), color="blue", fontsize=8)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def main():
    T_raw = np.load(TEMPLATE_PATH).astype(np.float32)
    U = np.load(SALUX_2D_PATH).astype(np.float32)

    with open(BEST_VIEW_JSON, "r") as f:
        best_view = json.load(f)

    R = np.array(best_view["R"], dtype=np.float32)
    scale = float(best_view["scale"])
    translation = np.array(best_view["translation"], dtype=np.float32)

    if scale <= 0:
        raise ValueError(f"Scale must be positive. Got {scale}")

    T0 = T_raw - T_raw[0]

    P_before = project_points(T0, R, scale, translation)
    err_before_all = mean_err(P_before, U, list(range(21)))
    err_before_tips = mean_err(P_before, U, [4, 8, 12, 16, 20])

    print("Before refine:")
    print("  all_err :", err_before_all)
    print("  tips_err:", err_before_tips)

    T_fit = T0.copy()
    logs = []

    for finger in REFINE_FINGERS:
        T_fit, log = refine_one_finger(
            T_fit,
            U,
            R,
            scale,
            translation,
            finger
        )
        logs.append(log)
        print(log)

    P_after = project_points(T_fit, R, scale, translation)
    err_after_all = mean_err(P_after, U, list(range(21)))
    err_after_tips = mean_err(P_after, U, [4, 8, 12, 16, 20])

    result = {
        "input_template": TEMPLATE_PATH,
        "input_salux_2d": SALUX_2D_PATH,
        "input_best_view": BEST_VIEW_JSON,
        "before": {
            "all_err": err_before_all,
            "tips_err": err_before_tips,
        },
        "after": {
            "all_err": err_after_all,
            "tips_err": err_after_tips,
        },
        "finger_logs": logs,
        "note": "T_fit is centered at wrist, same coordinate convention as viewpoint search."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    np.save(OUT_NPY, T_fit)

    title = (
        f"Finger-level refine | "
        f"all {err_before_all:.4f}->{err_after_all:.4f}, "
        f"tips {err_before_tips:.4f}->{err_after_tips:.4f}"
    )

    draw_overlay(U, P_before, P_after, title, OUT_PNG)

    print()
    print("Saved fitted template:", OUT_NPY)
    print("Saved log:", OUT_JSON)
    print("Saved overlay:", OUT_PNG)


if __name__ == "__main__":
    main()