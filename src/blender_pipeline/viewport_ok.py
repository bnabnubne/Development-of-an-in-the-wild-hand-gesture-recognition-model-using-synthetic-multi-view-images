import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math

TEMPLATE_PATH = "./templates/ok_template_joints.npy"
SALUX_2D_PATH = "./2d/ok/ok_6990.npy"

OUT_DIR = Path("./debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "ok_6990_fitted_template_joints.npy"
OUT_JSON = OUT_DIR / "ok_6990_fit_log.json"
OUT_PNG = OUT_DIR / "ok_6990_fit_overlay.png"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

PALM_IDS = [0, 5, 9, 13, 17]
ALL_IDS = list(range(21))
TIP_IDS = [4, 8, 12, 16, 20]

FINGER_CHAINS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_PIVOTS = {
    "thumb": 1,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

JOINT_GROUPS = {
    "thumb_mcp": {
        "pivot": 1,
        "affected": [2, 3, 4],
        "target_ids": [2, 3, 4],
    },
    "thumb_ip": {
        "pivot": 2,
        "affected": [3, 4],
        "target_ids": [3, 4],
    },
    "thumb_tip": {
        "pivot": 3,
        "affected": [4],
        "target_ids": [4],
    },

    "index_mcp": {
        "pivot": 5,
        "affected": [6, 7, 8],
        "target_ids": [6, 7, 8],
    },
    "index_pip": {
        "pivot": 6,
        "affected": [7, 8],
        "target_ids": [7, 8],
    },
    "index_dip": {
        "pivot": 7,
        "affected": [8],
        "target_ids": [8],
    },

    "middle_mcp": {
        "pivot": 9,
        "affected": [10, 11, 12],
        "target_ids": [10, 11, 12],
    },
    "middle_pip": {
        "pivot": 10,
        "affected": [11, 12],
        "target_ids": [11, 12],
    },
    "middle_dip": {
        "pivot": 11,
        "affected": [12],
        "target_ids": [12],
    },

    "ring_mcp": {
        "pivot": 13,
        "affected": [14, 15, 16],
        "target_ids": [14, 15, 16],
    },
    "ring_pip": {
        "pivot": 14,
        "affected": [15, 16],
        "target_ids": [15, 16],
    },
    "ring_dip": {
        "pivot": 15,
        "affected": [16],
        "target_ids": [16],
    },

    "pinky_mcp": {
        "pivot": 17,
        "affected": [18, 19, 20],
        "target_ids": [18, 19, 20],
    },
    "pinky_pip": {
        "pivot": 18,
        "affected": [19, 20],
        "target_ids": [19, 20],
    },
    "pinky_dip": {
        "pivot": 19,
        "affected": [20],
        "target_ids": [20],
    },
}

RX_VALUES = range(-90, 91, 10)
RY_VALUES = range(-90, 91, 10)
RZ_VALUES = range(-180, 181, 10)

WHOLE_FINGER_ANGLE_RANGE = range(-35, 36, 5)

JOINT_ANGLE_RANGE = range(-25, 26, 5)

HIERARCHICAL_PASSES = 2


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
    return scale * pts_2d + translation


def fit_similarity_2d_positive(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    denom = np.sum(src_c ** 2)
    if denom < 1e-8:
        return None, None, None

    s = np.sum(src_c * dst_c) / denom

    if s <= 0:
        return None, None, None

    t = dst_mean - s * src_mean
    aligned = s * src + t

    return float(s), t.astype(np.float32), aligned


def mean_err(P, U, ids):
    return float(np.mean(np.linalg.norm(P[ids] - U[ids], axis=1)))


def total_score(P, U):
    return (
        0.45 * mean_err(P, U, ALL_IDS)
        + 0.35 * mean_err(P, U, TIP_IDS)
        + 0.20 * mean_err(P, U, PALM_IDS)
    )


def search_best_view(T0, U):
    best = None

    for rx in RX_VALUES:
        for ry in RY_VALUES:
            for rz in RZ_VALUES:
                R = make_rotation(rx, ry, rz)
                T_rot = (R @ T0.T).T
                P = T_rot[:, :2]

                s, t, _ = fit_similarity_2d_positive(
                    P[PALM_IDS],
                    U[PALM_IDS]
                )

                if s is None:
                    continue

                P_aligned = s * P + t

                err_palm = mean_err(P_aligned, U, PALM_IDS)
                err_all = mean_err(P_aligned, U, ALL_IDS)
                err_tips = mean_err(P_aligned, U, TIP_IDS)

                score = 0.65 * err_palm + 0.25 * err_all + 0.10 * err_tips

                if best is None or score < best["score"]:
                    best = {
                        "score": float(score),
                        "err_palm": float(err_palm),
                        "err_all": float(err_all),
                        "err_tips": float(err_tips),
                        "rx": float(rx),
                        "ry": float(ry),
                        "rz": float(rz),
                        "scale": float(s),
                        "translation": t.tolist(),
                        "R": R.tolist(),
                    }

    if best is None:
        raise RuntimeError("No valid positive-scale viewpoint found.")

    return best


def rotate_points_about_pivot(T, pivot_id, affected_ids, axis, angle_deg):
    T_new = T.copy()

    if axis == "x":
        Rloc = rot_x(angle_deg)
    elif axis == "y":
        Rloc = rot_y(angle_deg)
    elif axis == "z":
        Rloc = rot_z(angle_deg)
    else:
        raise ValueError(axis)

    pivot = T[pivot_id].copy()

    for j in affected_ids:
        v = T[j] - pivot
        T_new[j] = pivot + (Rloc @ v)

    return T_new


def refine_group(T_current, U, R, scale, translation, group_name, group_def, angle_range):
    base_P = project_points(T_current, R, scale, translation)
    base_score = total_score(base_P, U)

    base_local = mean_err(base_P, U, group_def["target_ids"])
    base = base_score + 0.50 * base_local

    best = {
        "group": group_name,
        "axis": "none",
        "angle": 0,
        "score": float(base),
        "local_err": float(base_local),
        "improved": False,
    }
    best_T = T_current.copy()

    for axis in ["x", "y", "z"]:
        for angle in angle_range:
            if angle == 0:
                continue

            T_candidate = rotate_points_about_pivot(
                T_current,
                pivot_id=group_def["pivot"],
                affected_ids=group_def["affected"],
                axis=axis,
                angle_deg=angle,
            )

            P = project_points(T_candidate, R, scale, translation)

            local_err = mean_err(P, U, group_def["target_ids"])
            score = total_score(P, U) + 0.50 * local_err

            if score < best["score"]:
                best = {
                    "group": group_name,
                    "axis": axis,
                    "angle": int(angle),
                    "score": float(score),
                    "local_err": float(local_err),
                    "improved": True,
                }
                best_T = T_candidate.copy()

    return best_T, best


def refine_whole_finger(T_current, U, R, scale, translation, finger_name):
    chain = FINGER_CHAINS[finger_name]
    pivot = FINGER_PIVOTS[finger_name]

    group_def = {
        "pivot": pivot,
        "affected": chain,
        "target_ids": chain,
    }

    return refine_group(
        T_current,
        U,
        R,
        scale,
        translation,
        f"whole_{finger_name}",
        group_def,
        WHOLE_FINGER_ANGLE_RANGE,
    )


def draw_overlay(U, P_initial, P_final, title, out_path):
    plt.figure(figsize=(8, 8))

    for a, b in CONNECTIONS:
        plt.plot([U[a,0], U[b,0]], [U[a,1], U[b,1]], color="red", linewidth=2, alpha=0.7)
    plt.scatter(U[:,0], U[:,1], color="red", s=35, label="Salux 2D")

    for a, b in CONNECTIONS:
        plt.plot([P_initial[a,0], P_initial[b,0]], [P_initial[a,1], P_initial[b,1]], color="gray", linewidth=1.4, alpha=0.45)
    plt.scatter(P_initial[:,0], P_initial[:,1], color="gray", s=20, label="Initial projection")

    for a, b in CONNECTIONS:
        plt.plot([P_final[a,0], P_final[b,0]], [P_final[a,1], P_final[b,1]], color="blue", linewidth=2, alpha=0.9)
    plt.scatter(P_final[:,0], P_final[:,1], color="blue", s=35, label="Fitted template")

    for i in range(21):
        plt.text(U[i,0], U[i,1], str(i), color="red", fontsize=8)
        plt.text(P_final[i,0], P_final[i,1], str(i), color="blue", fontsize=8)

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

    if T_raw.shape != (21, 3):
        raise ValueError(T_raw.shape)
    if U.shape != (21, 2):
        raise ValueError(U.shape)

    T0 = T_raw - T_raw[0]

    best_view = search_best_view(T0, U)

    R = np.array(best_view["R"], dtype=np.float32)
    scale = float(best_view["scale"])
    translation = np.array(best_view["translation"], dtype=np.float32)

    P_initial = project_points(T0, R, scale, translation)

    print("\n=== BEST VIEW ===")
    print(json.dumps(best_view, indent=2))

    print("\n=== INITIAL ERRORS ===")
    print("all :", mean_err(P_initial, U, ALL_IDS))
    print("tips:", mean_err(P_initial, U, TIP_IDS))
    print("palm:", mean_err(P_initial, U, PALM_IDS))

    T_fit = T0.copy()
    whole_logs = []

    print("\n=== WHOLE FINGER REFINE ===")
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        T_fit, log = refine_whole_finger(T_fit, U, R, scale, translation, finger)
        whole_logs.append(log)
        print(log)

    hier_logs = []

    print("\n=== HIERARCHICAL JOINT REFINE ===")
    ordered_groups = [
        "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip",

        "middle_mcp", "middle_pip", "middle_dip",
        "ring_mcp", "ring_pip", "ring_dip",
        "pinky_mcp", "pinky_pip", "pinky_dip",
    ]

    for pass_id in range(HIERARCHICAL_PASSES):
        print(f"\n--- pass {pass_id + 1} ---")
        for group_name in ordered_groups:
            T_fit, log = refine_group(
                T_fit,
                U,
                R,
                scale,
                translation,
                group_name,
                JOINT_GROUPS[group_name],
                JOINT_ANGLE_RANGE,
            )
            log["pass"] = pass_id + 1
            hier_logs.append(log)
            print(log)

    P_final = project_points(T_fit, R, scale, translation)

    final_errors = {
        "all": mean_err(P_final, U, ALL_IDS),
        "tips": mean_err(P_final, U, TIP_IDS),
        "palm": mean_err(P_final, U, PALM_IDS),
    }

    initial_errors = {
        "all": mean_err(P_initial, U, ALL_IDS),
        "tips": mean_err(P_initial, U, TIP_IDS),
        "palm": mean_err(P_initial, U, PALM_IDS),
    }

    print("\n=== FINAL ERRORS ===")
    print(json.dumps(final_errors, indent=2))

    np.save(OUT_NPY, T_fit)

    result = {
        "template_path": TEMPLATE_PATH,
        "salux_2d_path": SALUX_2D_PATH,
        "best_view": best_view,
        "initial_errors": initial_errors,
        "final_errors": final_errors,
        "whole_finger_logs": whole_logs,
        "hierarchical_logs": hier_logs,
        "output_npy": str(OUT_NPY),
        "note": "Fitted template joints are centered at wrist. Use same best_view R, scale, translation for projection/render alignment."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    title = (
        f"Template fitting OK | "
        f"all {initial_errors['all']:.4f}->{final_errors['all']:.4f}, "
        f"tips {initial_errors['tips']:.4f}->{final_errors['tips']:.4f}"
    )

    draw_overlay(U, P_initial, P_final, title, OUT_PNG)

    print("\nSaved fitted joints:", OUT_NPY)
    print("Saved log:", OUT_JSON)
    print("Saved overlay:", OUT_PNG)


if __name__ == "__main__":
    main()