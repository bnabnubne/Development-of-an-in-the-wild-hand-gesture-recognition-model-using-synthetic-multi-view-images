import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


POSE_NAME = "ok"
PROJECT_ROOT = Path(".")
RAW_POSE_DIR = Path(f"./dataset/skeleton_raw/{POSE_NAME}")
TEMPLATE_PATH = PROJECT_ROOT / "templates" / f"{POSE_NAME}_template_joints.npy"

OUT_ROOT = PROJECT_ROOT / f"batch_{POSE_NAME}"
OUT_2D_DIR = OUT_ROOT / "2d"
OUT_FIT_DIR = OUT_ROOT / "fitted"
OUT_LOG_DIR = OUT_ROOT / "logs"
OUT_OVERLAY_DIR = OUT_ROOT / "overlays"

for d in [OUT_2D_DIR, OUT_FIT_DIR, OUT_LOG_DIR, OUT_OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_FILES = None
SAVE_OVERLAY = False


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

PALM_IDS = [0, 5, 9, 13, 17]
ALIGN_IDS = [0, 5, 9, 13, 17]
ALIGN_WEIGHTS = np.array([5.0, 1.2, 1.5, 1.5, 1.2], dtype=np.float32)
TIP_IDS = [4, 8, 12, 16, 20]
ALL_IDS = list(range(21))

JOINT_GROUPS = {
    "thumb_mcp": {"pivot": 1, "affected": [2, 3, 4], "target_ids": [2, 3, 4]},
    "thumb_ip": {"pivot": 2, "affected": [3, 4], "target_ids": [3, 4]},
    "thumb_tip": {"pivot": 3, "affected": [4], "target_ids": [4]},

    "index_mcp": {"pivot": 5, "affected": [6, 7, 8], "target_ids": [6, 7, 8]},
    "index_pip": {"pivot": 6, "affected": [7, 8], "target_ids": [7, 8]},
    "index_dip": {"pivot": 7, "affected": [8], "target_ids": [8]},

    "middle_mcp": {"pivot": 9, "affected": [10, 11, 12], "target_ids": [10, 11, 12]},
    "middle_pip": {"pivot": 10, "affected": [11, 12], "target_ids": [11, 12]},
    "middle_dip": {"pivot": 11, "affected": [12], "target_ids": [12]},

    "ring_mcp": {"pivot": 13, "affected": [14, 15, 16], "target_ids": [14, 15, 16]},
    "ring_pip": {"pivot": 14, "affected": [15, 16], "target_ids": [15, 16]},
    "ring_dip": {"pivot": 15, "affected": [16], "target_ids": [16]},

    "pinky_mcp": {"pivot": 17, "affected": [18, 19, 20], "target_ids": [18, 19, 20]},
    "pinky_pip": {"pivot": 18, "affected": [19, 20], "target_ids": [19, 20]},
    "pinky_dip": {"pivot": 19, "affected": [20], "target_ids": [20]},
}

ORDERED_GROUPS = [
    "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

FINGER_SCALE_GROUPS = {
    "thumb_scale": {"base": 1, "affected": [2, 3, 4], "target_ids": [2, 3, 4]},
    "index_scale": {"base": 5, "affected": [6, 7, 8], "target_ids": [6, 7, 8]},
    "middle_scale": {"base": 9, "affected": [10, 11, 12], "target_ids": [10, 11, 12]},
    "ring_scale": {"base": 13, "affected": [14, 15, 16], "target_ids": [14, 15, 16]},
    "pinky_scale": {"base": 17, "affected": [18, 19, 20], "target_ids": [18, 19, 20]},
}

FINGER_SCALE_ORDER = [
    "thumb_scale",
    "index_scale",
    "middle_scale",
    "ring_scale",
    "pinky_scale",
]

PIP_DIP_GROUPS = [
    "thumb_ip", "thumb_tip",
    "index_pip", "index_dip",
    "middle_pip", "middle_dip",
    "ring_pip", "ring_dip",
    "pinky_pip", "pinky_dip",
]

RX_VALUES = range(-90, 91, 10)
RY_VALUES = range(-90, 91, 10)
RZ_VALUES = range(-180, 181, 10)

LOCAL_VIEW_RADIUS = 10
LOCAL_VIEW_STEP = 2
FINE_VIEW_RADIUS = 2
FINE_VIEW_STEP = 1
FINGER_SCALE_VALUES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

FINGER_CONNECTIONS = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)],
}

THUMB_INDEX_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
]

EXTENDED_FINGER_CONNECTIONS = [
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def rot_x(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def make_R(rx, ry, rz):
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def extract_2d_from_raw(raw):
    if raw.shape != (21, 3):
        raise ValueError(f"Expected raw shape (21,3), got {raw.shape}")
    return raw[:, :2].astype(np.float32)


def project_raw(T, R):
    return (R @ T.T).T[:, :2]


def project(T, R, s, t):
    return s * project_raw(T, R) + t


def mean_err(P, U, ids):
    return float(np.mean(np.linalg.norm(P[ids] - U[ids], axis=1)))


def unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def mean_bone_dir_err(P, U, connections=CONNECTIONS):
    errs = []
    for a, b in connections:
        dp = unit(P[b] - P[a])
        du = unit(U[b] - U[a])
        if dp is None or du is None:
            continue
        cos_sim = float(np.clip(np.dot(dp, du), -1.0, 1.0))
        errs.append(1.0 - cos_sim)
    if not errs:
        return 1.0
    return float(np.mean(errs))


def finger_dir_err(P, U, finger_connections):
    return mean_bone_dir_err(P, U, finger_connections)


def pair_direction_error(P, U, a, b):
    dp = unit(P[b] - P[a])
    du = unit(U[b] - U[a])
    if dp is None or du is None:
        return 1.0
    cos_sim = float(np.clip(np.dot(dp, du), -1.0, 1.0))
    return 1.0 - cos_sim


def total_score(P, U):
    return (
        0.45 * mean_err(P, U, ALL_IDS)
        + 0.35 * mean_err(P, U, TIP_IDS)
        + 0.20 * mean_err(P, U, PALM_IDS)
    )


def fit_similarity_weighted(src, dst, weights):
    w = np.asarray(weights, dtype=np.float32)
    w = w / (np.sum(w) + 1e-8)

    sm = np.sum(src * w[:, None], axis=0)
    dm = np.sum(dst * w[:, None], axis=0)

    sc = src - sm
    dc = dst - dm

    denom = np.sum(w[:, None] * (sc ** 2))
    if denom < 1e-8:
        return None, None

    s = np.sum(w[:, None] * sc * dc) / denom
    if s <= 0:
        return None, None

    t = dm - s * sm
    return float(s), t.astype(np.float32)


def angle_values(center, radius, step, lo, hi):
    start = max(lo, center - radius)
    stop = min(hi, center + radius)
    vals = []
    v = start
    while v <= stop + 1e-6:
        vals.append(float(v))
        v += step
    return vals


def evaluate_view(T, U, rx, ry, rz, stage):
    R = make_R(rx, ry, rz)
    P0 = project_raw(T, R)

    s, t = fit_similarity_weighted(P0[ALIGN_IDS], U[ALIGN_IDS], ALIGN_WEIGHTS)
    if s is None:
        return None

    P = s * P0 + t

    err_all = mean_err(P, U, ALL_IDS)
    err_tip = mean_err(P, U, TIP_IDS)
    err_palm = mean_err(P, U, PALM_IDS)
    err_dir_all = mean_bone_dir_err(P, U)
    err_dir_ti = finger_dir_err(P, U, THUMB_INDEX_CONNECTIONS)
    err_dir_ext = finger_dir_err(P, U, EXTENDED_FINGER_CONNECTIONS)

    score = (
        0.25 * err_all
        + 0.25 * err_palm
        + 0.15 * err_tip
        + 0.15 * err_dir_all
        + 0.07 * err_dir_ext
        + 0.03 * err_dir_ti
    )

    return {
        "score": float(score),
        "stage": stage,
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz),
        "R": R.tolist(),
        "scale": float(s),
        "translation": t.tolist(),
        "err_all": err_all,
        "err_tips": err_tip,
        "err_palm": err_palm,
        "err_dir_all": err_dir_all,
        "err_dir_thumb_index": err_dir_ti,
        "err_dir_extended": err_dir_ext,
    }


def search_view_grid(T, U, rx_values, ry_values, rz_values, stage, initial_best=None):
    best = initial_best
    for rx in rx_values:
        for ry in ry_values:
            for rz in rz_values:
                cand = evaluate_view(T, U, rx, ry, rz, stage)
                if cand is None:
                    continue
                if best is None or cand["score"] < best["score"]:
                    best = cand
    return best


def search_best_view(T, U):
    best = search_view_grid(T, U, RX_VALUES, RY_VALUES, RZ_VALUES, stage="coarse")
    if best is None:
        raise RuntimeError("No valid view found")

    best = search_view_grid(
        T,
        U,
        angle_values(best["rx"], LOCAL_VIEW_RADIUS, LOCAL_VIEW_STEP, -90, 90),
        angle_values(best["ry"], LOCAL_VIEW_RADIUS, LOCAL_VIEW_STEP, -90, 90),
        angle_values(best["rz"], LOCAL_VIEW_RADIUS, LOCAL_VIEW_STEP, -180, 180),
        stage="local",
        initial_best=best,
    )

    best = search_view_grid(
        T,
        U,
        angle_values(best["rx"], FINE_VIEW_RADIUS, FINE_VIEW_STEP, -90, 90),
        angle_values(best["ry"], FINE_VIEW_RADIUS, FINE_VIEW_STEP, -90, 90),
        angle_values(best["rz"], FINE_VIEW_RADIUS, FINE_VIEW_STEP, -180, 180),
        stage="fine",
        initial_best=best,
    )

    return best


def rot_axis_angle(axis, angle_deg):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float32)


def rotate_about_pivot_with_matrix(T, pivot_id, affected_ids, Rloc):
    out = T.copy()
    pivot = T[pivot_id].copy()
    for j in affected_ids:
        out[j] = pivot + (Rloc @ (T[j] - pivot))
    return out


def angle_2d(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    cross = a[0] * b[1] - a[1] * b[0]
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.degrees(math.atan2(cross, dot))


def angle_candidates(max_abs_angle, step, desired_angle):
    vals = set(range(-max_abs_angle, max_abs_angle + 1, step))
    guided = int(round(float(np.clip(desired_angle, -max_abs_angle, max_abs_angle)) / step) * step)
    guided_inv = int(round(float(np.clip(-desired_angle, -max_abs_angle, max_abs_angle)) / step) * step)
    vals.add(guided)
    vals.add(guided_inv)
    return sorted(vals)


def refine_group(T, U, R, s, t, group_name, group_def, camera_axis_template):
    P_base = project(T, R, s, t)

    pivot_id = group_def["pivot"]
    target_tip = group_def["target_ids"][-1]
    base_local = mean_err(P_base, U, group_def["target_ids"])
    base_dir = pair_direction_error(P_base, U, pivot_id, target_tip)

    target_vec = U[target_tip] - U[pivot_id]
    current_vec = P_base[target_tip] - P_base[pivot_id]
    desired_angle = angle_2d(current_vec, target_vec)

    if group_name.startswith(("thumb", "index")):
        beta_local = 0.80
        beta_dir = 0.15
        reg_w = 0.0008
        max_abs_angle = 35
    else:
        beta_local = 0.50
        beta_dir = 0.20
        reg_w = 0.0015
        max_abs_angle = 15

    angle_range = angle_candidates(max_abs_angle, 5, desired_angle)
    base_score = beta_local * base_local + beta_dir * base_dir

    best_T = T
    best = {
        "group": group_name,
        "axis": "none",
        "angle": 0,
        "score": float(base_score),
        "local_err": float(base_local),
        "dir_err": float(base_dir),
        "desired_angle_2d": float(desired_angle),
        "improved": False,
    }

    for angle in angle_range:
        if angle == 0:
            continue

        Rloc = rot_axis_angle(camera_axis_template, angle)
        cand = rotate_about_pivot_with_matrix(T, pivot_id, group_def["affected"], Rloc)
        P = project(cand, R, s, t)

        local_err = mean_err(P, U, group_def["target_ids"])
        dir_err = pair_direction_error(P, U, pivot_id, target_tip)
        score = beta_local * local_err + beta_dir * dir_err + reg_w * abs(angle)

        if score < best["score"]:
            best_T = cand
            best = {
                "group": group_name,
                "axis": "camera",
                "angle": int(angle),
                "score": float(score),
                "local_err": float(local_err),
                "dir_err": float(dir_err),
                "desired_angle_2d": float(desired_angle),
                "improved": True,
            }

    return best_T, best


def scale_finger_chain(T, base_id, affected_ids, alpha):
    out = T.copy()
    base = T[base_id].copy()
    for j in affected_ids:
        out[j] = base + alpha * (T[j] - base)
    return out


def refine_finger_scale(T, U, R, s, t, group_name, group_def):
    P_base = project(T, R, s, t)

    base_id = group_def["base"]
    target_tip = group_def["target_ids"][-1]
    base_local = mean_err(P_base, U, group_def["target_ids"])
    base_dir = pair_direction_error(P_base, U, base_id, target_tip)
    base_tip = mean_err(P_base, U, [target_tip])

    beta_local = 0.70
    beta_tip = 0.50
    beta_dir = 0.10
    reg_w = 0.02
    base_score = beta_local * base_local + beta_tip * base_tip + beta_dir * base_dir

    best_T = T
    best = {
        "group": group_name,
        "alpha": 1.0,
        "score": float(base_score),
        "local_err": float(base_local),
        "tip_err": float(base_tip),
        "dir_err": float(base_dir),
        "improved": False,
    }

    for alpha in FINGER_SCALE_VALUES:
        if abs(alpha - 1.0) < 1e-8:
            continue

        cand = scale_finger_chain(T, base_id, group_def["affected"], alpha)
        P = project(cand, R, s, t)
        local_err = mean_err(P, U, group_def["target_ids"])
        tip_err = mean_err(P, U, [target_tip])
        dir_err = pair_direction_error(P, U, base_id, target_tip)
        score = (
            beta_local * local_err
            + beta_tip * tip_err
            + beta_dir * dir_err
            + reg_w * abs(alpha - 1.0)
        )

        if score < best["score"]:
            best_T = cand
            best = {
                "group": group_name,
                "alpha": float(alpha),
                "score": float(score),
                "local_err": float(local_err),
                "tip_err": float(tip_err),
                "dir_err": float(dir_err),
                "improved": True,
            }

    return best_T, best


def draw_overlay(U, P_initial, P_final, title, out_path):
    plt.figure(figsize=(8, 8))

    for a, b in CONNECTIONS:
        plt.plot(
            [P_initial[a, 0], P_initial[b, 0]],
            [P_initial[a, 1], P_initial[b, 1]],
            color="black",
            linewidth=2.2,
            alpha=0.55,
            linestyle="--",
            zorder=1,
        )
        plt.plot([U[a, 0], U[b, 0]], [U[a, 1], U[b, 1]], "r-", linewidth=2.6, alpha=0.75, zorder=2)
        plt.plot([P_final[a, 0], P_final[b, 0]], [P_final[a, 1], P_final[b, 1]], "b-", linewidth=1.8, alpha=0.9, zorder=3)

    plt.scatter(P_initial[:, 0], P_initial[:, 1], facecolors="white", edgecolors="black", linewidths=1.5, s=70, label="Initial template", zorder=4)
    plt.scatter(U[:, 0], U[:, 1], c="red", s=40, label="Salux 2D", zorder=5)
    plt.scatter(P_final[:, 0], P_final[:, 1], c="blue", s=30, label="Fitted template", zorder=6)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def direction_report(P_initial, P_final, U):
    return {
        finger: {
            "initial": mean_bone_dir_err(P_initial, U, conns),
            "final": mean_bone_dir_err(P_final, U, conns),
        }
        for finger, conns in FINGER_CONNECTIONS.items()
    }


def fit_one(raw_path, template_centered):
    sample_id = raw_path.stem

    raw = np.load(raw_path).astype(np.float32)
    U = extract_2d_from_raw(raw)
    np.save(OUT_2D_DIR / f"{sample_id}.npy", U)

    best_view = search_best_view(template_centered, U)
    R = np.array(best_view["R"], dtype=np.float32)
    s = float(best_view["scale"])
    t = np.array(best_view["translation"], dtype=np.float32)
    camera_axis_template = R.T @ np.array([0, 0, 1], dtype=np.float32)

    P_initial = project(template_centered, R, s, t)
    T_fit = template_centered.copy()
    logs = []

    for name in ORDERED_GROUPS:
        T_fit, log = refine_group(T_fit, U, R, s, t, name, JOINT_GROUPS[name], camera_axis_template)
        log["stage"] = "direction"
        logs.append(log)

    for name in FINGER_SCALE_ORDER:
        T_fit, log = refine_finger_scale(T_fit, U, R, s, t, name, FINGER_SCALE_GROUPS[name])
        log["stage"] = "finger_scale"
        logs.append(log)

    for name in PIP_DIP_GROUPS:
        T_fit, log = refine_group(T_fit, U, R, s, t, name, JOINT_GROUPS[name], camera_axis_template)
        log["stage"] = "small_local"
        logs.append(log)

    P_final = project(T_fit, R, s, t)

    initial_errors = {
        "all": mean_err(P_initial, U, ALL_IDS),
        "tips": mean_err(P_initial, U, TIP_IDS),
        "palm": mean_err(P_initial, U, PALM_IDS),
    }
    final_errors = {
        "all": mean_err(P_final, U, ALL_IDS),
        "tips": mean_err(P_final, U, TIP_IDS),
        "palm": mean_err(P_final, U, PALM_IDS),
    }

    np.save(OUT_FIT_DIR / f"{sample_id}.npy", T_fit)

    result = {
        "sample_id": sample_id,
        "raw_path": str(raw_path),
        "template_path": str(TEMPLATE_PATH),
        "best_view": best_view,
        "initial_errors": initial_errors,
        "final_errors": final_errors,
        "direction_errors": direction_report(P_initial, P_final, U),
        "improved_steps": sum(1 for x in logs if x["improved"]),
        "logs": logs,
        "output_npy": str(OUT_FIT_DIR / f"{sample_id}.npy"),
        "pipeline": [
            "global_view_search",
            "camera_axis_direction_refine",
            "finger_chain_scale_refine",
            "small_pip_dip_refine",
        ],
    }

    with open(OUT_LOG_DIR / f"{sample_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    if SAVE_OVERLAY:
        title = (
            f"{sample_id} | "
            f"all {initial_errors['all']:.4f}->{final_errors['all']:.4f}, "
            f"tips {initial_errors['tips']:.4f}->{final_errors['tips']:.4f}"
        )
        draw_overlay(U, P_initial, P_final, title, OUT_OVERLAY_DIR / f"{sample_id}.png")

    return result


def main():
    template = np.load(TEMPLATE_PATH).astype(np.float32)
    if template.shape != (21, 3):
        raise ValueError(template.shape)

    template_centered = template - template[0]

    raw_files = sorted(RAW_POSE_DIR.glob("*.npy"))
    if MAX_FILES is not None:
        raw_files = raw_files[:MAX_FILES]

    print("Pose:", POSE_NAME)
    print("Raw files:", len(raw_files))
    print("Input:", RAW_POSE_DIR)
    print("Template:", TEMPLATE_PATH)
    print("Output:", OUT_ROOT)

    summary = []

    for path in tqdm(raw_files):
        try:
            result = fit_one(path, template_centered)
            summary.append({
                "sample_id": result["sample_id"],
                "initial_all": result["initial_errors"]["all"],
                "final_all": result["final_errors"]["all"],
                "initial_tips": result["initial_errors"]["tips"],
                "final_tips": result["final_errors"]["tips"],
                "initial_palm": result["initial_errors"]["palm"],
                "final_palm": result["final_errors"]["palm"],
                "improved_steps": result["improved_steps"],
                "status": "ok",
            })
        except Exception as e:
            summary.append({
                "sample_id": path.stem,
                "status": "failed",
                "error": str(e),
            })

    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ok_rows = [x for x in summary if x["status"] == "ok"]

    print("\nDONE")
    print("Success:", len(ok_rows), "/", len(summary))

    if ok_rows:
        final_all = np.array([x["final_all"] for x in ok_rows], dtype=np.float32)
        final_tips = np.array([x["final_tips"] for x in ok_rows], dtype=np.float32)
        print("final_all mean:", float(final_all.mean()))
        print("final_all max :", float(final_all.max()))
        print("final_tips mean:", float(final_tips.mean()))
        print("final_tips max :", float(final_tips.max()))


if __name__ == "__main__":
    main()
