import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, math

TEMPLATE_PATH = "./templates/ok_template_joints.npy"
SALUX_2D_PATH = "./batch_ok/2d/ok (111).npy"

OUT_DIR = Path("./debug_fixed_view")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "ok (111)_fitted_template_joints.npy"
OUT_JSON = OUT_DIR / "ok (111)_fit_log.json"
OUT_PNG = OUT_DIR / "ok (111)_overlay_fixed_view.png"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

PALM_IDS = [0,5,9,13,17]
ALIGN_IDS = [0, 5, 9, 13, 17]
ALIGN_WEIGHTS = np.array([5.0, 1.2, 1.5, 1.5, 1.2], dtype=np.float32)

TIP_IDS = [4,8,12,16,20]
ALL_IDS = list(range(21))

JOINT_GROUPS = {
    "thumb_mcp":  {"pivot":1,  "affected":[2,3,4],      "target_ids":[2,3,4]},
    "thumb_ip":   {"pivot":2,  "affected":[3,4],        "target_ids":[3,4]},
    "thumb_tip":  {"pivot":3,  "affected":[4],          "target_ids":[4]},

    "index_mcp":  {"pivot":5,  "affected":[6,7,8],      "target_ids":[6,7,8]},
    "index_pip":  {"pivot":6,  "affected":[7,8],        "target_ids":[7,8]},
    "index_dip":  {"pivot":7,  "affected":[8],          "target_ids":[8]},

    "middle_mcp": {"pivot":9,  "affected":[10,11,12],   "target_ids":[10,11,12]},
    "middle_pip": {"pivot":10, "affected":[11,12],      "target_ids":[11,12]},
    "middle_dip": {"pivot":11, "affected":[12],         "target_ids":[12]},

    "ring_mcp":   {"pivot":13, "affected":[14,15,16],   "target_ids":[14,15,16]},
    "ring_pip":   {"pivot":14, "affected":[15,16],      "target_ids":[15,16]},
    "ring_dip":   {"pivot":15, "affected":[16],         "target_ids":[16]},

    "pinky_mcp":  {"pivot":17, "affected":[18,19,20],   "target_ids":[18,19,20]},
    "pinky_pip":  {"pivot":18, "affected":[19,20],      "target_ids":[19,20]},
    "pinky_dip":  {"pivot":19, "affected":[20],         "target_ids":[20]},
}

ORDERED_GROUPS = [
    "thumb_mcp","thumb_ip","thumb_tip",
    "index_mcp","index_pip","index_dip",

    "middle_mcp","middle_pip","middle_dip",
    "ring_mcp","ring_pip","ring_dip",
    "pinky_mcp","pinky_pip","pinky_dip",
]

FINGER_SCALE_GROUPS = {
    "thumb_scale":  {"base":1,  "affected":[2,3,4],      "target_ids":[2,3,4]},
    "index_scale":  {"base":5,  "affected":[6,7,8],      "target_ids":[6,7,8]},
    "middle_scale": {"base":9,  "affected":[10,11,12],   "target_ids":[10,11,12]},
    "ring_scale":   {"base":13, "affected":[14,15,16],   "target_ids":[14,15,16]},
    "pinky_scale":  {"base":17, "affected":[18,19,20],   "target_ids":[18,19,20]},
}

FINGER_SCALE_ORDER = [
    "thumb_scale",
    "index_scale",
    "middle_scale",
    "ring_scale",
    "pinky_scale",
]

PIP_DIP_GROUPS = [
    "thumb_ip","thumb_tip",
    "index_pip","index_dip",
    "middle_pip","middle_dip",
    "ring_pip","ring_dip",
    "pinky_pip","pinky_dip",
]

RX_VALUES = range(-90, 91, 10)
RY_VALUES = range(-90, 91, 10)
RZ_VALUES = range(-180, 181, 10)

LOCAL_VIEW_RADIUS = 10
LOCAL_VIEW_STEP = 2
FINE_VIEW_RADIUS = 2
FINE_VIEW_STEP = 1

ANGLE_RANGE_MAIN = range(-25, 26, 5)
ANGLE_RANGE_EXTENDED = range(-10, 11, 5)
FINGER_SCALE_VALUES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

PASSES = 2


def rot_x(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def rot_y(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def rot_z(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def make_R(rx, ry, rz):
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)

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

    if len(errs) == 0:
        return 1.0

    return float(np.mean(errs))


def finger_dir_err(P, U, finger_connections):
    return mean_bone_dir_err(P, U, finger_connections)


THUMB_INDEX_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
]

EXTENDED_FINGER_CONNECTIONS = [
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

FINGER_CONNECTIONS = {
    "thumb":  [(0,1),(1,2),(2,3),(3,4)],
    "index":  [(0,5),(5,6),(6,7),(7,8)],
    "middle": [(0,9),(9,10),(10,11),(11,12)],
    "ring":   [(0,13),(13,14),(14,15),(15,16)],
    "pinky":  [(0,17),(17,18),(18,19),(19,20)],
}

def group_finger_name(group_name):
    return group_name.split("_")[0]

def local_direction_error(P, U, group_name):
    finger = group_finger_name(group_name)
    return mean_bone_dir_err(P, U, FINGER_CONNECTIONS[finger])

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
    best = search_view_grid(
        T,
        U,
        RX_VALUES,
        RY_VALUES,
        RZ_VALUES,
        stage="coarse",
    )

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

def rotate_about_pivot(T, pivot_id, affected_ids, axis, angle):
    if axis == "x":
        Rloc = rot_x(angle)
    elif axis == "y":
        Rloc = rot_y(angle)
    elif axis == "z":
        Rloc = rot_z(angle)
    else:
        raise ValueError(axis)

    out = T.copy()
    pivot = T[pivot_id].copy()

    for j in affected_ids:
        out[j] = pivot + (Rloc @ (T[j] - pivot))

    return out
def rot_axis_angle(axis, angle_deg):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return np.array([
        [c+x*x*C, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, c+z*z*C],
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
    best_candidate = None

    for angle in angle_range:
        if angle == 0:
            continue

        Rloc = rot_axis_angle(camera_axis_template, angle)
        cand = rotate_about_pivot_with_matrix(
            T,
            pivot_id,
            group_def["affected"],
            Rloc
        )

        P = project(cand, R, s, t)

        local_err = mean_err(P, U, group_def["target_ids"])
        dir_err = pair_direction_error(P, U, pivot_id, target_tip)

        score = beta_local * local_err + beta_dir * dir_err + reg_w * abs(angle)

        candidate_log = {
            "group": group_name,
            "axis": "camera",
            "angle": int(angle),
            "score": float(score),
            "local_err": float(local_err),
            "dir_err": float(dir_err),
            "desired_angle_2d": float(desired_angle),
            "improved": score < base_score,
        }

        if best_candidate is None or score < best_candidate["score"]:
            best_candidate = candidate_log

        if score < best["score"]:
            best_T = cand
            best = candidate_log.copy()
            best["improved"] = True

    print(
        group_name,
        "base_local=", base_local,
        "best_local=", best["local_err"],
        "axis=", best["axis"],
        "angle=", best["angle"],
        "improved=", best["improved"]
    )
    if best_candidate is not None and not best["improved"]:
        print(
            "  best_candidate",
            "local=", best_candidate["local_err"],
            "axis=", best_candidate["axis"],
            "angle=", best_candidate["angle"],
            "score=", best_candidate["score"]
        )

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

        cand = scale_finger_chain(
            T,
            base_id,
            group_def["affected"],
            alpha
        )
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

    print(
        group_name,
        "base_local=", base_local,
        "best_local=", best["local_err"],
        "alpha=", best["alpha"],
        "tip_err=", best["tip_err"],
        "improved=", best["improved"]
    )

    return best_T, best

def draw_overlay(U, P_initial, P_final, out_path, title):
    plt.figure(figsize=(8, 8))

    for a, b in CONNECTIONS:
        plt.plot([P_initial[a,0], P_initial[b,0]], [P_initial[a,1], P_initial[b,1]],
                 color="black", linewidth=2.2, alpha=0.55, linestyle="--", zorder=1)
        plt.plot([U[a,0], U[b,0]], [U[a,1], U[b,1]], "r-", linewidth=2.6, alpha=0.75, zorder=2)
        plt.plot([P_final[a,0], P_final[b,0]], [P_final[a,1], P_final[b,1]],
                 "b-", linewidth=1.8, alpha=0.9, zorder=3)

    plt.scatter(
        P_initial[:,0], P_initial[:,1],
        facecolors="white", edgecolors="black", linewidths=1.5,
        s=70, label="Initial template", zorder=4
    )
    plt.scatter(U[:,0], U[:,1], c="red", s=40, label="Salux 2D", zorder=5)
    plt.scatter(P_final[:,0], P_final[:,1], c="blue", s=30, label="Fitted template", zorder=6)

    for i in range(21):
        plt.text(
            P_initial[i,0], P_initial[i,1], str(i),
            color="black", fontsize=8, ha="right", va="bottom", zorder=7
        )
        plt.text(U[i,0], U[i,1], str(i), color="red", fontsize=8, zorder=8)
        plt.text(P_final[i,0], P_final[i,1], str(i), color="blue", fontsize=8, zorder=9)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    
def print_finger_direction_report(U, P_initial, P_final):
    print("\n=== FINGER DIRECTION ERROR: INITIAL -> FITTED ===")
    for finger, conns in FINGER_CONNECTIONS.items():
        e0 = mean_bone_dir_err(P_initial, U, conns)
        e1 = mean_bone_dir_err(P_final, U, conns)
        print(f"{finger:6s}: {e0:.4f} -> {e1:.4f}")
        
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
    s = float(best_view["scale"])
    t = np.array(best_view["translation"], dtype=np.float32)
    camera_axis_template = R.T @ np.array([0, 0, 1], dtype=np.float32)

    P_initial = project(T0, R, s, t)
    print("\n=== WRIST ALIGNMENT CHECK ===")
    print("U wrist        :", U[0])
    print("Initial wrist  :", P_initial[0])
    print("Wrist error    :", np.linalg.norm(P_initial[0] - U[0]))
    
    T_fit = T0.copy()
    logs = []

    print("\n=== CAMERA-AXIS DIRECTION REFINE ===")
    for name in ORDERED_GROUPS:
        T_fit, log = refine_group(
            T_fit,
            U,
            R,
            s,
            t,
            name,
            JOINT_GROUPS[name],
            camera_axis_template
        )
        log["stage"] = "direction"
        logs.append(log)

    print("\n=== FINGER CHAIN SCALE REFINE ===")
    for name in FINGER_SCALE_ORDER:
        T_fit, log = refine_finger_scale(
            T_fit,
            U,
            R,
            s,
            t,
            name,
            FINGER_SCALE_GROUPS[name]
        )
        log["stage"] = "finger_scale"
        logs.append(log)

    print("\n=== SMALL PIP/DIP REFINE ===")
    for name in PIP_DIP_GROUPS:
        T_fit, log = refine_group(
            T_fit,
            U,
            R,
            s,
            t,
            name,
            JOINT_GROUPS[name],
            camera_axis_template
        )
        log["stage"] = "small_local"
        logs.append(log)

    P_final = project(T_fit, R, s, t)
    print_finger_direction_report(U, P_initial, P_final)

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

    result = {
        "template_path": TEMPLATE_PATH,
        "salux_2d_path": SALUX_2D_PATH,
        "best_view": best_view,
        "initial_errors": initial_errors,
        "final_errors": final_errors,
        "logs": logs,
        "output_npy": str(OUT_NPY),
        "note": "Fixed-view fitting. Initial and fitted projections use the exact same R, scale, and translation."
    }

    np.save(OUT_NPY, T_fit)

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    improved_count = sum(1 for x in logs if x["improved"])

    title = (
        f"Fixed view | all {initial_errors['all']:.4f}->{final_errors['all']:.4f}, "
        f"tips {initial_errors['tips']:.4f}->{final_errors['tips']:.4f}, "
        f"updates={improved_count}"
    )

    draw_overlay(U, P_initial, P_final, OUT_PNG, title)

    print("BEST VIEW:")
    print(json.dumps(best_view, indent=2))
    print("INITIAL ERRORS:")
    print(json.dumps(initial_errors, indent=2))
    print("FINAL ERRORS:")
    print(json.dumps(final_errors, indent=2))
    print("IMPROVED STEPS:", improved_count)
    print("Saved:", OUT_NPY)
    print("Saved:", OUT_JSON)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
