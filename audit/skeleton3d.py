import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ROOTS = {
    "salux_3d": Path("./dataset/skeleton_final_v2"),
    "droh_3d": Path("./test/skeleton_final_v2"),
}

OUT_DIR = Path("./audit_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
INDEX_MCP_IDX = 5
PINKY_MCP_IDX = 17

WRIST_TOL = 1e-3
MID_SCALE_TOL = 5e-2

# =========================
# HELPERS
# =========================
def safe_norm(x):
    return float(np.linalg.norm(x))

def palm_sign(kp: np.ndarray):
    wrist = kp[WRIST_IDX]
    index = kp[INDEX_MCP_IDX]
    middle = kp[MIDDLE_MCP_IDX]
    pinky = kp[PINKY_MCP_IDX]

    x_axis = pinky - index
    y_axis = middle - wrist

    nx = np.linalg.norm(x_axis)
    ny = np.linalg.norm(y_axis)

    if nx < 1e-8 or ny < 1e-8:
        return np.nan

    x_axis = x_axis / nx
    y_axis = y_axis / ny
    z_axis = np.cross(x_axis, y_axis)

    nz = np.linalg.norm(z_axis)
    if nz < 1e-8:
        return np.nan

    z_axis = z_axis / nz
    return float(z_axis[2])

# =========================
# MAIN
# =========================
all_summaries = {}

for tag, root in ROOTS.items():
    if not root.exists():
        print(f"[WARN] Missing root: {root}")
        continue

    rows = []
    outliers = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        action = class_dir.name

        for npy_path in sorted(class_dir.glob("*.npy")):
            try:
                kp = np.load(npy_path).astype(np.float32)
            except Exception as e:
                outliers.append({
                    "path": str(npy_path),
                    "reason": f"load_error: {e}"
                })
                continue

            row = {
                "dataset": tag,
                "action": action,
                "path": str(npy_path),
                "shape_ok": int(kp.shape == (21, 3)),
                "has_nan": int(np.isnan(kp).any()),
                "has_inf": int(np.isinf(kp).any()),
            }

            if kp.shape != (21, 3):
                row.update({
                    "wrist_norm": np.nan,
                    "mid_scale": np.nan,
                    "global_min": np.nan,
                    "global_max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "palm_sign_z": np.nan,
                })
                outliers.append({
                    "path": str(npy_path),
                    "reason": f"bad_shape_{kp.shape}"
                })
                rows.append(row)
                continue

            wrist = kp[WRIST_IDX]
            wrist_norm = safe_norm(wrist)
            mid_scale = safe_norm(kp[MIDDLE_MCP_IDX] - kp[WRIST_IDX])
            z_sign = palm_sign(kp)

            row.update({
                "wrist_norm": wrist_norm,
                "mid_scale": mid_scale,
                "global_min": float(kp.min()),
                "global_max": float(kp.max()),
                "mean": float(kp.mean()),
                "std": float(kp.std()),
                "palm_sign_z": z_sign,
                "wrist_ok": int(wrist_norm <= WRIST_TOL),
                "mid_scale_ok": int(abs(mid_scale - 1.0) <= MID_SCALE_TOL),
            })

            if wrist_norm > WRIST_TOL:
                outliers.append({
                    "path": str(npy_path),
                    "reason": f"wrist_not_zero:{wrist_norm:.6f}"
                })

            if abs(mid_scale - 1.0) > MID_SCALE_TOL:
                outliers.append({
                    "path": str(npy_path),
                    "reason": f"mid_scale_not_one:{mid_scale:.6f}"
                })

            if np.isnan(z_sign):
                outliers.append({
                    "path": str(npy_path),
                    "reason": "palm_sign_nan"
                })

            rows.append(row)

    df = pd.DataFrame(rows)
    outlier_df = pd.DataFrame(outliers)

    csv_path = OUT_DIR / f"{tag}_audit_3d.csv"
    outlier_path = OUT_DIR / f"{tag}_audit_3d_outliers.csv"
    df.to_csv(csv_path, index=False)
    outlier_df.to_csv(outlier_path, index=False)

    summary = {
        "num_samples": int(len(df)),
        "num_shape_ok": int(df["shape_ok"].sum()) if len(df) else 0,
        "num_has_nan": int(df["has_nan"].sum()) if len(df) else 0,
        "num_has_inf": int(df["has_inf"].sum()) if len(df) else 0,
        "num_wrist_ok": int(df["wrist_ok"].sum()) if "wrist_ok" in df else 0,
        "num_mid_scale_ok": int(df["mid_scale_ok"].sum()) if "mid_scale_ok" in df else 0,
        "wrist_norm_mean": float(df["wrist_norm"].dropna().mean()) if "wrist_norm" in df and df["wrist_norm"].dropna().size else None,
        "wrist_norm_std": float(df["wrist_norm"].dropna().std()) if "wrist_norm" in df and df["wrist_norm"].dropna().size else None,
        "mid_scale_mean": float(df["mid_scale"].dropna().mean()) if "mid_scale" in df and df["mid_scale"].dropna().size else None,
        "mid_scale_std": float(df["mid_scale"].dropna().std()) if "mid_scale" in df and df["mid_scale"].dropna().size else None,
        "palm_sign_z_mean": float(df["palm_sign_z"].dropna().mean()) if "palm_sign_z" in df and df["palm_sign_z"].dropna().size else None,
        "num_outliers": int(len(outlier_df)),
        "per_class_counts": df["action"].value_counts().to_dict() if len(df) else {},
    }

    all_summaries[tag] = summary

    print(f"\n===== AUDIT 3D: {tag} =====")
    print(json.dumps(summary, indent=2))

summary_path = OUT_DIR / "summary_audit_3d.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, indent=2)

print("\nDONE")
print("Saved to:", OUT_DIR)