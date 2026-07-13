import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ROOT = Path("./test/droh_real2d_wristnorm")
OUT_DIR = Path("./audit_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

WRIST_TOL = 1e-3
MID_SCALE_TOL = 5e-2

# =========================
# MAIN
# =========================
rows = []
outliers = []

for class_dir in sorted(ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    action = class_dir.name

    for npy_path in sorted(class_dir.glob("*.npy")):
        try:
            kp = np.load(npy_path).astype(np.float32)
        except Exception as e:
            outliers.append({
                "path": str(npy_path),
                "reason": f"load_error:{e}"
            })
            continue

        row = {
            "action": action,
            "path": str(npy_path),
            "shape_ok": int(kp.shape == (21, 2)),
            "has_nan": int(np.isnan(kp).any()),
            "has_inf": int(np.isinf(kp).any()),
        }

        if kp.shape != (21, 2):
            row.update({
                "wrist_norm": np.nan,
                "mid_scale": np.nan,
                "global_min": np.nan,
                "global_max": np.nan,
                "mean": np.nan,
                "std": np.nan,
            })
            rows.append(row)
            outliers.append({
                "path": str(npy_path),
                "reason": f"bad_shape_{kp.shape}"
            })
            continue

        wrist = kp[WRIST_IDX]
        wrist_norm = float(np.linalg.norm(wrist))
        mid_scale = float(np.linalg.norm(kp[MIDDLE_MCP_IDX] - kp[WRIST_IDX]))

        row.update({
            "wrist_norm": wrist_norm,
            "mid_scale": mid_scale,
            "global_min": float(kp.min()),
            "global_max": float(kp.max()),
            "mean": float(kp.mean()),
            "std": float(kp.std()),
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

        rows.append(row)

df = pd.DataFrame(rows)
outlier_df = pd.DataFrame(outliers)

df.to_csv(OUT_DIR / "droh_real2d_audit.csv", index=False)
outlier_df.to_csv(OUT_DIR / "droh_real2d_outliers.csv", index=False)

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
    "mean_mean": float(df["mean"].dropna().mean()) if "mean" in df and df["mean"].dropna().size else None,
    "std_mean": float(df["std"].dropna().mean()) if "std" in df and df["std"].dropna().size else None,
    "num_outliers": int(len(outlier_df)),
    "per_class_counts": df["action"].value_counts().to_dict() if len(df) else {},
}

with open(OUT_DIR / "summary_droh_real2d.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== AUDIT DrOh REAL 2D =====")
print(json.dumps(summary, indent=2))
print("\nDONE")
print("Saved to:", OUT_DIR)