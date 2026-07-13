import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ROOT = Path("./dataset/multiview_blender")
OUT_DIR = Path("./audit_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEW_NAMES = ["Cam_0", "Cam_2", "Cam_4", "Cam_5"]

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

WRIST_TOL = 1e-3
MID_SCALE_TOL = 5e-2

# =========================
# MAIN
# =========================
all_view_summaries = {}
all_rows = []
all_outliers = []

for class_dir in sorted(ROOT.iterdir()):
    if not class_dir.is_dir():
        continue
    action = class_dir.name

    for sample_dir in sorted(class_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        for view_name in VIEW_NAMES:
            npy_path = sample_dir / f"{view_name}.npy"

            if not npy_path.exists():
                all_outliers.append({
                    "path": str(npy_path),
                    "reason": "missing_view_file"
                })
                continue

            try:
                kp = np.load(npy_path).astype(np.float32)
            except Exception as e:
                all_outliers.append({
                    "path": str(npy_path),
                    "reason": f"load_error:{e}"
                })
                continue

            row = {
                "action": action,
                "sample_id": sample_id,
                "view": view_name,
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
                all_rows.append(row)
                all_outliers.append({
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
                all_outliers.append({
                    "path": str(npy_path),
                    "reason": f"wrist_not_zero:{wrist_norm:.6f}"
                })

            if abs(mid_scale - 1.0) > MID_SCALE_TOL:
                all_outliers.append({
                    "path": str(npy_path),
                    "reason": f"mid_scale_not_one:{mid_scale:.6f}"
                })

            all_rows.append(row)

df = pd.DataFrame(all_rows)
outlier_df = pd.DataFrame(all_outliers)

df.to_csv(OUT_DIR / "salux_multiview2d_audit.csv", index=False)
outlier_df.to_csv(OUT_DIR / "salux_multiview2d_outliers.csv", index=False)

for view_name in VIEW_NAMES:
    sub = df[df["view"] == view_name].copy()
    if len(sub) == 0:
        continue

    summary = {
        "num_samples": int(len(sub)),
        "num_shape_ok": int(sub["shape_ok"].sum()),
        "num_has_nan": int(sub["has_nan"].sum()),
        "num_has_inf": int(sub["has_inf"].sum()),
        "num_wrist_ok": int(sub["wrist_ok"].sum()),
        "num_mid_scale_ok": int(sub["mid_scale_ok"].sum()),
        "wrist_norm_mean": float(sub["wrist_norm"].dropna().mean()) if sub["wrist_norm"].dropna().size else None,
        "wrist_norm_std": float(sub["wrist_norm"].dropna().std()) if sub["wrist_norm"].dropna().size else None,
        "mid_scale_mean": float(sub["mid_scale"].dropna().mean()) if sub["mid_scale"].dropna().size else None,
        "mid_scale_std": float(sub["mid_scale"].dropna().std()) if sub["mid_scale"].dropna().size else None,
        "mean_mean": float(sub["mean"].dropna().mean()) if sub["mean"].dropna().size else None,
        "std_mean": float(sub["std"].dropna().mean()) if sub["std"].dropna().size else None,
        "per_class_counts": sub["action"].value_counts().to_dict(),
    }
    all_view_summaries[view_name] = summary

    print(f"\n===== AUDIT SYNTHETIC 2D: {view_name} =====")
    print(json.dumps(summary, indent=2))

with open(OUT_DIR / "summary_salux_multiview2d.json", "w", encoding="utf-8") as f:
    json.dump(all_view_summaries, f, indent=2)

print("\nDONE")
print("Saved to:", OUT_DIR)