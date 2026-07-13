import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
SALUX_CSV = Path("./metadata/salux_multiview.csv")
DROH_CSV = Path("./metadata/droh_real2d_wristnorm.csv")

OUT_DIR = Path("./audit_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALUX_COMPARE_VIEWS = ["cam_0_path", "cam_2_path", "cam_4_path", "cam_5_path"]

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
INDEX_MCP_IDX = 5
PINKY_MCP_IDX = 17

# =========================
# HELPERS
# =========================
def load_stats_from_paths(paths):
    rows = []

    for p in paths:
        kp = np.load(p).astype(np.float32)  # (21,2)

        wrist = kp[WRIST_IDX]
        middle = kp[MIDDLE_MCP_IDX]
        index = kp[INDEX_MCP_IDX]
        pinky = kp[PINKY_MCP_IDX]

        hand_height = float(np.linalg.norm(middle - wrist))
        palm_width = float(np.linalg.norm(pinky - index))
        ratio = palm_width / (hand_height + 1e-6)

        rows.append({
            "global_min": float(kp.min()),
            "global_max": float(kp.max()),
            "mean": float(kp.mean()),
            "std": float(kp.std()),
            "wrist_norm": float(np.linalg.norm(wrist)),
            "mid_scale": hand_height,
            "palm_width": palm_width,
            "palm_width_to_height": ratio,
        })

    return pd.DataFrame(rows)

# =========================
# LOAD DATA
# =========================
salux_df = pd.read_csv(SALUX_CSV)
droh_df = pd.read_csv(DROH_CSV)

salux_test_df = salux_df[salux_df["split"] == "test"].copy()

# =========================
# SALUX per-view summary
# =========================
results = {"salux": {}, "droh": {}}

for view_col in SALUX_COMPARE_VIEWS:
    paths = salux_test_df[view_col].tolist()
    sdf = load_stats_from_paths(paths)

    summary = {
        "num_samples": int(len(sdf)),
        "global_min_mean": float(sdf["global_min"].mean()),
        "global_max_mean": float(sdf["global_max"].mean()),
        "mean_mean": float(sdf["mean"].mean()),
        "std_mean": float(sdf["std"].mean()),
        "wrist_norm_mean": float(sdf["wrist_norm"].mean()),
        "mid_scale_mean": float(sdf["mid_scale"].mean()),
        "palm_width_mean": float(sdf["palm_width"].mean()),
        "palm_width_to_height_mean": float(sdf["palm_width_to_height"].mean()),
    }

    results["salux"][view_col] = summary

# =========================
# DROH real summary
# =========================
droh_paths = droh_df["input_path"].tolist()
ddf = load_stats_from_paths(droh_paths)

results["droh"] = {
    "num_samples": int(len(ddf)),
    "global_min_mean": float(ddf["global_min"].mean()),
    "global_max_mean": float(ddf["global_max"].mean()),
    "mean_mean": float(ddf["mean"].mean()),
    "std_mean": float(ddf["std"].mean()),
    "wrist_norm_mean": float(ddf["wrist_norm"].mean()),
    "mid_scale_mean": float(ddf["mid_scale"].mean()),
    "palm_width_mean": float(ddf["palm_width"].mean()),
    "palm_width_to_height_mean": float(ddf["palm_width_to_height"].mean()),
}

# =========================
# SAVE
# =========================
with open(OUT_DIR / "summary_compare_2d_distributions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Flatten CSV for easier reading
flat_rows = []

for view_col, summary in results["salux"].items():
    row = {"dataset": "salux", "view": view_col}
    row.update(summary)
    flat_rows.append(row)

row = {"dataset": "droh_real2d", "view": "real_single_view"}
row.update(results["droh"])
flat_rows.append(row)

pd.DataFrame(flat_rows).to_csv(OUT_DIR / "compare_2d_distributions.csv", index=False)

print("\n===== COMPARE 2D DISTRIBUTIONS =====")
print(json.dumps(results, indent=2))
print("\nDONE")
print("Saved to:", OUT_DIR)