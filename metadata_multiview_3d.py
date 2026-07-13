import pandas as pd
from pathlib import Path

BASELINE_CSV = Path("./metadata/salux_baseline.csv")
MV3D_ROOT = Path("./dataset/mv3d_salux")
OUT_CSV = Path("./metadata/salux_multiview3d.csv")

base_df = pd.read_csv(BASELINE_CSV)

rows = []
missing = 0

for _, row in base_df.iterrows():
    action = row["action"]
    sample_id = row["sample_id"]
    split = row["split"]

    sample_dir = MV3D_ROOT / action / sample_id
    cam0 = sample_dir / "cam_0.npy"
    cam1 = sample_dir / "cam_1.npy"
    cam2 = sample_dir / "cam_2.npy"
    cam3 = sample_dir / "cam_3.npy"

    if not (cam0.exists() and cam1.exists() and cam2.exists() and cam3.exists()):
        print(f"[MISS] {sample_dir}")
        missing += 1
        continue

    rows.append({
        "action": action,
        "sample_id": sample_id,
        "cam_0_path": str(cam0),
        "cam_1_path": str(cam1),
        "cam_2_path": str(cam2),
        "cam_3_path": str(cam3),
        "split": split,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("DONE")
print("Saved:", OUT_CSV)
print("Rows:", len(df))
print("Missing:", missing)
print(df["split"].value_counts())