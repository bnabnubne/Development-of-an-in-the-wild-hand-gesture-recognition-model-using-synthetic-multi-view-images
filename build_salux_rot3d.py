import pandas as pd
from pathlib import Path

BASELINE_CSV = Path("./metadata/salux_baseline.csv")
ROT3D_ROOT = Path("./dataset/rot3d_salux")
OUT_CSV = Path("./metadata/salux_rot3d.csv")

base_df = pd.read_csv(BASELINE_CSV)

rows = []
missing = 0

for _, row in base_df.iterrows():
    action = row["action"]
    sample_id = row["sample_id"]
    split = row["split"]

    class_dir = ROT3D_ROOT / action
    pattern = f"{sample_id}_rot*.npy"
    files = sorted(class_dir.glob(pattern))

    if len(files) == 0:
        print(f"[MISS] {action}/{sample_id}")
        missing += 1
        continue

    for f in files:
        rows.append({
            "action": action,
            "sample_id": sample_id,
            "aug_id": f.stem,
            "input_path": str(f),
            "split": split
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("DONE")
print("Saved:", OUT_CSV)
print("Rows:", len(df))
print("Missing base samples:", missing)
print(df["split"].value_counts())