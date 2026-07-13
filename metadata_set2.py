import pandas as pd
from pathlib import Path
 
ROOT = Path("./test/skeleton_final_v2")
OUT_DIR = Path("./metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for class_dir in sorted(ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    action = class_dir.name

    for npy_path in sorted(class_dir.glob("*.npy")):
        rows.append({
            "action": action,
            "sample_id": npy_path.stem,
            "input_path": str(npy_path),
            "split": "test"
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise ValueError("Không tìm thấy dữ liệu trong skeleton_final_filtered")

out_csv = OUT_DIR / "droh_baseline.csv"
df.to_csv(out_csv, index=False)

print("DONE")
print("Saved:", out_csv)
print()
print(df["action"].value_counts())