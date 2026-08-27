import pandas as pd
from pathlib import Path

ROOT = Path("./test/multiview_blender") 
OUT_DIR = Path("./metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for class_dir in sorted(ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    action = class_dir.name

    for sample_dir in sorted(class_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        cam_0 = sample_dir / "Cam_0.npy"
        cam_2 = sample_dir / "Cam_2.npy"
        cam_4 = sample_dir / "Cam_4.npy"
        cam_5 = sample_dir / "Cam_5.npy"

        if not (cam_0.exists() and cam_2.exists() and cam_4.exists() and cam_5.exists()):
            continue

        rows.append({
            "action": action,
            "sample_id": sample_dir.name,
            "cam_0_path": str(cam_0),
            "cam_2_path": str(cam_2),
            "cam_4_path": str(cam_4),
            "cam_5_path": str(cam_5),
            "split": "test"
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise ValueError("Không tìm thấy dữ liệu multiview DrOh")

out_csv = OUT_DIR / "droh_multiview.csv"
df.to_csv(out_csv, index=False)

print("DONE")
print("Saved:", out_csv)
print()
print(df["action"].value_counts())