import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
ROOT = Path("./dataset/multiview_blender")
OUT_DIR = Path("./metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise ValueError("Không tìm thấy dữ liệu multiview Salux")

train_df, temp_df = train_test_split(
    df,
    test_size=VAL_RATIO + TEST_RATIO,
    stratify=df["action"],
    random_state=RANDOM_STATE
)

val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_df, test_df = train_test_split(
    temp_df,
    test_size=1 - val_relative,
    stratify=temp_df["action"],
    random_state=RANDOM_STATE
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

out_csv = OUT_DIR / "salux_multiview.csv"
full_df.to_csv(out_csv, index=False)

print("DONE")
print("Saved:", out_csv)
print()
print(full_df["split"].value_counts())
print()
print(full_df["action"].value_counts())