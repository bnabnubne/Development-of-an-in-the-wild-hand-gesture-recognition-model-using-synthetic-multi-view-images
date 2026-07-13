import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
 
ROOT = Path("./dataset/skeleton_final_v2")
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

    for npy_path in sorted(class_dir.glob("*.npy")):
        rows.append({
            "action": action,
            "sample_id": npy_path.stem,
            "input_path": str(npy_path)
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise ValueError("Không tìm thấy dữ liệu trong skeleton_final_v2")

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

out_csv = OUT_DIR / "salux_baseline.csv"
full_df.to_csv(out_csv, index=False)

print("DONE")
print("Saved:", out_csv)
print()
print(full_df["split"].value_counts())
print()
print(full_df["action"].value_counts())