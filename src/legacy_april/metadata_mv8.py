import pandas as pd
from pathlib import Path

CSV = "./metadata/salux_multiview3d_8cam.csv"

df = pd.read_csv(CSV)

cam_cols = sorted(
    [c for c in df.columns if c.startswith("cam_") and c.endswith("_path")],
    key=lambda x: int(x.split("_")[1])
)

print("Rows:", len(df))
print("Camera columns:", cam_cols)
print("Number of cameras:", len(cam_cols))
print("\nSplit counts:")
print(df["split"].value_counts())

print("\nClass counts:")
print(df["action"].value_counts())

print("\nChecking file paths...")
for col in cam_cols:
    missing = df[col].apply(lambda p: not Path(p).exists()).sum()
    print(col, "missing:", missing)

train_ids = set(df[df["split"] == "train"]["sample_id"].astype(str))
val_ids = set(df[df["split"] == "val"]["sample_id"].astype(str))
test_ids = set(df[df["split"] == "test"]["sample_id"].astype(str))

print("\nOverlap check:")
print("train/val overlap :", len(train_ids & val_ids))
print("train/test overlap:", len(train_ids & test_ids))
print("val/test overlap  :", len(val_ids & test_ids))

assert len(cam_cols) == 8, "Expected 8 camera columns"
assert all(df[col].apply(lambda p: Path(p).exists()).all() for col in cam_cols), "Some camera files are missing"
assert len(train_ids & val_ids) == 0
assert len(train_ids & test_ids) == 0
assert len(val_ids & test_ids) == 0

print("\nCHECK PASSED")