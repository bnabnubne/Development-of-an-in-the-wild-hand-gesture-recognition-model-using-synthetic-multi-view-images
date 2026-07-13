from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = Path("./metadata/rgb_skeleton_multiview.csv")
EXPECTED_NUM_CAMS = 8


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    cam_cols = sorted(
        [c for c in df.columns if c.startswith("rgb_cam_") and c.endswith("_path")],
        key=lambda x: int(x.split("_")[2]),
    )
    skeleton_cam_cols = sorted(
        [c for c in df.columns if c.startswith("skeleton_cam_") and c.endswith("_path")],
        key=lambda x: int(x.split("_")[2]),
    )

    print("CSV:", CSV_PATH)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Camera columns:", cam_cols)
    print("Skeleton camera columns:", skeleton_cam_cols)

    if len(cam_cols) != EXPECTED_NUM_CAMS:
        raise ValueError(f"Expected {EXPECTED_NUM_CAMS} camera columns, got {len(cam_cols)}")
    if len(skeleton_cam_cols) != EXPECTED_NUM_CAMS:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CAMS} skeleton camera columns, got {len(skeleton_cam_cols)}"
        )

    print("\nSplit counts:")
    print(df["split"].value_counts())

    print("\nAction counts:")
    print(df["action"].value_counts())

    print("\nAction x split counts:")
    action_split = df.groupby(["action", "split"]).size().unstack(fill_value=0)
    print(action_split)

    print("\nMissing skeleton paths:")
    missing_skeleton = df["skeleton_path"].apply(lambda p: not Path(str(p)).exists()).sum()
    print(missing_skeleton)

    print("\nMissing multiview skeleton paths:")
    for col in skeleton_cam_cols:
        missing = df[col].apply(lambda p: not Path(str(p)).exists()).sum()
        print(col, "missing:", missing)

    print("\nMissing RGB paths:")
    for col in cam_cols:
        missing = df[col].apply(lambda p: not Path(str(p)).exists()).sum()
        print(col, "missing:", missing)

    print("\nSplit overlap check by action/sample_id:")
    keys = df["action"].astype(str) + "::" + df["sample_id"].astype(str)
    train_ids = set(keys[df["split"] == "train"])
    val_ids = set(keys[df["split"] == "val"])
    test_ids = set(keys[df["split"] == "test"])
    print("train/val :", len(train_ids & val_ids))
    print("train/test:", len(train_ids & test_ids))
    print("val/test  :", len(val_ids & test_ids))

    if missing_skeleton:
        raise RuntimeError("Some skeleton files are missing.")

    for col in skeleton_cam_cols:
        if df[col].apply(lambda p: not Path(str(p)).exists()).any():
            raise RuntimeError(f"Some multiview skeleton files are missing in {col}.")

    for col in cam_cols:
        if df[col].apply(lambda p: not Path(str(p)).exists()).any():
            raise RuntimeError(f"Some RGB files are missing in {col}.")

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Sample ID leakage between splits.")

    required_splits = {"train", "val", "test"}
    for action, sub_df in df.groupby("action"):
        missing_splits = required_splits - set(sub_df["split"])
        if missing_splits:
            raise RuntimeError(f"Action '{action}' is missing splits: {sorted(missing_splits)}")

    print("\nCHECK PASSED")


if __name__ == "__main__":
    main()
