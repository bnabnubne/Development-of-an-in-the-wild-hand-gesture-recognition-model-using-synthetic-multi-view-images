from pathlib import Path

import pandas as pd


APRIL_ROOT = Path(".")
MODEL_ROOT = Path(".")

DROH_SKELETON_CSV = APRIL_ROOT / "metadata" / "droh_baseline.csv"
DROH_IMAGE_META_CSV = APRIL_ROOT / "test" / "skeleton_final_metadata.csv"
OUT_CSV_5CLS = MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv"
OUT_CSV_7CLS = MODEL_ROOT / "metadata" / "droh_rgb_skeleton_7cls.csv"

CLASSES_5 = ["ok", "paper", "rock", "scissors", "the-finger"]
CLASSES_7 = ["ok", "paper", "rock", "scissors", "the-finger", "thumbdown", "thumbup"]


def build_one(skeleton_df, image_df, classes, out_csv):
    skeleton_df = skeleton_df[skeleton_df["action"].isin(classes)].copy()
    image_df = image_df[image_df["class"].isin(classes)].copy()
    image_df = image_df.rename(columns={"class": "action"})
    image_df["sample_id"] = image_df["raw_path"].apply(lambda p: Path(str(p)).stem)

    merged = skeleton_df.merge(
        image_df[["action", "sample_id", "image_path", "handedness", "raw_path"]],
        on=["action", "sample_id"],
        how="left",
    )

    merged = merged.rename(columns={"input_path": "skeleton_path"})
    merged["split"] = "test"

    merged["image_exists"] = merged["image_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    merged["skeleton_exists"] = merged["skeleton_path"].apply(lambda p: Path(str(p)).exists())

    before_filter = len(merged)
    missing_image = int((~merged["image_exists"]).sum())
    missing_skeleton = int((~merged["skeleton_exists"]).sum())
    merged = merged[merged["image_exists"] & merged["skeleton_exists"]].copy()

    out_cols = [
        "action",
        "sample_id",
        "split",
        "image_path",
        "skeleton_path",
        "handedness",
        "raw_path",
    ]
    out_df = merged[out_cols].copy()
    out_df.to_csv(out_csv, index=False)

    print("DONE")
    print("Output:", out_csv)
    print("Before filter:", before_filter)
    print("Rows:", len(out_df))
    print("\nAction counts:")
    print(out_df["action"].value_counts())
    print("\nMissing skeleton:", missing_skeleton)
    print("Missing RGB:", missing_image)


def main():
    if not DROH_SKELETON_CSV.exists():
        raise FileNotFoundError(DROH_SKELETON_CSV)
    if not DROH_IMAGE_META_CSV.exists():
        raise FileNotFoundError(DROH_IMAGE_META_CSV)

    OUT_CSV_5CLS.parent.mkdir(parents=True, exist_ok=True)

    skeleton_df = pd.read_csv(DROH_SKELETON_CSV)
    image_df = pd.read_csv(DROH_IMAGE_META_CSV)

    build_one(skeleton_df, image_df, CLASSES_5, OUT_CSV_5CLS)
    print("\n" + "=" * 60 + "\n")
    build_one(skeleton_df, image_df, CLASSES_7, OUT_CSV_7CLS)


if __name__ == "__main__":
    main()
