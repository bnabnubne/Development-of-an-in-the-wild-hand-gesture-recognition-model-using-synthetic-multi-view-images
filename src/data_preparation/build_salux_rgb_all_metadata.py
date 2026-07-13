from pathlib import Path

import pandas as pd


APRIL_ROOT = Path(".")
MODEL_ROOT = Path(".")
SALUX_IMAGE_ROOT = Path("./data/raw/FullLabelled")

SALUX_BASELINE_CSV = APRIL_ROOT / "metadata" / "salux_baseline.csv"
OUT_CSV = MODEL_ROOT / "metadata" / "salux_original_rgb_all_5cls.csv"
CLASSES_5 = ["ok", "paper", "rock", "scissors", "the-finger"]


def find_image(action, sample_id):
    class_dir = SALUX_IMAGE_ROOT / action
    for suffix in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        path = class_dir / f"{sample_id}{suffix}"
        if path.exists():
            return path
    return None


def main():
    df = pd.read_csv(SALUX_BASELINE_CSV)
    df = df[df["action"].isin(CLASSES_5)].copy()

    rows = []
    missing_images = 0
    missing_skeleton = 0
    for _, row in df.iterrows():
        image_path = find_image(row["action"], row["sample_id"])
        skeleton_path = Path(row["input_path"])
        if image_path is None:
            missing_images += 1
            continue
        if not skeleton_path.exists():
            missing_skeleton += 1
            continue
        rows.append({
            "action": row["action"],
            "sample_id": row["sample_id"],
            "split": row["split"],
            "image_path": str(image_path),
            "skeleton_path": str(skeleton_path),
        })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print("Output:", OUT_CSV)
    print("Rows:", len(out_df))
    print("Missing images:", missing_images)
    print("Missing skeleton:", missing_skeleton)
    if len(out_df):
        print("Split counts:")
        print(out_df["split"].value_counts())
        print("Action counts:")
        print(out_df["action"].value_counts())


if __name__ == "__main__":
    main()
