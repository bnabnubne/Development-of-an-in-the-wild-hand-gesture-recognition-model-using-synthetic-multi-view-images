from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(".")
MODEL_ROOT = PROJECT_ROOT / "model"
APRIL_ROOT = Path(".")

SKELETON_CSV = APRIL_ROOT / "metadata" / "salux_baseline.csv"
SKELETON_MV_CSV = APRIL_ROOT / "metadata" / "salux_multiview3d_8cam_front.csv"
OUT_CSV = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview.csv"

# Use the poses that already have fitted/rendered data in Jun.
POSES = ["ok", "paper", "rock", "scissors", "thefinger"]
CAMERA_IDS = list(range(8))

# Pilot mode: cap each pose so already-rendered classes like ok/paper do not dominate.
# Set to None for full dataset metadata after rendering everything.
MAX_SAMPLES_PER_POSE = 200
MIN_SAMPLES_PER_POSE = 50
REQUIRE_ALL_SPLITS_PER_POSE = True
BALANCE_RANDOM_STATE = 42

# If True, only keep samples with every camera image present.
# If False, keep rows and mark missing camera paths as empty strings.
REQUIRE_ALL_CAMS = True
REQUIRE_ALL_SKELETON_CAMS = True

# Default render layout:
#   data/renders_ok/ok_123/ok_123_cam_00.png
RENDER_DIR_TEMPLATE = "renders_{pose}"

APRIL_TO_JUN_POSE = {
    "the-finger": "thefinger",
    "the_finger": "thefinger",
    "middlefinger": "thefinger",
}

JUN_TO_CANONICAL_LABEL = {
    "thefinger": "the-finger",
}


def normalize_pose_for_render(action):
    action = str(action).strip().lower()
    return APRIL_TO_JUN_POSE.get(action, action)


def canonical_label(action):
    pose = normalize_pose_for_render(action)
    return JUN_TO_CANONICAL_LABEL.get(pose, pose)


def render_paths_for_sample(pose, sample_id):
    render_root = PROJECT_ROOT / RENDER_DIR_TEMPLATE.format(pose=pose)
    sample_dir = render_root / sample_id
    return [
        sample_dir / f"{sample_id}_cam_{cam_id:02d}.png"
        for cam_id in CAMERA_IDS
    ]


def cap_samples_per_pose(df):
    if len(df) == 0:
        return df

    capped = []
    for _, pose_df in df.groupby("render_pose", sort=True):
        if MIN_SAMPLES_PER_POSE is not None and len(pose_df) < MIN_SAMPLES_PER_POSE:
            continue

        if REQUIRE_ALL_SPLITS_PER_POSE:
            missing_splits = {"train", "val", "test"} - set(pose_df["split"])
            if missing_splits:
                continue

        if MAX_SAMPLES_PER_POSE is None:
            capped.append(pose_df)
            continue

        if len(pose_df) <= MAX_SAMPLES_PER_POSE:
            capped.append(pose_df)
            continue

        # Preserve split proportions as much as possible, then fill any remainder.
        sampled_parts = []
        remaining_budget = MAX_SAMPLES_PER_POSE
        split_counts = pose_df["split"].value_counts(normalize=True)

        for split_name, frac in split_counts.items():
            n = int(round(frac * MAX_SAMPLES_PER_POSE))
            split_df = pose_df[pose_df["split"] == split_name]
            n = min(n, len(split_df), remaining_budget)
            if n <= 0:
                continue
            sampled = split_df.sample(n=n, random_state=BALANCE_RANDOM_STATE)
            sampled_parts.append(sampled)
            remaining_budget -= n

        sampled_df = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else pose_df.iloc[:0]

        if remaining_budget > 0:
            used_index = set(sampled_df.index)
            rest_df = pose_df[~pose_df.index.isin(used_index)]
            if len(rest_df):
                sampled_df = pd.concat(
                    [
                        sampled_df,
                        rest_df.sample(
                            n=min(remaining_budget, len(rest_df)),
                            random_state=BALANCE_RANDOM_STATE,
                        ),
                    ],
                    ignore_index=False,
                )

        capped.append(sampled_df)

    if not capped:
        return df.iloc[:0].copy()

    return pd.concat(capped, ignore_index=True)


def build_metadata():
    if not SKELETON_CSV.exists():
        raise FileNotFoundError(f"Missing skeleton CSV: {SKELETON_CSV}")
    if not SKELETON_MV_CSV.exists():
        raise FileNotFoundError(f"Missing multiview skeleton CSV: {SKELETON_MV_CSV}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    skeleton_df = pd.read_csv(SKELETON_CSV)
    required_cols = {"action", "sample_id", "input_path", "split"}
    missing_cols = required_cols - set(skeleton_df.columns)
    if missing_cols:
        raise ValueError(f"Skeleton CSV missing columns: {sorted(missing_cols)}")

    mv_df = pd.read_csv(SKELETON_MV_CSV)
    mv_required_cols = {"action", "sample_id", "split", "orig_input_path"}
    mv_required_cols.update({f"cam_{cam_id}_path" for cam_id in CAMERA_IDS})
    mv_missing_cols = mv_required_cols - set(mv_df.columns)
    if mv_missing_cols:
        raise ValueError(f"Multiview skeleton CSV missing columns: {sorted(mv_missing_cols)}")

    mv_df = mv_df.copy()
    mv_df["render_pose"] = mv_df["action"].apply(normalize_pose_for_render)
    mv_key_to_row = {
        (row["render_pose"], str(row["sample_id"]), row["split"]): row
        for _, row in mv_df.iterrows()
    }

    pose_set = set(POSES)
    rows = []
    skipped_pose = 0
    skipped_missing_rgb = 0
    skipped_missing_mv_skeleton = 0

    for _, row in skeleton_df.iterrows():
        sample_id = str(row["sample_id"])
        render_pose = normalize_pose_for_render(row["action"])

        if render_pose not in pose_set:
            skipped_pose += 1
            continue

        mv_key = (render_pose, sample_id, row["split"])
        mv_row = mv_key_to_row.get(mv_key)
        if mv_row is None:
            skipped_missing_mv_skeleton += 1
            continue

        skeleton_path = Path(mv_row["orig_input_path"])
        if not skeleton_path.exists():
            skipped_missing_mv_skeleton += 1
            continue

        skeleton_cam_paths = [
            Path(mv_row[f"cam_{cam_id}_path"])
            for cam_id in CAMERA_IDS
        ]
        skeleton_cam_exists = [path.exists() for path in skeleton_cam_paths]

        if REQUIRE_ALL_SKELETON_CAMS and not all(skeleton_cam_exists):
            skipped_missing_mv_skeleton += 1
            continue

        cam_paths = render_paths_for_sample(render_pose, sample_id)
        cam_exists = [path.exists() for path in cam_paths]

        if REQUIRE_ALL_CAMS and not all(cam_exists):
            skipped_missing_rgb += 1
            continue

        out_row = {
            "action": canonical_label(row["action"]),
            "render_pose": render_pose,
            "sample_id": sample_id,
            "split": row["split"],
            "skeleton_path": str(skeleton_path),
            "skeleton_orig_path": str(skeleton_path),
            "num_existing_cams": int(sum(cam_exists)),
            "num_existing_skeleton_cams": int(sum(skeleton_cam_exists)),
        }

        for cam_id, rgb_path, rgb_exists, skel_path, skel_exists in zip(
            CAMERA_IDS,
            cam_paths,
            cam_exists,
            skeleton_cam_paths,
            skeleton_cam_exists,
        ):
            out_row[f"rgb_cam_{cam_id}_path"] = str(rgb_path) if rgb_exists else ""
            out_row[f"skeleton_cam_{cam_id}_path"] = str(skel_path) if skel_exists else ""

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    before_cap = len(out_df)
    out_df = cap_samples_per_pose(out_df)
    out_df.to_csv(OUT_CSV, index=False)

    print("DONE")
    print("Skeleton CSV:", SKELETON_CSV)
    print("Output CSV  :", OUT_CSV)
    print("Rows kept   :", len(out_df))
    if MAX_SAMPLES_PER_POSE is not None:
        print("Before cap  :", before_cap)
        print("Pose cap    :", MAX_SAMPLES_PER_POSE)
    if MIN_SAMPLES_PER_POSE is not None:
        print("Min pose samples:", MIN_SAMPLES_PER_POSE)
    print("Skipped pose:", skipped_pose)
    print("Skipped RGB :", skipped_missing_rgb)
    print("Skipped MV skeleton:", skipped_missing_mv_skeleton)

    if len(out_df):
        print("\nSplit counts:")
        print(out_df["split"].value_counts())
        print("\nAction counts:")
        print(out_df["action"].value_counts())
        print("\nCamera completeness:")
        print(out_df["num_existing_cams"].value_counts().sort_index())
        print("\nSkeleton camera completeness:")
        print(out_df["num_existing_skeleton_cams"].value_counts().sort_index())


if __name__ == "__main__":
    build_metadata()
