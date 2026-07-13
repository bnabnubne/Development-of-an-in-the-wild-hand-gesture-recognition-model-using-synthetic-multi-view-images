import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(".")
DEFAULT_RAW_ROOT = Path("./dataset/skeleton_raw")
fitter = None   

SUPPORTED_POSES = [
    "ok",
    "paper",
    "rock",
    "scissors",
    "thefinger",
    "thumbdown",
    "thumbup",
]

POSE_ALIASES = {
    "the-finger": "thefinger",
    "the_finger": "thefinger",
    "middlefinger": "thefinger",
    "scrissors": "scissors",
}

RAW_DIR_NAMES = {
    "thefinger": "the-finger",
}

TEMPLATE_NAME_CANDIDATES = {
    "ok": ["ok_template_joints.npy"],
    "paper": ["paper_template_joints.npy"],
    "rock": ["rock_template_joints.npy"],
    "scissors": ["scissors_template_joints.npy", "scrissors_template_joints.npy"],
    "thefinger": ["thefinger_template_joints.npy", "the-finger_template_joints.npy"],
    "thumbdown": ["thumbdown_template_joints.npy"],
    "thumbup": ["thumbup_template_joints.npy"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch fit one pose using the current template fitting pipeline."
    )
    parser.add_argument(
        "--pose",
        default=None,
        help="One of: ok, paper, rock, scissors, thefinger, thumbdown, thumbup.",
    )
    parser.add_argument("--list-poses", action="store_true", help="Print supported poses and exit.")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Raw skeleton input dir. Default: data/skeleton_raw/{pose}.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Template joints .npy. Default: templates/{pose}_template_joints.npy with known fallbacks.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root. Default: ./batch_{pose}.",
    )
    parser.add_argument("--limit", "--max-files", dest="limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Fit only this sample stem. Can be passed multiple times.",
    )
    parser.add_argument("--save-overlay", action="store_true", help="Save overlay PNGs.")
    return parser.parse_args()


def normalize_pose(pose):
    key = pose.strip().lower()
    key = POSE_ALIASES.get(key, key)

    if key not in SUPPORTED_POSES:
        supported = ", ".join(SUPPORTED_POSES)
        raise ValueError(f"Unsupported pose '{pose}'. Supported poses: {supported}")

    return key


def resolve_template_path(pose, explicit_template):
    if explicit_template:
        path = Path(explicit_template)
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path

    names = TEMPLATE_NAME_CANDIDATES[pose]
    for name in names:
        path = PROJECT_ROOT / "templates" / name
        if path.exists():
            return path

    tried = ", ".join(str(PROJECT_ROOT / "templates" / name) for name in names)
    raise FileNotFoundError(f"No template found for pose '{pose}'. Tried: {tried}")


def configure_fitter(args):
    global fitter

    if args.pose is None:
        supported = ", ".join(SUPPORTED_POSES)
        raise ValueError(f"Missing --pose. Supported poses: {supported}")

    if fitter is None:
        import batch_fit_ok as fitter_module
        fitter = fitter_module

    pose = normalize_pose(args.pose)
    raw_dir_name = RAW_DIR_NAMES.get(pose, pose)
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_ROOT / raw_dir_name
    template_path = resolve_template_path(pose, args.template)
    out_root = Path(args.out_root) if args.out_root else PROJECT_ROOT / f"batch_{pose}"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw pose dir not found: {raw_dir}")

    fitter.POSE_NAME = pose
    fitter.RAW_POSE_DIR = raw_dir
    fitter.TEMPLATE_PATH = template_path
    fitter.OUT_ROOT = out_root
    fitter.OUT_2D_DIR = out_root / "2d"
    fitter.OUT_FIT_DIR = out_root / "fitted"
    fitter.OUT_LOG_DIR = out_root / "logs"
    fitter.OUT_OVERLAY_DIR = out_root / "overlays"
    fitter.MAX_FILES = args.limit
    fitter.SAVE_OVERLAY = args.save_overlay

    for d in [fitter.OUT_2D_DIR, fitter.OUT_FIT_DIR, fitter.OUT_LOG_DIR, fitter.OUT_OVERLAY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    return pose, raw_dir, template_path, out_root


def collect_raw_files(raw_dir, samples, limit):
    if samples:
        raw_files = []
        for sample in samples:
            path = raw_dir / f"{sample}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Sample not found: {path}")
            raw_files.append(path)
    else:
        raw_files = sorted(raw_dir.glob("*.npy"))

    if limit is not None:
        raw_files = raw_files[:limit]

    return raw_files


def run_fit(raw_files, template_centered, summary_name="summary.json"):
    summary = []

    for path in tqdm(raw_files):
        try:
            result = fitter.fit_one(path, template_centered)
            summary.append({
                "sample_id": result["sample_id"],
                "initial_all": result["initial_errors"]["all"],
                "final_all": result["final_errors"]["all"],
                "initial_tips": result["initial_errors"]["tips"],
                "final_tips": result["final_errors"]["tips"],
                "initial_palm": result["initial_errors"]["palm"],
                "final_palm": result["final_errors"]["palm"],
                "improved_steps": result["improved_steps"],
                "status": "ok",
            })
        except Exception as e:
            summary.append({
                "sample_id": path.stem,
                "status": "failed",
                "error": str(e),
            })

    with open(fitter.OUT_ROOT / summary_name, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def print_summary(summary):
    ok_rows = [x for x in summary if x["status"] == "ok"]

    print("\nDONE")
    print("Success:", len(ok_rows), "/", len(summary))

    if not ok_rows:
        return

    final_all = np.array([x["final_all"] for x in ok_rows], dtype=np.float32)
    final_tips = np.array([x["final_tips"] for x in ok_rows], dtype=np.float32)

    print("final_all mean:", float(final_all.mean()))
    print("final_all max :", float(final_all.max()))
    print("final_tips mean:", float(final_tips.mean()))
    print("final_tips max :", float(final_tips.max()))


def main():
    args = parse_args()
    if args.list_poses:
        print("\n".join(SUPPORTED_POSES))
        return

    pose, raw_dir, template_path, out_root = configure_fitter(args)

    template = np.load(template_path).astype(np.float32)
    if template.shape != (21, 3):
        raise ValueError(f"Expected template shape (21,3), got {template.shape}: {template_path}")

    template_centered = template - template[0]
    raw_files = collect_raw_files(raw_dir, args.sample, args.limit)
    summary_name = "summary.json"
    if args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("--shard-index and --num-shards must be provided together")
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Expected 0 <= shard-index < num-shards")
        raw_files = raw_files[args.shard_index::args.num_shards]
        summary_name = f"summary_shard_{args.shard_index}_of_{args.num_shards}.json"

    print("Pose:", pose)
    print("Raw files:", len(raw_files))
    print("Input:", raw_dir)
    print("Template:", template_path)
    print("Output:", out_root)

    summary = run_fit(raw_files, template_centered, summary_name)
    print_summary(summary)


if __name__ == "__main__":
    main()
