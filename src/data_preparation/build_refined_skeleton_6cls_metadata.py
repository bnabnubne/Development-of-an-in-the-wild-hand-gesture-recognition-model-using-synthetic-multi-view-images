import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
APRIL_ROOT = Path(".")
SOURCE_CSV = APRIL_ROOT / "metadata" / "salux_baseline.csv"
OUT_CSV = PROJECT_ROOT / "model" / "metadata" / "salux_refined_skeleton_6cls.csv"
AUDIT_JSON = PROJECT_ROOT / "model" / "metadata" / "salux_refined_skeleton_6cls_audit.json"

POSE_DIRS = {
    "ok": PROJECT_ROOT / "batch_ok" / "fitted",
    "paper": PROJECT_ROOT / "batch_paper" / "fitted",
    "rock": PROJECT_ROOT / "batch_rock" / "fitted",
    "scissors": PROJECT_ROOT / "batch_scissors" / "fitted",
    "the-finger": PROJECT_ROOT / "batch_thefinger" / "fitted",
    "thumbup": PROJECT_ROOT / "model/data/salux_refined_6cls/thumb_sources/thumbup/fitted",
    "thumbdown": PROJECT_ROOT / "model/data/salux_refined_6cls/thumb_sources/thumbdown/fitted",
}


def main():
    source = pd.read_csv(SOURCE_CSV)
    rows, missing = [], []
    for row in source.itertuples(index=False):
        refined = POSE_DIRS[row.action] / f"{row.sample_id}.npy"
        if not refined.is_file():
            missing.append({"action": row.action, "sample_id": row.sample_id, "path": str(refined)})
            continue
        rows.append({
            "action": "thumb" if row.action in {"thumbup", "thumbdown"} else row.action,
            "source_action": row.action,
            "sample_id": row.sample_id,
            "split": row.split,
            "raw_path": row.input_path,
            "refined_path": str(refined),
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    bad_arrays = []
    for row in out.itertuples(index=False):
        for kind, path in [("raw", row.raw_path), ("refined", row.refined_path)]:
            x = np.load(path, allow_pickle=False)
            if x.shape != (21, 3) or not np.isfinite(x).all():
                bad_arrays.append({"sample_id": row.sample_id, "kind": kind,
                                   "shape": list(x.shape), "finite": bool(np.isfinite(x).all())})

    split_sets = {s: set(out.loc[out.split == s, "sample_id"]) for s in ["train", "val", "test"]}
    overlap = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    audit = {
        "source_rows": len(source), "output_rows": len(out), "missing": missing,
        "bad_arrays": bad_arrays, "duplicate_sample_ids": int(out.duplicated("sample_id").sum()),
        "split_overlap": overlap,
        "class_split_counts": {
            split: {k: int(v) for k, v in part.action.value_counts().sort_index().items()}
            for split, part in out.groupby("split")
        },
        "warning": "refined_path was produced with a ground-truth pose-specific template",
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if missing or bad_arrays or any(overlap.values()) or len(out) != len(source):
        raise RuntimeError("Refined metadata audit failed")


if __name__ == "__main__":
    main()
