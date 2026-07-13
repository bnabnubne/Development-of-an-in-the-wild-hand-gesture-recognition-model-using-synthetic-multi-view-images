import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
BASELINE_CSV = PROJECT_ROOT.parent / "April" / "metadata" / "salux_baseline.csv"
OUT_CSV = MODEL_ROOT / "metadata" / "salux_refined_skeleton_5cls.csv"
OUT_AUDIT = MODEL_ROOT / "metadata" / "salux_refined_skeleton_5cls_audit.json"

CLASS_ROOTS = {
    "ok": PROJECT_ROOT / "batch_ok" / "fitted",
    "paper": PROJECT_ROOT / "batch_paper" / "fitted",
    "rock": PROJECT_ROOT / "batch_rock" / "fitted",
    "scissors": PROJECT_ROOT / "batch_scissors" / "fitted",
    "the-finger": PROJECT_ROOT / "batch_thefinger" / "fitted",
}


def main():
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline["action"].isin(CLASS_ROOTS)].copy()
    rows, missing, invalid = [], [], []
    for row in baseline.itertuples(index=False):
        refined_path = CLASS_ROOTS[row.action] / f"{row.sample_id}.npy"
        if not refined_path.exists():
            missing.append(f"{row.action}/{row.sample_id}")
            continue
        x = np.load(refined_path)
        if x.shape != (21, 3) or not np.isfinite(x).all():
            invalid.append(str(refined_path))
            continue
        rows.append({
            "action": row.action,
            "sample_id": row.sample_id,
            "split": row.split,
            "mediapipe_path": row.input_path,
            "refined_path": str(refined_path.resolve()),
        })

    out = pd.DataFrame(rows).sort_values(["split", "action", "sample_id"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    audit = {
        "source_rows_5cls": len(baseline),
        "matched_rows": len(out),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "missing": missing,
        "invalid": invalid,
        "class_counts": out["action"].value_counts().sort_index().to_dict(),
        "split_counts": out["split"].value_counts().sort_index().to_dict(),
        "refined_artifact": "class template fitted to 2D, followed by refinement",
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if missing or invalid or len(out) != len(baseline):
        raise RuntimeError("Refined skeleton metadata audit failed")


if __name__ == "__main__":
    main()
