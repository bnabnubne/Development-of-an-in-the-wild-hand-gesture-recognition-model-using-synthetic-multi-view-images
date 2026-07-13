import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
DROH_CSV = PROJECT_ROOT.parent / "April" / "metadata" / "droh_baseline.csv"
REFINED_ROOT = MODEL_ROOT / "data" / "droh_refined"
OUT_CSV = MODEL_ROOT / "metadata" / "droh_refined_skeleton_5cls.csv"
OUT_AUDIT = MODEL_ROOT / "metadata" / "droh_refined_skeleton_5cls_audit.json"
CLASSES = ["ok", "paper", "rock", "scissors", "the-finger"]


def main():
    source = pd.read_csv(DROH_CSV)
    source = source[source["action"].isin(CLASSES)].copy()
    rows, missing, invalid = [], [], []
    for row in source.itertuples(index=False):
        refined = REFINED_ROOT / row.action / "fitted" / f"{row.sample_id}.npy"
        if not refined.exists():
            missing.append(f"{row.action}/{row.sample_id}")
            continue
        x = np.load(refined)
        if x.shape != (21, 3) or not np.isfinite(x).all():
            invalid.append(str(refined))
            continue
        rows.append({
            "action": row.action,
            "sample_id": row.sample_id,
            "split": "test",
            "mediapipe_path": row.input_path,
            "refined_path": str(refined.resolve()),
        })
    out = pd.DataFrame(rows).sort_values(["action", "sample_id"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    audit = {
        "source_rows_5cls": len(source), "matched_rows": len(out),
        "missing_count": len(missing), "invalid_count": len(invalid),
        "missing": missing, "invalid": invalid,
        "class_counts": out["action"].value_counts().sort_index().to_dict(),
        "refined_root": str(REFINED_ROOT),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if missing or invalid or len(out) != len(source):
        raise RuntimeError("DrOh refined metadata audit failed")


if __name__ == "__main__":
    main()
