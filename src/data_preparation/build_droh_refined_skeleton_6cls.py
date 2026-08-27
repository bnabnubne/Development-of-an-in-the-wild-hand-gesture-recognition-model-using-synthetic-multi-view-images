
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parent
DROH_CSV = Path("./metadata/droh_baseline.csv")
REFINED_ROOT = MODEL_ROOT / "data/droh_refined"
OUT_CSV = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls.csv"
OUT_AUDIT = MODEL_ROOT / "metadata/droh_refined_skeleton_6cls_audit.json"


def main():
    source = pd.read_csv(DROH_CSV)
    rows, missing, invalid = [], [], []
    for row in source.itertuples(index=False):
        refined = REFINED_ROOT / row.action / "fitted" / f"{row.sample_id}.npy"
        if not refined.exists():
            missing.append(f"{row.action}/{row.sample_id}")
            continue
        value = np.load(refined, allow_pickle=False)
        if value.shape != (21, 3) or not np.isfinite(value).all():
            invalid.append(str(refined))
            continue
        rows.append({
            "action": "thumb" if row.action in {"thumbup", "thumbdown"} else row.action,
            "source_action": row.action, "sample_id": row.sample_id, "split": "test",
            "raw_path": row.input_path, "refined_path": str(refined.resolve()),
        })
    out = pd.DataFrame(rows).sort_values(["source_action", "sample_id"])
    out.to_csv(OUT_CSV, index=False)
    audit = {
        "source_rows": len(source), "matched_rows": len(out),
        "missing_count": len(missing), "invalid_count": len(invalid),
        "missing": missing, "invalid": invalid,
        "class_counts_merged": out.action.value_counts().sort_index().to_dict(),
        "source_action_counts": out.source_action.value_counts().sort_index().to_dict(),
        "warning": "Fitted DrOh uses the ground-truth class to select a template; oracle diagnostic only.",
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if missing or invalid or len(out) != len(source):
        raise RuntimeError("Incomplete DrOh fitted 6-class manifest")


if __name__ == "__main__":
    main()
