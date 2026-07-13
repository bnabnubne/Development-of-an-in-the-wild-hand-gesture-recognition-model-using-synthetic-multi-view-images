import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "camera_ready"


def collect(experiment):
    rows = []
    for path in sorted((RESULTS / experiment).glob("seed_*/summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def stats(rows, key):
    values = np.array([row[key] for row in rows], dtype=float) * 100
    return {
        "values": values.round(4).tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


experiments = [
    "gru_original",
    "gru_mv4_lambda0",
    "gru_mv8_lambda0",
    "gru_mv8_lambda0.3",
    "rot3d_clean",
    "gcn_original",
    "gcn_mv8_lambda0",
    "gcn_mv8_lambda0.3",
]

summary = {}
for experiment in experiments:
    rows = collect(experiment)
    if rows:
        summary[experiment] = {
            "seeds": [row["seed"] for row in rows],
            "canonical_test_accuracy_percent": stats(rows, "canonical_test_accuracy"),
            "handinwild_test_accuracy_percent": stats(rows, "handinwild_test_accuracy"),
        }

ce = {row["seed"]: row for row in collect("gru_mv8_lambda0")}
consistency = {row["seed"]: row for row in collect("gru_mv8_lambda0.3")}
paired_seeds = sorted(set(ce) & set(consistency))
paired_gains = np.array([
    (consistency[seed]["handinwild_test_accuracy"] - ce[seed]["handinwild_test_accuracy"]) * 100
    for seed in paired_seeds
])
summary["paired_consistency_gain_over_ce_only"] = {
    "seeds": paired_seeds,
    "gains_percentage_points": paired_gains.round(4).tolist(),
    "mean_gain": float(paired_gains.mean()),
    "sample_std": float(paired_gains.std(ddof=1)) if len(paired_gains) > 1 else None,
}

output = RESULTS / "aggregate_summary.json"
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
