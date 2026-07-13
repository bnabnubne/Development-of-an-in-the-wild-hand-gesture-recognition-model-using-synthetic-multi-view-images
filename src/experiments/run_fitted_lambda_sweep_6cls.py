"""Run the fitted MV-consistency lambda sweep sequentially."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LAMBDAS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def tag_float(value):
    return str(value).replace(".", "p")


def main():
    for value in LAMBDAS:
        out = ROOT / "results" / f"fitted_anchor_multiview_lambda_{tag_float(value)}_6cls"
        summary = out / "summary.json"
        if summary.exists():
            print(f"[skip] lambda={value:g} already complete: {summary}", flush=True)
            continue
        print(f"[run] lambda={value:g}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "train_fitted_anchor_multiview_lambda_sweep_6cls.py"),
                "--lambda-weight",
                str(value),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
