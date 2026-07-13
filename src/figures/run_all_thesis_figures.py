from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "figure_4_4_main_droh_accuracy.py",
    "figure_4_5_lambda_sweep_curve.py",
    "figure_4_8_template_fitting_examples.py",
    "figure_4_9_before_after_refinement.py",
    "figure_4_12_recognition_examples.py",
]


def main():
    for script in SCRIPTS:
        print(f"[figure] {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == "__main__":
    main()
