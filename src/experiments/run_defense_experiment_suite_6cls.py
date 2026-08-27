
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_defense_ablation_6cls.py"
OUT_ROOT = ROOT / "results/defense_experiments_6cls"
SEEDS = [0, 1, 42]


def tag(model, normalization, views, weight, seed):
    return f"{model}_{normalization}_views{views}_lambda{str(weight).replace('.', 'p')}_seed{seed}"


def specs_for_suite(name):
    if name == "core":
        specs = []
        for seed in SEEDS:
            specs.extend([
                ("gru", "wrist_middle", 0, 0.0, seed),
                ("gru", "wrist_middle", 8, 0.0, seed),
                ("gru", "wrist_middle", 8, 0.3, seed),
            ])
        return specs
    if name == "views":
        return [
            ("gru", "wrist_middle", views, 0.3, seed)
            for seed in SEEDS
            for views in [2, 4]
        ]
    if name == "views_ce":
        return [
            ("gru", "wrist_middle", views, 0.0, seed)
            for seed in SEEDS
            for views in [2, 4]
        ]
    if name == "normalization":
        return [
            ("gru", "palm_robust", views, 0.3 if views else 0.0, seed)
            for seed in SEEDS
            for views in [0, 8]
        ]
    if name == "normalization_ce":
        return [
            ("gru", "palm_robust", 8, 0.0, seed)
            for seed in SEEDS
        ]
    if name == "model":
        return [
            ("mlp", "wrist_middle", views, 0.3 if views else 0.0, seed)
            for seed in SEEDS
            for views in [0, 8]
        ]
    if name == "model_ce":
        return [
            ("mlp", "wrist_middle", 8, 0.0, seed)
            for seed in SEEDS
        ]
    if name == "raw_lambda_sweep":
        return [
            ("gru", "wrist_middle", 8, weight, seed)
            for seed in SEEDS
            for weight in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        ]
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["core", "views", "views_ce", "normalization", "normalization_ce", "model", "model_ce", "raw_lambda_sweep"], required=True)
    parser.add_argument("--seed-filter", type=int, choices=SEEDS, default=None)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    failures = []
    for model, normalization, views, weight, seed in specs_for_suite(args.suite):
        if args.seed_filter is not None and seed != args.seed_filter:
            continue
        experiment = tag(model, normalization, views, weight, seed)
        out_dir = OUT_ROOT / experiment
        summary = out_dir / "summary.json"
        if summary.exists():
            try:
                if json.loads(summary.read_text(encoding="utf-8")).get("status") == "complete":
                    print(f"[skip] {experiment}", flush=True)
                    continue
            except Exception:
                pass
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(TRAINER),
            "--seed", str(seed),
            "--num-views", str(views),
            "--lambda-weight", str(weight),
            "--normalization", normalization,
            "--model", model,
        ]
        print(f"[run] {experiment}", flush=True)
        with (out_dir / "run.log").open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=ROOT.parent,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            failures.append(experiment)
            print(f"[failed] {experiment}; see {out_dir / 'run.log'}", flush=True)
        else:
            result = json.loads(summary.read_text(encoding="utf-8"))
            droh = result["evaluations"]["droh_raw"]
            print(
                f"[done] {experiment} acc={droh['accuracy']:.6f} "
                f"macro_f1={droh['macro_f1']:.6f} "
                f"seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )

    if failures:
        raise SystemExit(f"Failed experiments: {failures}")
    print(f"[suite complete] {args.suite}", flush=True)


if __name__ == "__main__":
    main()
