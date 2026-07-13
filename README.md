# Development of an In-the-Wild Hand Gesture Recognition Model Using Synthetic Multi-View Images

This repository contains the cleaned source-code package for the undergraduate thesis project (ĐATN):

**Development of an in-the-wild hand gesture recognition model using synthetic multi-view images.**

The repository is organized as a source-code archive for thesis management and review. It intentionally excludes the full datasets, generated render folders, training logs, and model checkpoints.

## Project overview

The project studies static hand gesture recognition in uncontrolled conditions. The main pipeline includes:

- MediaPipe hand landmark extraction from RGB images
- 2D/3D skeleton normalization and metadata generation
- Blender-based pose fitting and synthetic multi-view rendering
- Skeleton-based and RGB/skeleton-fusion recognition models
- Multi-view consistency learning and ablation experiments
- Thesis figures, summaries, report, and slides

## Repository structure

```text
.
├── src/
│   ├── data_preparation/     # metadata building, extraction, bbox/refined-skeleton preparation
│   ├── blender_pipeline/     # pose fitting, refinement, rendering, overlay and viewport scripts
│   ├── training/             # model training scripts
│   ├── evaluation/           # testing/evaluation scripts
│   ├── experiments/          # experiment runners, audits, summaries and analysis scripts
│   ├── demo/                 # webcam/demo utilities
│   ├── figures/              # scripts that generate thesis figures
│   └── legacy_april/         # earlier prototype scripts kept for traceability
├── metadata/
│   ├── thesis/               # compact thesis metadata CSV/JSON files
│   └── april/                # earlier split/protocol metadata from the April workspace
├── results/
│   ├── thesis_figures/       # generated figures used in the thesis
│   └── experiment_summaries/ # lightweight summaries and reports only
├── blender/
│   ├── scenes/               # Blender scenes used by the synthetic-view pipeline
│   └── templates/            # gesture-specific template scenes
└── docs/
    ├── thesis_report_final.pdf
    └── thesis_slides.pdf
```

## What is intentionally excluded

The following files are not pushed to GitHub:

- full RGB datasets
- generated `.npy` skeleton datasets
- generated render folders
- batch fitting logs and overlays
- model checkpoints (`.pt`, `.pth`, `.ckpt`)
- large prediction dumps
- cache files (`__pycache__`, `.DS_Store`, etc.)

Only code, metadata, lightweight result summaries, Blender templates/scenes, report, and slides are included.

## Main source folders

- `src/data_preparation/`: scripts for building metadata, extracting hand boxes, RGB/skeleton metadata, refined skeleton metadata, and dataset manifests.
- `src/blender_pipeline/`: scripts for hand-pose fitting, template refinement, batch rendering, and synthetic multi-view generation.
- `src/training/`: scripts for training skeleton, RGB, RGB-skeleton fusion, refined-skeleton, and multiview-consistency models.
- `src/evaluation/`: evaluation scripts for external Hand-in-the-Wild testing and ablation settings.
- `src/experiments/`: higher-level experiment suites and result summarization scripts.
- `src/demo/`: webcam and final demo scripts.
- `src/figures/`: scripts used to generate thesis figures.
- `src/legacy_april/`: earlier prototype code from the April workspace, retained for reproducibility/history but separated from the final thesis code.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Some scripts require Blender and should be executed through Blender's Python runtime.

## Data paths

Many scripts expect local dataset paths. Since the full datasets are not included, update the path constants in the relevant scripts or place data in the expected local structure before running.

## Thesis documents

- `docs/thesis_report_final.pdf`
- `docs/thesis_slides.pdf`

## Notes

This repository is intended for source-code management and thesis reproducibility at a clean-project level. It is not a full data release.

