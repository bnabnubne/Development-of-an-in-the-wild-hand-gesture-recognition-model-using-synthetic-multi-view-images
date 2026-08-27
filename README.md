# Development of an in-the-wild hand gesture recognition model using synthetic multi-view images

This repository contains the source code and small public files used for my undergraduate thesis.

The project studies static hand gesture recognition from RGB images. The main input is a hand skeleton extracted by MediaPipe. Synthetic views are generated in Blender and used during training to improve robustness to hand viewpoint changes.

## Main folders

- `src/data_preparation`: metadata, MediaPipe landmarks, hand boxes, and skeleton CSV files.
- `src/blender_pipeline`: Blender pose fitting, template refinement, rendering, and overlays.
- `src/training`: model training scripts.
- `src/evaluation`: evaluation scripts for Salux, DrOh, and ablation runs.
- `src/experiments`: experiment runners and summary scripts.
- `src/demo`: webcam and image demo code.
- `src/figures`: scripts used to draw thesis figures.
- `metadata`: small CSV and JSON files.
- `results`: selected result summaries and thesis figures.
- `blender`: Blender scenes and gesture templates.
- `docs`: report, slides, and paper PDF files.

## Install

Use a Python environment and install the packages in requirements.txt.

```bash
pip install -r requirements.txt
```

Some scripts need Blender and should be run with Blender's Python runtime.

## Data

The full datasets, rendered images, checkpoints, and training logs are not included. Most scripts expect local paths, so update the path constants before running them on another machine.

The repository is meant to keep the code and small files needed to understand the project, not to be a full dataset release.
