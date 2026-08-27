# Development of an in-the-wild hand gesture recognition model using synthetic multi-view images

This repository contains the source code and small public files used for my undergraduate thesis.

The project studies static hand gesture recognition from RGB images in uncontrolled settings. The main input is a 3D hand skeleton extracted by MediaPipe. Blender is used to create synthetic camera views from the same hand pose. These views are used during training so the model can handle changes in hand viewpoint better.

## Problem

Hand gesture models often work well when the camera angle is similar to the training data. They can fail when the hand is rotated, tilted, or partly foreshortened. This project tests whether synthetic multi-view hand skeletons can reduce that problem.

The target gestures are:

- OK
- Paper
- Rock
- Scissors
- The-Finger
- Thumb

## Method

The main pipeline is:

1. Read an RGB hand image.
2. Extract 3D hand landmarks with MediaPipe Hands.
3. Normalize the skeleton by hand side, center, scale, and palm direction.
4. Train a GRU-based classifier on the normalized skeleton.
5. Add Blender-generated camera views during training.
6. Evaluate on an external in-the-wild test set.

The project also includes fitted-template experiments. Those experiments are useful for analysis, but they are not the main deployable setting because some fitted tests use gesture-template information.

## Data

The code uses two main data sources:

- Salux: used for training, validation, and internal testing.
- DrOh: used as an external hand-in-the-wild test set.

The full image datasets are not included in this repository. Only small metadata files are kept. Local dataset paths must be updated before running the scripts on another machine.

## Main result

The submitted thesis result compares a raw skeleton baseline with a model trained using eight Blender views and a consistency loss.

- Raw skeleton baseline on DrOh: 77.93 percent.
- Raw skeleton with Blender8 consistency on DrOh: 78.96 percent.

Later checks in this repository also include multi-seed runs, ensemble tests, fitted-template analysis, and viewpoint ablations. These results are kept in `results/experiment_summaries`.

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

Use a Python environment and install the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Some scripts need Blender and should be run with Blender's Python runtime.

## Run

Most scripts are standalone Python files. Update the path constants near the top of each script before running.

Example:

```bash
python src/evaluation/test_droh_external.py
```

Webcam demo scripts are in `src/demo`.

```bash
python src/demo/webcam_demo.py
```

## Not included

The repository does not include:

- full RGB datasets
- rendered image folders
- model checkpoints
- training logs
- large prediction dumps
- local debug outputs

The repository is meant to keep the code and small files needed to understand the project, not to be a full dataset release.
