Development of an in-the-wild hand gesture recognition model using synthetic multi-view images

This repository contains the source code and small public files used for my undergraduate thesis.

The project studies static hand gesture recognition from RGB images. The main input is a hand skeleton extracted by MediaPipe. Synthetic views are generated in Blender and used during training to improve robustness to hand viewpoint changes.

Main folders

src/data_preparation
Scripts for building metadata, extracting MediaPipe landmarks, preparing hand boxes, and writing skeleton CSV files.

src/blender_pipeline
Blender scripts for fitting poses, refining templates, rendering views, and checking overlays.

src/training
Training scripts for skeleton, RGB, fusion, refined skeleton, and multi-view consistency models.

src/evaluation
Evaluation scripts for Salux, DrOh, and ablation runs.

src/experiments
Small scripts for running experiment batches and collecting summary files.

src/demo
Webcam and image demo code.

src/figures
Scripts used to draw the figures in the thesis report.

metadata
Small CSV and JSON files kept for reproducibility.

results
Selected result summaries and thesis figures.

blender
The Blender scenes and gesture templates used by the rendering pipeline.

docs
The report, slides, and paper PDF files.

Install

Use a Python environment and install the packages in requirements.txt.

pip install -r requirements.txt

Some scripts need Blender and should be run with Blender's Python runtime.

Data

The full datasets, rendered images, checkpoints, and training logs are not included. Most scripts expect local paths, so update the path constants before running them on another machine.

The repository is meant to keep the code and small files needed to understand the project, not to be a full dataset release.
