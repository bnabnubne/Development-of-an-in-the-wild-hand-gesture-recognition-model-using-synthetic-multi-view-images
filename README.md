# Development of an In-the-Wild Hand Gesture Recognition Model Using Synthetic Multi-View Images

This repository is a cleaned thesis/project snapshot for developing a 3D-skeleton hand gesture recognition model using synthetic multi-view camera-space views.

The project investigates cross-dataset static hand gesture recognition with:

- MediaPipe-based 3D hand skeleton extraction
- Blender-based synthetic multi-view generation
- GRU-based skeleton classification
- ROT3D augmentation baseline
- Multi-view cross-entropy baseline
- Original-anchored multi-view consistency learning
- Supervised contrastive variants

## Repository contents

```text
.
├── *.py                 # Training, testing, preprocessing, metadata and experiment scripts
├── audit/               # Dataset/skeleton audit scripts
├── assets/              # Small figures used for reports/slides
├── blender/             # Blender scene for synthetic multi-view rendering
├── docs/                # Thesis report and defense slides
├── metadata/            # CSV metadata and split files only; no full skeleton dataset
└── results/             # Lightweight summaries/reports/audit outputs; no model checkpoints
```

## Important data note

The full datasets are intentionally not included in this repository because they are large and may contain raw/derived data not suitable for public version control.

Excluded on purpose:

- full image/skeleton datasets
- generated `.npy` skeleton folders
- model checkpoints (`.pt`, `.pth`, `.ckpt`)
- logs/cache files
- zipped dataset archives

The `metadata/` folder contains CSV files describing splits and paths from the original local workspace. To fully reproduce experiments, place the corresponding datasets in the expected structure or update paths in the scripts.

## Main camera-ready experiment scripts

- `camera_ready_experiments.py`: standardized 7-class runner for GRU/GCN, synthetic-view CE-only baseline, and consistency experiments.
- `clean_rot3d_camera_ready.py`: ROT3D baseline with augmentation only on the CanonicalSet training split and original skeletons for validation/testing.
- `aggregate_camera_ready.py`: aggregates camera-ready summaries across seeds.

## Main reported results

The camera-ready result summaries are stored under `results/camera_ready/`.

Key 7-class results:

| Method | CanonicalSet Acc. (%) | HandinWildSet Acc. (%) |
|---|---:|---:|
| 3D GRU Baseline | 94.20 | 62.96 |
| ROT3D-clean | 95.93 | 70.22 |
| MV-CE (4 views) | 95.22 | 59.56 |
| MV-Consistency (4 views) | 91.56 | 67.70 |
| MV-SupCon (4 views) | 93.90 | 64.44 |
| MV-CE (8 views) | 94.51 | 68.44 |
| MV-Consistency (8 views) | 94.61 | 75.85 |
| MV-SupCon (8 views) | 95.32 | 72.29 |

## Environment

Install the main Python dependencies:

```bash
pip install -r requirements.txt
```

The experiments were developed with Python, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, OpenCV, MediaPipe, and Blender for synthetic view generation.

## Example usage

Run a standardized 8-view CE-only experiment:

```bash
python camera_ready_experiments.py --backbone gru --views 8 --consistency-weight 0 --seed 42
```

Run the 8-view original-anchored consistency model:

```bash
python camera_ready_experiments.py --backbone gru --views 8 --consistency-weight 0.3 --seed 42
```

Run the clean ROT3D baseline:

```bash
python clean_rot3d_camera_ready.py --seed 42
```

Aggregate camera-ready results:

```bash
python aggregate_camera_ready.py
```

## Thesis documents

- `docs/thesis_report_final.pdf`
- `docs/thesis_slides.pdf`

