# Meaningful experiment inventory

This directory has been cleaned to retain only experiments that support a clear thesis or defense conclusion.

## 1. Submitted raw-domain recognition

| Experiment | Salux raw | DrOh raw | Meaning |
|---|---:|---:|---|
| Raw baseline, seed 42 | 96.85% | 77.93% | Deployable single-view baseline |
| Raw + 8-view consistency, seed 42, lambda=0.3 | 97.05% | 78.96% | Submitted main result; +1.03 pp external accuracy |

Artifacts:

- `paper_controlled_raw_baseline_6cls/`
- `paper_controlled_raw_mv_6cls/`

## 2. Template-conditioned fitted-domain analysis

| Experiment | Salux fitted | Salux raw | DrOh raw | DrOh fitted oracle | Meaning |
|---|---:|---:|---:|---:|---|
| Fitted baseline | 99.90% | 18.92% | 20.74% | 81.78% | Establishes severe raw/fitted mismatch |
| Fitted + Blender-8 MV, lambda=0.3 | 99.90% | 47.41% | 36.15% | 89.33% | MV helps inside fitted domain but does not make it deployable on raw inputs |

Artifacts:

- `fitted_baseline_6cls/`
- `fitted_anchor_multiview_6cls/`

The fitted DrOh result is oracle-only because the ground-truth gesture template is used during fitting.

## 3. Fitted lambda sweep

Lambda values `0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0` are retained because they support one conclusion: changing lambda cannot solve the raw/fitted representation mismatch. The best DrOh raw result is only 40.00%, while fitted-oracle performance can reach 98.96%.

Artifacts:

- `fitted_anchor_multiview_lambda_*_6cls/`
- `fitted_anchor_multiview_lambda_sweep_6cls_summary.json`

## 4. Exploratory label-free template recognition

All 675 DrOh samples were fitted to all six templates, giving 4,050 complete fits. Minimum-cost template recognition reaches 67.41%. This is retained as a negative but meaningful result: template selection is itself a classifier and remains below the raw baseline.

Artifact:

- `label_free_template_pipeline_6cls_droh/`

Smoke and parallel-smoke outputs were removed.

## 5. Post-submission defense experiments

Thirty-nine controlled runs cover three seeds, CE-only versus consistency, 0/2/4/8 views, GRU versus MLP, and normalization ablations.

Main results:

| Protocol | DrOh accuracy, mean +/- std | Macro F1, mean +/- std |
|---|---:|---:|
| Raw GRU | 75.41 +/- 4.23% | 74.51 +/- 4.49% |
| GRU + 8-view consistency | 78.81 +/- 1.86% | 78.10 +/- 2.33% |
| GRU + 8-view CE-only | **81.33 +/- 1.82%** | **81.00 +/- 1.87%** |
| Fixed three-seed MV CE-only ensemble | **84.00%** | **83.83%** |

Conclusion: supervised eight-view augmentation is the strongest reproducible component. The `lambda=0.3` cosine consistency term is not consistently beneficial.

Artifacts:

- `defense_experiments_6cls/`
- `defense_demo_examples/`
- `defense_experiments_6cls/FINAL_DEFENSE_HANDOFF.md`

## 6. Report and defense figures

- `thesis_figures/`

These include the submitted-result figures, fitting/refinement illustrations, confusion matrices, and post-submission defense plots.

## Removed categories

- Obsolete five-class pipelines.
- Duplicate old-pipeline reruns superseded by paper-controlled results.
- Raw-anchor plus fitted-view experiments based on an incorrect protocol interpretation.
- Bridge-loss experiments not part of the submitted method and not required for the final defense conclusion.
- Early refined/raw hybrid and Procrustes variants that did not establish a stable improvement.
- RGB, RGB-skeleton fusion and hand-crop development runs that were unstable and excluded from the final thesis protocol.
- Smoke tests, debug outputs and stale aggregate JSON files containing superseded conclusions.

