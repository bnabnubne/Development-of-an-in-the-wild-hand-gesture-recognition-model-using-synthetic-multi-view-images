# Cleanup manifest

Cleanup date: 2026-07-07.

## Retained

- `MEANINGFUL_EXPERIMENTS.md`
- `CLEANUP_MANIFEST.md`
- `paper_controlled_raw_baseline_6cls/`
- `paper_controlled_raw_mv_6cls/`
- `fitted_baseline_6cls/`
- `fitted_anchor_multiview_6cls/`
- `fitted_anchor_multiview_lambda_0p05_6cls/`
- `fitted_anchor_multiview_lambda_0p1_6cls/`
- `fitted_anchor_multiview_lambda_0p3_6cls/`
- `fitted_anchor_multiview_lambda_0p5_6cls/`
- `fitted_anchor_multiview_lambda_0p7_6cls/`
- `fitted_anchor_multiview_lambda_0p9_6cls/`
- `fitted_anchor_multiview_lambda_1p0_6cls/`
- `fitted_anchor_multiview_lambda_sweep_6cls_summary.json`
- `label_free_template_pipeline_6cls_droh/`
- `defense_experiments_6cls/`
- `defense_demo_examples/`
- `thesis_figures/`

## Removed

Everything else previously located directly under `model/results/`, including obsolete five-class runs, duplicate reruns, bridge experiments, misunderstood raw/fitted hybrids, unsuccessful RGB development runs, smoke/debug outputs, and stale summary files.

Source code was not deleted solely because an experiment result was removed. Keeping code preserves provenance and avoids breaking imports; the authoritative retained artifacts are the items listed above.
