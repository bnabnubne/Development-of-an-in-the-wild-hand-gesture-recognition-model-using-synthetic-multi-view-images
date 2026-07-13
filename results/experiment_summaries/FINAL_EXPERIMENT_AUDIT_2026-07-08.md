# Final Experiment Audit (2026-07-08)

## Scope and integrity checks

- Task: six-class static hand-gesture recognition (`ok`, `paper`, `rock`,
  `scissors`, `the-finger`, `thumb`).
- Salux split: 4,582 train / 982 validation / 983 internal test.
- DrOh: 675 external-test samples; never used for checkpoint selection.
- Final matched experiment: five independent seeds 0, 1, 7, 21, and 42.
- Audit passed for all 15 final prediction files: identical 675-sample ordering,
  identical ground truth, exactly six classes, and no missing rows.
- Accuracy is `correct / total`. Across-seed SD is sample SD (`ddof=1`).
- Ensemble prediction is the argmax after averaging class probabilities from equal-size
  model sets. It has no across-seed SD because it is one fused predictor.

## Experiments that are complete and meaningful

### Submitted thesis protocol

1. Raw normalized single-view baseline on Salux and DrOh.
2. Raw normalized Blender8 training with original-anchored consistency, lambda=0.3.
3. Fitted single-view baseline evaluated in raw and oracle-fitted domains.
4. Fitted Blender8 consistency evaluated in raw and oracle-fitted domains.
5. Fitted lambda sweep, used only as an oracle/refinement diagnostic.
6. Exploratory label-free template selection/fusion, reported as below the deployable
   raw multiview pipeline and not used as a final claim.

### Post-submission robustness and ablations

1. Three-seed raw baseline, Blender8 CE-only, and Blender8 consistency from scratch.
2. Number of training views: 0, 2, 4, and 8.
3. Full raw lambda sweep: 0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0.
4. Architecture control: GRU versus MLP.
5. Normalization control: wrist-middle scale versus palm-centroid robust scale.
6. Clean camera-coordinate/canonical factorial to isolate the viewpoint mechanism.
7. Consistency schedule study: CE warm-up followed by low-LR consistency fine-tuning.
8. Critical matched lambda=0 low-LR control, showing what extra fine-tuning alone does.
9. Five-seed replication of original CE, matched lambda=0 control, and matched
   lambda=0.3 consistency.
10. Equal-size five-model probability ensembles and paired significance tests.
11. Salux internal-test check and per-class DrOh analysis for the final ensembles.

## Final controlled result

### Five-seed single-model means

| System | DrOh accuracy | Macro-F1 |
|---|---:|---:|
| Original Blender8 CE checkpoint | 80.68 +/- 1.73% | 80.34 +/- 1.75% |
| Matched low-LR CE control, lambda=0 | 81.24 +/- 1.86% | 80.80 +/- 1.98% |
| **Matched low-LR consistency, lambda=0.3** | **83.44 +/- 1.97%** | **83.21 +/- 2.02%** |

The mean improvement over the matched control is +2.19 pp. The five-pair t-test gives
p=0.126, so this seed-level test is not statistically significant with n=5.

### Equal-size five-model ensembles

| System | Salux internal | DrOh external | DrOh macro-F1 |
|---|---:|---:|---:|
| Raw single-view baseline | — | 76.74% (518/675) | 75.75% |
| Original Blender8 CE | 97.25% | 84.30% (569/675) | 84.10% |
| Matched low-LR CE control | 97.15% | 83.70% (565/675) | 83.53% |
| **Matched low-LR consistency, lambda=0.3** | **97.25%** | **85.33% (576/675)** | **85.19%** |

For the clean matched ensemble comparison, consistency is +1.63 pp over lambda=0.
There are 17 consistency-only correct samples and 6 control-only correct samples;
exact McNemar p=0.0347 and paired-sample bootstrap 95% CI is [0.30, 2.96] pp.

The strongest defensible claim is therefore: with the same CE warm-up, low-LR schedule,
five seeds, data, validation rule, architecture, and ensemble size, adding lambda=0.3
consistency improves external DrOh accuracy from 83.70% to 85.33% while preserving
97.25% Salux internal accuracy.

For the full-method demo, the equal-size five-seed comparison is raw single-view
baseline 76.74% versus final Blender8 + consistency 85.33%, a +8.59 pp gain. Exact
McNemar p=2.87e-9 and bootstrap 95% CI is [5.78, 11.41] pp. This is a combined-method
comparison; the matched 83.70% versus 85.33% comparison above isolates consistency.

## What is not a final claim

- Oracle-fitted DrOh results use ground-truth-template information and are diagnostics,
  not deployable hand-in-the-wild accuracy.
- Label-free fitting remains exploratory and does not beat the raw multiview pipeline.
- The old six-member CE/consistency hybrid at 84.30% is no longer the highlighted best
  system; it also had a different ensemble size and no significant gain over CE-only.
- The clean viewpoint factorial isolates rotation robustness but is not a replacement
  for the real DrOh external test.
- The supervised-contrastive alpha sweep in the MAPR paper is a separate SupCon method.
  It is not part of the six-class thesis pipeline's final consistency claim and is not
  required to validate lambda. Adding it now would be a new comparative branch, not a
  missing control for the stated method.

## Remaining optional work

The core experiment matrix is sufficient for the defense. No further hyperparameter
sweep on DrOh should be run, because repeated external-test-driven selection would weaken
the evaluation protocol. The only useful additions are presentation or future-work items:

1. Update the webcam/image demo to load the final five lambda=0.3 checkpoints and show
   probability-ensemble inference from one normalized skeleton.
2. Add latency, parameter count, and model-size measurements for the baseline, one
   consistency model, and the five-model ensemble.
3. Present the saved final confusion matrix and the per-class table, explicitly noting
   the +8.93 pp Scissors gain and -3.67 pp Rock regression.
4. If time and data permit, collect a new viewpoint-stratified RGB test set with angle
   bins. This would test the visual claim directly without reusing DrOh for tuning.
5. Treat deployable, label-free refinement as future work unless a template-free or
   independently selected fitting method is completed and evaluated without oracle labels.

More seeds could narrow uncertainty, but five independent seeds plus the matched control
and paired sample-level test are adequate for the current defense. The highest-value next
step is the faithful end-to-end demo, not another lambda or architecture sweep.
