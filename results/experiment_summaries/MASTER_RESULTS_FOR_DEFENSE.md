# Master Results for Thesis Defense

## 1. Result hierarchy

Results must not be mixed across these categories:

1. Submitted thesis result: fixed seed 42, original training protocol.
2. Post-submission single-model result: mean and sample SD across independent seeds.
3. Post-submission ensemble: one fused predictor obtained by averaging model probabilities.
4. Oracle/refinement diagnostics: use ground-truth template information and are not deployable.
5. Controlled synthetic tests: isolate a mechanism but are not real DrOh accuracy.

For every DrOh result, accuracy is `number correct / 675`. Mean +/- SD is computed using
sample SD (`ddof=1`) across the stated independently trained seeds. An ensemble is a
single fused predictor and therefore has no across-seed SD. Earlier ablations use seeds
0, 1, and 42; the final matched consistency comparison uses seeds 0, 1, 7, 21, and 42.

## 2. Submitted thesis results

| Experiment | Training difference | DrOh input | Accuracy | Status |
|---|---|---|---:|---|
| Raw baseline | Normalized detector skeleton, no MV | Normalized raw | 77.93% (526/675) | Main submitted baseline, seed 42 |
| Raw + Blender8 consistency | Anchor + 8 views, CE + cosine consistency, lambda=0.3 | Normalized raw | **78.96% (533/675)** | Main submitted result, seed 42 |
| Fitted baseline | Template-fitted Salux, no MV | DrOh raw | 20.74% | Fitted-to-raw mismatch |
| Fitted + Blender8 consistency | Fitted anchor + 8 views, lambda=0.3 | DrOh raw | 36.15% | Fitted-to-raw mismatch remains |
| Fitted baseline | Template-fitted Salux | DrOh oracle-fitted | 81.78% | Oracle diagnostic |
| Fitted + Blender8 consistency | Fitted anchor + 8 views, lambda=0.3 | DrOh oracle-fitted | 89.33% | Oracle diagnostic |

The submitted claim remains a +1.03 percentage-point gain from 77.93% to 78.96% while
retaining single-view inference.

## 3. Post-submission single-model comparison

All rows use the same normalized skeleton representation, GRU, Salux split, Blender8
camera set, and DrOh single-view evaluation.

| Experiment | Main difference | DrOh accuracy (mean +/- SD) | Macro-F1 | Interpretation |
|---|---|---:|---:|---|
| Raw baseline | No synthetic views | 75.41 +/- 4.23% | 74.51 +/- 4.49% | Three-seed baseline |
| Blender8 CE-only | CE on anchor + 8 views, no consistency penalty | 81.33 +/- 1.82% | 81.00 +/- 1.87% | MV augmentation gain |
| Blender8 fixed consistency | CE + cosine consistency from epoch 1, lambda=0.3 | 78.81 +/- 1.86% | 78.10 +/- 2.33% | Submitted loss extended to three seeds |
| CE warm-up -> consistency fine-tune | Start from CE-only checkpoint, then lambda=0.3 cosine fine-tuning; all-view validation selection | 82.62 +/- 1.11% | 82.34 +/- 1.11% | Preliminary three-seed schedule result |

Per-seed comparison for the best consistency schedule:

| Seed | Blender8 CE-only | CE warm-up -> consistency lambda=0.3 | Gain |
|---:|---:|---:|---:|
| 0 | 82.07% | 83.26% | +1.19 pp |
| 1 | 82.67% | 83.26% | +0.59 pp |
| 42 | 79.26% | 81.33% | +2.07 pp |

This preliminary result motivated the five-seed matched-control experiment below. The
matched control is necessary because extra low-learning-rate training and all-view
checkpoint selection can themselves change accuracy even when lambda=0.

### Final matched five-seed control

All three rows use the exact same five CE-pretrained seeds (0, 1, 7, 21, 42). The last
two rows use the same low learning rate, epoch budget, validation rule, data, and model;
only the consistency weight differs.

| System | Difference | DrOh accuracy mean +/- SD | Macro-F1 mean +/- SD |
|---|---|---:|---:|
| Original Blender8 CE checkpoint | Before the added fine-tuning stage | 80.68 +/- 1.73% | 80.34 +/- 1.75% |
| Matched low-LR CE control | Same fine-tuning schedule, lambda=0 | 81.24 +/- 1.86% | 80.80 +/- 1.98% |
| **Matched low-LR consistency** | **Same schedule, lambda=0.3** | **83.44 +/- 1.97%** | **83.21 +/- 2.02%** |

The consistency row improves the matched lambda=0 control by +2.19 pp on average. With
only five seed pairs, the seed-level paired t-test is not significant (p=0.126), so the
mean difference should be described as a consistent numerical improvement, not as a
population-level proof from seeds alone.

## 4. Deployable single-view ensembles

Every member receives the same one normalized DrOh skeleton. No test-time synthetic
view generation or voting over generated views is used.

| System | Members/fusion | Correct | DrOh accuracy | Macro-F1 | Role |
|---|---|---:|---:|---:|---|
| Raw baseline ensemble | 3 seeds, mean probabilities | 516/675 | 76.44% | 75.44% | Ensemble baseline |
| Fixed consistency ensemble | 3 lambda=0.3 models trained from scratch | 538/675 | 79.70% | 78.89% | Consistency improves baseline |
| Consistency-finetuned ensemble | 3 lambda=0.3 fine-tuned models | 561/675 | 83.11% | 82.91% | Consistency-only ensemble |
| Blender8 CE-only ensemble | 3 CE-only seeds | 567/675 | **84.00%** | **83.83%** | Strongest equal-size three-model ensemble |
| CE + consistency hybrid ensemble | Equal mean of 3 CE-only + 3 lambda=0.3 fine-tuned models | 569/675 | 84.30% | 84.15% | Earlier six-member diagnostic |

The six-model hybrid improves only +0.30 pp over the three-model CE ensemble. The paired
McNemar p-value is 0.791 (6 CE-only-only correct cases versus 8 hybrid-only correct cases),
so the numerical gain is not statistically significant. This is retained as an earlier
three-seed diagnostic, not the final highlighted system.

### Final equal-size five-model ensembles

| System | Members/fusion | Correct | DrOh accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Raw single-view baseline | 5 seeds, mean probabilities | 518/675 | 76.74% | 75.75% |
| Original Blender8 CE checkpoints | 5 seeds, mean probabilities | 569/675 | 84.30% | 84.10% |
| Matched low-LR CE control | Same 5 seeds and fine-tuning schedule, lambda=0 | 565/675 | 83.70% | 83.53% |
| **Matched low-LR consistency** | **Same 5 seeds and schedule, lambda=0.3** | **576/675** | **85.33%** | **85.19%** |

The clean causal comparison is consistency versus the matched low-LR CE control:
+1.63 pp, 17 consistency-only correct cases versus 6 control-only correct cases, exact
McNemar p=0.0347, and paired-sample bootstrap 95% CI [0.30, 2.96] pp. Compared with the
original CE checkpoint ensemble, consistency is +1.04 pp, but that separate comparison
is not significant (p=0.167; bootstrap CI [-0.15, 2.37] pp).

Therefore the five-model lambda=0.3 consistency ensemble is the best observed deployable
system and the matched-control comparison is the strongest evidence for a consistency
benefit. Inference still uses one normalized skeleton per sample; no generated test views,
template fitting, oracle labels, or DrOh-based checkpoint selection are used.

For the end-to-end contribution and webcam comparison, use the equal-size raw baseline
versus final system: 76.74% to 85.33% (+8.59 pp). The final model uniquely corrects 78
samples while the baseline uniquely corrects 20; exact McNemar p=2.87e-9 and bootstrap
95% CI for the gain is [5.78, 11.41] pp. This comparison measures the combined effect of
Blender8 training plus the final consistency schedule. Use the 83.70% versus 85.33%
matched-control comparison when isolating the contribution of consistency alone.

On the 983-sample Salux internal test, the corresponding five-model ensembles obtain
97.25% (original CE), 97.15% (matched control), and 97.25% (lambda=0.3 consistency).
Thus the external DrOh gain does not come with a measurable loss of internal accuracy.

Per-class DrOh accuracy for the clean matched comparison:

| Class | Samples | Low-LR CE control | Lambda=0.3 consistency | Change |
|---|---:|---:|---:|---:|
| OK | 106 | 90.57% | 90.57% | +0.00 pp |
| Paper | 123 | 85.37% | 86.99% | +1.63 pp |
| Rock | 109 | 81.65% | 77.98% | -3.67 pp |
| Scissors | 112 | 64.29% | 73.21% | +8.93 pp |
| The-Finger | 103 | 88.35% | 90.29% | +1.94 pp |
| Thumb | 122 | 91.80% | 92.62% | +0.82 pp |

The largest benefit is on Scissors; Rock remains the main regression to discuss rather
than hiding behind the aggregate score.

## 5. Full normalized Blender8 consistency lambda sweep

| lambda | Salux all-view validation | DrOh accuracy mean +/- SD | DrOh ensemble |
|---:|---:|---:|---:|
| 0 | 96.38% | 81.33 +/- 1.82% | **84.00%** |
| 0.01 | 96.59% | 80.40 +/- 1.34% | 81.19% |
| 0.03 | 96.75% | 78.91 +/- 1.85% | 80.89% |
| 0.05 | 96.58% | 81.23 +/- 1.44% | 83.41% |
| 0.10 | 96.92% | **81.98 +/- 2.74%** | 81.63% |
| 0.20 | 96.76% | 77.88 +/- 2.52% | 79.26% |
| 0.30 | **97.02%** | 78.81 +/- 1.86% | 79.70% |
| 0.50 | 96.90% | 77.48 +/- 1.12% | 80.59% |
| 0.70 | 96.84% | 79.90 +/- 2.69% | 81.63% |
| 0.90 | 96.73% | 79.90 +/- 1.90% | 80.44% |
| 1.00 | 96.75% | 80.10 +/- 1.85% | 82.81% |

Lambda=0.3 is selected by the held-out Salux anchor + eight-view validation metric and
matches the submitted thesis setting. Its ensemble improves over the baseline ensemble
by +3.26 pp (79.70% versus 76.44%), exact McNemar p=0.0352, bootstrap CI [0.44, 6.22]
pp. Lambda=0.1 is the descriptive maximum mean DrOh score but cannot be selected using
DrOh external test performance.

## 6. Number-of-view, architecture, and normalization ablations

### Number of views, CE-only GRU

| Training views | DrOh accuracy mean +/- SD |
|---:|---:|
| 0 | 75.41 +/- 4.23% |
| 2 | 77.23 +/- 2.89% |
| 4 | 77.48 +/- 1.41% |
| 8 | **81.33 +/- 1.82%** |

### Architecture

| Architecture | Raw baseline | Blender8 CE-only | Blender8 lambda=0.3 |
|---|---:|---:|---:|
| GRU | 75.41 +/- 4.23% | **81.33 +/- 1.82%** | 78.81 +/- 1.86% |
| MLP | 75.01 +/- 1.74% | 80.35 +/- 1.37% | 80.15 +/- 1.21% |

### Normalization

| Normalization | Raw baseline | Blender8 CE-only | Blender8 lambda=0.3 |
|---|---:|---:|---:|
| Wrist-middle scale | 75.41 +/- 4.23% | **81.33 +/- 1.82%** | 78.81 +/- 1.86% |
| Palm-centroid robust scale | 77.58 +/- 3.38% | 75.95 +/- 1.70% | 79.75 +/- 3.33% |

These ablations show that eight views are needed for the strongest tested coverage, the
effect is not exclusive to GRU, and normalization interacts with the consistency loss.

## 7. Refinement, fitted-domain, and label-free analyses

| Experiment | Test domain | Accuracy | Interpretation |
|---|---|---:|---|
| Fitted baseline | Salux fitted | 99.90% | Easy standardized source/fitted domain |
| Fitted + Blender8 | DrOh oracle-fitted | 89.33% | Oracle diagnostic |
| Fitted lambda=1.0 | DrOh oracle-fitted | 98.96% | Oracle diagnostic; not deployable |
| Minimum fitting-cost template selection | DrOh raw, label-free selection | 67.41% | Exploratory |
| Best explored cost/probability fusion | DrOh raw | 74.37% | Exploratory, below deployable raw MV |

The 89.33% and 98.96% oracle results are high because the ground-truth gesture class is
used to choose the fitting template. They must never be compared directly with deployable
single-view DrOh recognition.

## 8. Controlled mechanism analysis

| Representation/training | DrOh accuracy mean +/- SD | Purpose |
|---|---:|---|
| Camera-coordinate single-view | 31.70 +/- 1.57% | Viewpoint retained, no MV |
| Camera-coordinate + Blender8 | **52.05 +/- 2.63%** | Isolated learned rotation robustness |
| Canonical single-view | 71.06 +/- 3.82% | Deterministic orientation removal |
| Canonical + Blender8 before preprocessing | 72.79 +/- 3.23% | MV after ideal canonicalization adds little |

On synthetic unseen Salux angles, camera-coordinate single-view obtains 13.90-35.61%,
while camera-coordinate Blender8 obtains 91.56-96.34%. This is controlled skeleton-space
rotation evidence, not real RGB occlusion evidence.

## 9. Why the highlighted results are higher

1. Blender8 supervised CE expands each Salux pose into camera-space variants, reducing
   dependence on one orientation and improving cross-dataset generalization.
2. Training across seeds shows that MV also reduces optimization variance relative to the
   raw baseline.
3. Direct consistency from epoch 1 can over-regularize embeddings before the classifier
   learns class boundaries. CE warm-up first learns discriminative structure; one or two
   low-learning-rate consistency epochs then align views without destroying that structure.
4. The final five-seed matched experiment separates consistency from the effect of extra
   low-LR fine-tuning. The lambda=0.3 ensemble corrects 17 samples missed by its matched
   lambda=0 control while losing 6, yielding 576/675.
5. Extremely high fitted/oracle results are caused by standardized geometry and
   ground-truth-template information, not by a deployable hand-in-the-wild recognizer.

## 10. Recommended defense-slide headline

Submitted thesis result:

`Raw baseline 77.93% -> Blender8 consistency (lambda=0.3) 78.96%`

Post-submission three-seed ablation:

`Raw baseline 75.41 +/- 4.23% -> Blender8 CE-only 81.33 +/- 1.82%`

Final matched five-seed validation:

`Low-LR CE control 81.24 +/- 1.86% -> lambda=0.3 consistency 83.44 +/- 1.97%`

Final post-submission system:

`Best observed single-view system: five-seed lambda=0.3 consistency ensemble, 85.33%
(576/675), macro-F1 85.19%`

Fair end-to-end demo comparison:

`Five-seed raw baseline 76.74% -> five-seed Blender8 + consistency 85.33% (+8.59 pp)`

For the consistency claim, emphasize the matched five-seed control (+1.63 pp ensemble,
p=0.0347), while also reporting that the seed-level paired test is not significant with
only five seeds (p=0.126). Do not present oracle-fitted results as deployable accuracy.
