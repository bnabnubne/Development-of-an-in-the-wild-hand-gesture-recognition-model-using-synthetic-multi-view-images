# Unified thesis-defense results

## 1. Fixed project direction

The submitted project direction remains unchanged:

`single RGB image -> MediaPipe 3D skeleton -> geometric normalization -> single-view GRU inference`

Synthetic Blender views are generated from Salux and used only during training. DrOh is
external test-only data. Template fitting/refinement is a geometric/oracle analysis and is
not part of the deployable raw-domain recognition claim.

In the submitted thesis, **raw skeleton** means a detector-derived skeleton that has not
been template-fitted. It still includes `skeleton_final_v2` geometric normalization. It
must not be confused with the newly named **camera-coordinate skeleton**, which is the
MediaPipe skeleton before palm-axis orientation alignment.

## 2. Submitted thesis results (do not replace)

Main setting: GRU, seed 42, eight Blender views, consistency weight `lambda=0.3`.

| Method | Salux/test input | Salux (%) | DrOh (%) |
|---|---|---:|---:|
| Raw baseline | normalized raw | 96.85 | 77.93 |
| Raw + Blender8 MV consistency | normalized raw | 97.05 | 78.96 |

The submitted claim is a moderate **+1.03 percentage-point** improvement while retaining
single-view inference.

Template/refinement results in the submitted thesis:

| Training protocol | Test input | Accuracy (%) |
|---|---|---:|
| Fitted baseline | Salux fitted | 99.90 |
| Fitted baseline | Salux raw | 18.92 |
| Fitted baseline | DrOh raw | 20.74 |
| Fitted baseline | DrOh fitted oracle | 81.78 |
| Fitted + Blender8 MV, lambda=0.3 | Salux fitted | 99.90 |
| Fitted + Blender8 MV, lambda=0.3 | Salux raw | 47.41 |
| Fitted + Blender8 MV, lambda=0.3 | DrOh raw | 36.15 |
| Fitted + Blender8 MV, lambda=0.3 | DrOh fitted oracle | 89.33 |

Fitted-domain lambda sweep:

| lambda | DrOh raw (%) | DrOh fitted oracle (%) |
|---:|---:|---:|
| 0.05 | 32.15 | 92.15 |
| 0.10 | 29.04 | 89.19 |
| 0.30 | 36.15 | 89.33 |
| 0.50 | 35.56 | 89.19 |
| 0.70 | 40.00 | 90.81 |
| 0.90 | 39.70 | 92.59 |
| 1.00 | 39.56 | 98.96 |

Label-free template selection remains exploratory: minimum-cost selection 67.41%; best
explored cost/probability fusion 74.37%. DrOh fitted results use the ground-truth class
template and must always be labelled **oracle diagnostic**.

## 3. Post-submission validation under the original normalized-skeleton direction

These experiments keep the report's normalized skeleton representation and add three
seeds (`0, 1, 42`) and component ablations.

| Training protocol | DrOh accuracy, mean +/- std | Macro-F1, mean +/- std |
|---|---:|---:|
| Normalized raw baseline | 75.41 +/- 4.23 | 74.51 +/- 4.49 |
| Blender8 MV, CE-only | **81.33 +/- 1.82** | **81.00 +/- 1.87** |
| Blender8 MV + consistency, lambda=0.3 | 78.81 +/- 1.86 | 78.10 +/- 2.33 |

This confirms the old direction while refining the interpretation: most of the gain comes
from supervised synthetic-view augmentation; the cosine consistency term at `lambda=0.3`
is not consistently beneficial.

Fixed three-seed probability ensembles (still one skeleton at inference):

| Ensemble | Salux (%) | DrOh (%) | DrOh macro-F1 (%) |
|---|---:|---:|---:|
| Normalized raw baseline | 97.15 | 76.44 | 75.44 |
| Blender8 MV, CE-only | 97.25 | **84.00** | **83.83** |
| Blender8 MV + consistency | 97.15 | 79.70 | 78.89 |

Baseline ensemble versus CE-only MV ensemble: +7.56 pp; exact McNemar p=1.37e-6;
paired bootstrap 95% CI [4.59, 10.52] pp. The 84.00% result is post-submission and must
not be presented as a value already contained in the thesis.

Number-of-view ablation, CE-only GRU:

| Training views | DrOh accuracy (%) |
|---:|---:|
| 0 | 75.41 +/- 4.23 |
| 2 | 77.23 +/- 2.89 |
| 4 | 77.48 +/- 1.41 |
| 8 | **81.33 +/- 1.82** |

## 4. Additional controlled mechanism analysis

This is a supplementary 2x2 experiment, not a replacement for the submitted table. It
uses equal optimizer-step budgets, CE-only training, separated shuffle/view RNGs, and
generates each view before the selected orientation preprocessing.

| Representation/training | Salux (%) | DrOh (%) | DrOh macro-F1 (%) |
|---|---:|---:|---:|
| Camera-coordinate single-view | 97.42 +/- 0.50 | 31.70 +/- 1.57 | 26.90 +/- 2.05 |
| Camera-coordinate + Blender8 | 96.17 +/- 1.03 | **52.05 +/- 2.63** | **50.14 +/- 3.47** |
| Canonical single-view | 97.15 +/- 0.18 | 71.06 +/- 3.82 | 70.28 +/- 4.08 |
| Canonical + Blender8 | 96.91 +/- 0.48 | 72.79 +/- 3.23 | 71.93 +/- 3.37 |

Paired ensemble analysis:

- Camera-coordinate MV versus single: +21.04 pp, p=2.58e-27, bootstrap CI
  [17.33, 24.74] pp.
- Canonical MV versus single: +1.48 pp, p=0.203, bootstrap CI [-0.59, 3.56] pp.

Controlled Salux unseen-angle accuracy:

| Representation/training | -30 deg | 15 deg | 75 deg | 105 deg | 150 deg |
|---|---:|---:|---:|---:|---:|
| Camera-coordinate single | 32.99 | 35.61 | 15.90 | 13.90 | 21.13 |
| Camera-coordinate + Blender8 | **95.96** | **96.34** | **96.30** | **96.17** | **91.56** |
| Canonical single | 97.15 | 97.15 | 97.15 | 97.15 | 97.15 |
| Canonical + Blender8 | 96.91 | 96.91 | 96.91 | 96.91 | 96.91 |

Interpretation: Blender8 clearly learns rigid-rotation robustness when camera orientation
is retained. Palm-axis canonicalization deterministically removes most rigid rotation;
therefore MV offers no statistically significant additional gain after ideal
canonicalization. This does not invalidate the submitted augmentation result; it clarifies
that the submitted pipeline combines deterministic geometric normalization with learned
augmentation/regularization.

The unseen-angle test is skeleton-space evidence. It does not simulate RGB occlusion,
missing landmarks, or MediaPipe depth errors.

## 5. Recommended slide conclusions

1. The submitted result remains: Blender8 consistency improves DrOh from 77.93% to
   78.96% under the original seed-42 protocol.
2. Post-submission multi-seed analysis confirms the same direction and finds that
   CE-only Blender8 is stronger and more stable: 75.41% to 81.33% on average.
3. The fixed three-seed ensemble reaches 84.00% and is an additional defense system.
4. Controlled analysis shows that canonicalization and multiview are complementary
   mechanisms: canonicalization removes global orientation deterministically, while
   multiview teaches rotation robustness when orientation remains and regularizes the
   normalized-skeleton model.
5. Fitting/refinement produces regular geometry, but fitted-to-raw domain mismatch and
   oracle template selection prevent it from being the main deployable claim.

## 6. Do not use as main claims

- Invalid shared-RNG pilot runs.
- Virtual-camera webcam stress-test scores as real-world accuracy.
- DrOh oracle-fitted accuracy without the word **oracle**.
- Bridge experiments and unfinished label-free variants as final methods.
- 84.00% as if it appeared in the submitted PDF.

