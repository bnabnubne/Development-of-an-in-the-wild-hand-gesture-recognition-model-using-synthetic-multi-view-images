# Six-class hand gesture recognition demo

The demo compares two deployable single-view protocols on the same webcam skeleton:

1. Raw three-seed GRU ensemble (DrOh accuracy 76.44%).
2. Eight-view CE-only three-seed GRU ensemble (DrOh accuracy 84.00%).

Pipeline:

`webcam -> MediaPipe Hands -> skeleton_final_v2 preprocessing -> baseline and proposed ensembles -> predictions`

## Live webcam demo

```bash
python model/demo/webcam_demo.py
```

Press `Q` or `Esc` to close. To record a backup video for the defense:

```bash
python model/demo/webcam_demo.py --record defense_demo.mp4
```

The webcam demo mirrors frames by default so MediaPipe handedness follows its
selfie-camera assumption. Use `--no-mirror` for a non-mirrored external camera.

## Showing the multiview benefit

Use keys `1`-`6` to select the true gesture shown on screen, then press `Space`.
The demo freezes that one observed skeleton, reproduces the paper's eight Blender
camera angles, and compares both models at every view. Green cells are correct;
red cells are incorrect. `R` returns to live mode.

Key mapping: `1` OK, `2` Paper, `3` Rock, `4` Scissors, `5` The-Finger, `6` Thumb.

## Continuous live comparison (recommended for the defense)

```bash
python model/demo/webcam_multiview_live_demo.py
```

This version needs no snapshot. It continuously evaluates the current webcam
skeleton under the same eight Blender cameras and updates both model rows live.
The original snapshot-based version remains in `webcam_demo.py`.

## Fair real-viewpoint comparison (main defense demo)

```bash
python model/demo/webcam_real_viewpoint_demo.py
```

This is the recommended scientific comparison. It does not generate synthetic
views during inference. Physically rotate the hand: both models receive the same
live skeleton through the same DrOh preprocessing, while the interface records
correctness in six observed palm-yaw regions. Press `R` before each gesture.

## Thesis-defense visual showcase (recommended presentation)

```bash
python model/demo/defense_showcase_demo.py
```

`L` opens the live end-to-end webcam. `E` opens a deterministic replay of real
DrOh test samples where the raw baseline is wrong and Blender8 MV is correct.
The replay shows the RGB sample, raw landmarks, canonical model input, both
predictions, and locked aggregate test metrics. Arrow keys browse; `A` autoplays.
