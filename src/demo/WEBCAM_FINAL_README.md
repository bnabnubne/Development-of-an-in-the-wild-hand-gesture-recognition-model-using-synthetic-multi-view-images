# Final Webcam Demos

## Models

- Final: five-seed probability ensemble, Blender8 CE warm-up followed by lambda=0.3
  consistency fine-tuning.
- The classifier-level comparison uses a five-seed raw single-view baseline; both sides
  receive the same canonical skeleton.
- The full-system comparison uses a naive camera-coordinate baseline on one side and the
  complete canonical + Blender8 + consistency system on the other.
- No synthetic views, template fitting, oracle label, or refinement is used at inference.

All current displayed metrics must use the post-filter 605-sample DrOh snapshot. Older
675-row numbers are historical and include removed Scissors samples.

## Run final model only

From the project root:

```bash
python model/demo/webcam_final_5seed_demo.py
```

## Run final versus baseline

```bash
python model/demo/webcam_final_vs_baseline_demo.py
```

This comparison defaults to no temporal smoothing for either model, making transient
viewpoint errors visible without changing either model's input or threshold.

For the clearest honest defense sequence, use the classes with the largest locked DrOh
gain from raw baseline to final: `4` Scissors (+27.68 pp), `3` Rock (+19.27 pp), then `5`
The-Finger (+10.68 pp). Rotate and foreshorten the hand slowly while keeping all fingers
visible to MediaPipe. The challenge panel reports rolling 90-frame correctness, rescue
frames, reverse frames, and highlights each live baseline-wrong/final-correct event.

## Full pipeline versus naive end-to-end baseline

```bash
python model/demo/webcam_full_pipeline_vs_naive_demo.py
```

This is the visually strongest demo and intentionally compares two complete systems:

- Naive baseline: RGB -> MediaPipe -> handedness/center/scale -> camera-coordinate
  single-view GRU.
- Full system: RGB -> MediaPipe -> palm canonicalization -> Blender8 + lambda=0.3
  consistency five-model ensemble.

On the post-filter 605-sample DrOh snapshot, the locked comparison is 35.21% versus
87.77% (+52.56 pp). Label it **system-level improvement**, not a multiview-only gain,
because preprocessing and training objectives also differ.

Controls:

- `1` OK
- `2` Paper
- `3` Rock
- `4` Scissors
- `5` The-Finger
- `6` Thumb
- `R` reset rolling live statistics while keeping the selected ground truth
- `Q` or `Esc` quit

Optional examples:

```bash
python model/demo/webcam_final_vs_baseline_demo.py --camera 1
python model/demo/webcam_final_vs_baseline_demo.py --record defense_comparison.mp4
python model/demo/webcam_full_pipeline_vs_naive_demo.py --record defense_full_system.mp4
python model/demo/webcam_final_5seed_demo.py --smoothing-window 5
```

On macOS, run from a normal graphical Terminal or the IDE terminal and grant Camera
permission when prompted. A headless shell cannot create the OpenGL context required by
the installed MediaPipe build.

The live comparison intentionally uses an equal-size five-seed baseline. Do not replace
it with the weakest individual seed: that would make the visual gap larger but would no
longer be a defensible comparison.
