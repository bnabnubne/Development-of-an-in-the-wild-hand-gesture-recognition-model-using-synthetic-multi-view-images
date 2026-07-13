"""Fair live comparison: five-seed raw baseline versus five-seed final model.

Both systems receive the same single MediaPipe skeleton after identical preprocessing.
The baseline is the weakest legitimate method (no multiview training), not a selected
bad seed. The proposed system uses Blender8 training and lambda=0.3 consistency.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2

from inference_core import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    FINAL_CONSISTENCY_5_CHECKPOINTS,
    PROJECT_ROOT,
    RAW_BASELINE_5_CHECKPOINTS,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
    draw_comparison_result,
)


METRICS_PATH = PROJECT_ROOT / "model/results/webcam_final_comparison_6cls/metrics.json"


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing comparison checkpoints:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", default=None, help="Optional annotated MP4 output path")
    parser.add_argument(
        "--smoothing-window", type=int, default=1,
        help="Use 1 for the fair per-frame robustness challenge; increase for normal deployment.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    require_files(RAW_BASELINE_5_CHECKPOINTS + FINAL_CONSISTENCY_5_CHECKPOINTS)
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {METRICS_PATH}. Run model/prepare_webcam_final_comparison_6cls.py first."
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    baseline_accuracy = 100 * metrics["baseline"]["accuracy"]
    final_accuracy = 100 * metrics["final"]["accuracy"]

    baseline = EnsembleGestureRecognizer(
        RAW_BASELINE_5_CHECKPOINTS,
        smoothing_window=args.smoothing_window,
    )
    final = EnsembleGestureRecognizer(
        FINAL_CONSISTENCY_5_CHECKPOINTS,
        smoothing_window=args.smoothing_window,
    )
    extractor = MediaPipeHandExtractor(static_image_mode=False)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ground_truth = None
    baseline_history = deque(maxlen=90)
    final_history = deque(maxlen=90)
    live_rescues = 0
    live_reverses = 0
    writer = None
    last_time = perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            extraction = extractor.process_bgr(frame)
            baseline_prediction = final_prediction = None
            if extraction is None:
                baseline.reset()
                final.reset()
            else:
                skeleton = extraction["skeleton"]
                baseline_prediction = baseline.predict_skeleton(skeleton, smooth=True)
                final_prediction = final.predict_skeleton(skeleton, smooth=True)
                if ground_truth is not None:
                    baseline_ok = baseline_prediction["class"] == ground_truth
                    final_ok = final_prediction["class"] == ground_truth
                    baseline_history.append(baseline_ok)
                    final_history.append(final_ok)
                    live_rescues += int((not baseline_ok) and final_ok)
                    live_reverses += int(baseline_ok and (not final_ok))

            display = draw_comparison_result(
                frame,
                extraction,
                baseline_prediction,
                final_prediction,
                confidence_threshold=args.confidence_threshold,
                baseline_title="RAW SINGLE-VIEW BASELINE (5 seeds)",
                proposed_title="BLENDER8 + CONSISTENCY (5 seeds)",
                baseline_accuracy=baseline_accuracy,
                proposed_accuracy=final_accuracy,
            )
            now = perf_counter()
            fps = 1.0 / max(now - last_time, 1e-8)
            last_time = now
            height, width = display.shape[:2]

            footer = display.copy()
            cv2.rectangle(footer, (8, height - 150), (min(width - 8, 950), height - 8), (8, 10, 14), -1)
            display = cv2.addWeighted(footer, 0.80, display, 0.20, 0)
            cv2.putText(display, "FAIR LIVE CHALLENGE: same input, preprocessing, threshold and 5-model ensemble size",
                        (18, height - 126), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)
            if ground_truth is None:
                gt_text = "Press 1-6 to set GT | rotate/foreshorten hand slowly | R: reset | Q: quit"
                gt_color = (90, 205, 255)
            elif baseline_prediction is None:
                gt_text = f"Ground truth: {DISPLAY_NAMES[ground_truth]} | no hand detected"
                gt_color = (90, 205, 255)
            else:
                baseline_ok = baseline_prediction["class"] == ground_truth
                final_ok = final_prediction["class"] == ground_truth
                gt_text = (
                    f"GT: {DISPLAY_NAMES[ground_truth]} | "
                    f"Baseline: {'CORRECT' if baseline_ok else 'WRONG'} | "
                    f"Final: {'CORRECT' if final_ok else 'WRONG'}"
                )
                gt_color = (90, 235, 130) if final_ok and not baseline_ok else (235, 235, 235)
            cv2.putText(display, gt_text, (18, height - 98), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, gt_color, 2, cv2.LINE_AA)
            if baseline_history:
                baseline_rate = 100 * sum(baseline_history) / len(baseline_history)
                final_rate = 100 * sum(final_history) / len(final_history)
                rolling_text = (
                    f"Rolling {len(baseline_history)}/90 frames: baseline {baseline_rate:.1f}% | "
                    f"final {final_rate:.1f}% | rescue {live_rescues} | reverse {live_reverses}"
                )
                rolling_color = (90, 235, 130) if final_rate > baseline_rate else (225, 225, 225)
            else:
                rolling_text = "Rolling robustness starts after ground truth is selected"
                rolling_color = (195, 200, 210)
            cv2.putText(display, rolling_text, (18, height - 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, rolling_color, 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Locked DrOh: {baseline_accuracy:.2f}% -> {final_accuracy:.2f}% "
                        f"({metrics['difference_pp']:+.2f} pp) | FPS {fps:.1f}",
                        (18, height - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (90, 235, 130), 2, cv2.LINE_AA)
            if (
                ground_truth is not None and baseline_prediction is not None
                and baseline_prediction["class"] != ground_truth
                and final_prediction["class"] == ground_truth
            ):
                banner = "LIVE RESCUE: BASELINE WRONG -> FINAL CORRECT"
                cv2.rectangle(display, (max(8, width // 2 - 330), 280),
                              (min(width - 8, width // 2 + 330), 332), (25, 105, 45), -1)
                cv2.putText(display, banner, (max(18, width // 2 - 305), 315),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (105, 255, 145), 2, cv2.LINE_AA)

            if args.record and writer is None:
                writer = cv2.VideoWriter(
                    args.record,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (width, height),
                )
            if writer is not None:
                writer.write(display)

            cv2.imshow("Final vs Raw Baseline - press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                ground_truth = CLASS_NAMES[key - ord("1")]
                baseline.reset()
                final.reset()
                baseline_history.clear(); final_history.clear()
                live_rescues = live_reverses = 0
            elif key == ord("r"):
                baseline.reset()
                final.reset()
                baseline_history.clear(); final_history.clear()
                live_rescues = live_reverses = 0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
