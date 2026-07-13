from __future__ import annotations

import argparse
from time import perf_counter

import cv2

from inference_core import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    PROJECT_ROOT,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
    draw_comparison_result,
    draw_multiview_stress_result,
    evaluate_virtual_views,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", default=None, help="Optional path for recording the annotated demo.")
    args = parser.parse_args()

    result_root = PROJECT_ROOT / "model/results/defense_experiments_6cls"
    baseline = EnsembleGestureRecognizer(
        checkpoints=[
            result_root / f"gru_wrist_middle_views0_lambda0p0_seed{seed}/best.pt"
            for seed in [0, 1, 42]
        ],
        smoothing_window=7,
    )
    proposed = EnsembleGestureRecognizer(
        checkpoints=[
            result_root / f"gru_wrist_middle_views8_lambda0p0_seed{seed}/best.pt"
            for seed in [0, 1, 42]
        ],
        smoothing_window=7,
    )
    extractor = MediaPipeHandExtractor(static_image_mode=False)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    writer = None
    ground_truth = CLASS_NAMES[0]
    frozen_frame = None
    stress_result = None
    last_time = perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            extraction = extractor.process_bgr(frame)
            baseline_prediction = None
            proposed_prediction = None
            if extraction is not None:
                baseline_prediction = baseline.predict_skeleton(extraction["skeleton"], smooth=True)
                proposed_prediction = proposed.predict_skeleton(extraction["skeleton"], smooth=True)
            else:
                baseline.reset()
                proposed.reset()
            if stress_result is None:
                display = draw_comparison_result(frame, extraction, baseline_prediction, proposed_prediction)
                cv2.putText(display, f"GT [{CLASS_NAMES.index(ground_truth) + 1}]: {DISPLAY_NAMES[ground_truth]} | SPACE: test 8 views", (18, display.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (90, 235, 130), 2, cv2.LINE_AA)
            else:
                display = draw_multiview_stress_result(
                    frozen_frame, stress_result["extraction"], ground_truth,
                    stress_result["baseline"], stress_result["proposed"],
                )

            now = perf_counter()
            fps = 1.0 / max(now - last_time, 1e-8)
            last_time = now
            cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 145, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)

            if args.record and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.record, fourcc, 20.0, (display.shape[1], display.shape[0]))
            if writer is not None:
                writer.write(display)

            cv2.imshow("Hand Gesture Recognition - press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                ground_truth = CLASS_NAMES[key - ord("1")]
                stress_result = None
            elif key == ord("r"):
                stress_result = None
                baseline.reset(); proposed.reset()
            elif key == ord(" ") and extraction is not None:
                frozen_frame = frame.copy()
                stress_result = {
                    "extraction": extraction,
                    "baseline": evaluate_virtual_views(baseline, extraction["skeleton"], ground_truth),
                    "proposed": evaluate_virtual_views(proposed, extraction["skeleton"], ground_truth),
                }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
