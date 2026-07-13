"""Live webcam deployment of the final five-seed consistency ensemble.

The deployment path is MediaPipe -> canonical six-class skeleton -> probability
ensemble. Blender8 views are training-only; no synthetic view or fitting is used here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2

from inference_core import (
    FINAL_CONSISTENCY_5_CHECKPOINTS,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
    draw_result,
)


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing final checkpoints:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", default=None, help="Optional annotated MP4 output path")
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    require_files(FINAL_CONSISTENCY_5_CHECKPOINTS)
    recognizer = EnsembleGestureRecognizer(
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
            prediction = None
            if extraction is None:
                recognizer.reset()
            else:
                prediction = recognizer.predict_skeleton(extraction["skeleton"], smooth=True)

            display = draw_result(
                frame,
                extraction,
                prediction,
                confidence_threshold=args.confidence_threshold,
            )
            now = perf_counter()
            fps = 1.0 / max(now - last_time, 1e-8)
            last_time = now
            height, width = display.shape[:2]
            cv2.putText(display, "FINAL: Blender8 + consistency lambda=0.3 | 5-seed ensemble",
                        (18, height - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 235, 130), 2, cv2.LINE_AA)
            cv2.putText(display, "Single skeleton inference | DrOh 85.33% (576/675) | Q: quit",
                        (18, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(display, f"FPS {fps:.1f}", (max(18, width - 125), height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA)

            if args.record and writer is None:
                writer = cv2.VideoWriter(
                    args.record,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (width, height),
                )
            if writer is not None:
                writer.write(display)

            cv2.imshow("Final Hand Gesture Recognizer - press Q to quit", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
