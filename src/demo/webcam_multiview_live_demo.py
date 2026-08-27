
from __future__ import annotations

import argparse
from time import perf_counter

import cv2
import numpy as np

from inference_core import (
    CAMERA_ANGLES_DEG,
    CLASS_NAMES,
    CONNECTIONS,
    DISPLAY_NAMES,
    PROJECT_ROOT,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
    evaluate_virtual_views,
)


def draw_live_panel(frame, extraction, ground_truth, baseline_eval, proposed_eval, fps):
    height, width = frame.shape[:2]
    panel_width = max(470, int(width * 0.48))
    camera_width = width - panel_width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    camera = cv2.resize(frame, (camera_width, height))
    output[:, :camera_width] = camera

    if extraction is not None:
        points = extraction["raw_landmarks"]
        pixels = np.column_stack((points[:, 0] * camera_width, points[:, 1] * height)).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(output, tuple(pixels[a]), tuple(pixels[b]), (55, 215, 255), 3, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(output, (int(x), int(y)), 5, (35, 70, 255), -1, cv2.LINE_AA)

    x0 = camera_width
    cv2.rectangle(output, (x0, 0), (width, height), (13, 16, 22), -1)
    cv2.putText(output, "LIVE VIEWPOINT ROBUSTNESS", (x0 + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, f"Ground truth [{CLASS_NAMES.index(ground_truth)+1}]: {DISPLAY_NAMES[ground_truth]}", (x0 + 20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (205, 215, 225), 1, cv2.LINE_AA)

    if baseline_eval is None or proposed_eval is None:
        cv2.putText(output, "Show one full hand to the camera", (x0 + 20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 190, 255), 2, cv2.LINE_AA)
    else:
        available = panel_width - 40
        cell_width = available // 8

        def model_row(y, title, evaluation, accent):
            cv2.putText(output, title, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 2, cv2.LINE_AA)
            score = evaluation["correct"]
            majority = DISPLAY_NAMES[evaluation["majority"]]
            cv2.putText(output, f"{score}/8 correct | majority: {majority} | agreement: {evaluation['agreement']}/8", (x0 + 20, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA)
            for index, (angle, result) in enumerate(zip(CAMERA_ANGLES_DEG, evaluation["results"])):
                left = x0 + 20 + index * cell_width
                right = left + cell_width - 5
                good = result["class"] == ground_truth
                color = (65, 220, 105) if good else (75, 85, 245)
                cv2.rectangle(output, (left, y + 40), (right, y + 102), color, 2)
                cv2.putText(output, f"{angle:+d}", (left + 4, y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (205, 205, 205), 1, cv2.LINE_AA)
                name = DISPLAY_NAMES[result["class"]]
                cv2.putText(output, name[:5], (left + 4, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)
                cv2.putText(output, f"{result['confidence']*100:.0f}%", (left + 4, y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (205, 205, 205), 1, cv2.LINE_AA)

        model_row(105, "BASELINE - trained on one raw view", baseline_eval, (120, 190, 255))
        model_row(250, "PROPOSED - trained with Blender8 views", proposed_eval, (90, 235, 130))

        delta = proposed_eval["correct"] - baseline_eval["correct"]
        if delta > 0:
            message, color = f"MULTIVIEW BENEFIT: +{delta}/8 correct views", (90, 235, 130)
        elif delta < 0:
            message, color = f"This frame: baseline ahead by {-delta}/8", (120, 190, 255)
        else:
            message, color = "This frame: equal viewpoint score", (215, 215, 215)
        cv2.putText(output, message, (x0 + 20, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.59, color, 2, cv2.LINE_AA)

    cv2.putText(output, "Rotate/move the hand continuously - no capture required", (x0 + 20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 195, 205), 1, cv2.LINE_AA)
    cv2.putText(output, "1-6: ground truth | Q: quit", (x0 + 20, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 195, 205), 1, cv2.LINE_AA)
    cv2.putText(output, f"FPS {fps:.1f}", (x0 + panel_width - 85, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (175, 180, 190), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", default=None)
    parser.add_argument("--eval-interval", type=float, default=0.12, help="Seconds between Blender8 evaluations")
    args = parser.parse_args()

    result_root = PROJECT_ROOT / "model/results/defense_experiments_6cls"
    baseline = EnsembleGestureRecognizer([
        result_root / f"gru_wrist_middle_views0_lambda0p0_seed{seed}/best.pt" for seed in [0, 1, 42]
    ], smoothing_window=1)
    proposed = EnsembleGestureRecognizer([
        result_root / f"gru_wrist_middle_views8_lambda0p0_seed{seed}/best.pt" for seed in [0, 1, 42]
    ], smoothing_window=1)
    extractor = MediaPipeHandExtractor(static_image_mode=False)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    ground_truth = CLASS_NAMES[0]
    baseline_eval = proposed_eval = None
    last_eval = 0.0
    last_frame_time = perf_counter()
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            extraction = extractor.process_bgr(frame)
            now = perf_counter()
            if extraction is not None and now - last_eval >= args.eval_interval:
                baseline_eval = evaluate_virtual_views(baseline, extraction["skeleton"], ground_truth)
                proposed_eval = evaluate_virtual_views(proposed, extraction["skeleton"], ground_truth)
                last_eval = now
            elif extraction is None:
                baseline_eval = proposed_eval = None
            fps = 1.0 / max(now - last_frame_time, 1e-8)
            last_frame_time = now
            display = draw_live_panel(frame, extraction, ground_truth, baseline_eval, proposed_eval, fps)

            if args.record and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.record, fourcc, 20.0, (display.shape[1], display.shape[0]))
            if writer is not None:
                writer.write(display)

            cv2.imshow("Live Multiview Benefit - press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                ground_truth = CLASS_NAMES[key - ord("1")]
                last_eval = 0.0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
