
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd

from inference_core import (
    CLASS_NAMES,
    CONNECTIONS,
    DISPLAY_NAMES,
    PROJECT_ROOT,
    FINAL_CONSISTENCY_5_CHECKPOINTS,
    RAW_BASELINE_5_CHECKPOINTS,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
    draw_comparison_result,
)


APRIL_ROOT = Path(".")
PRED_ROOT = PROJECT_ROOT / "model/results/webcam_final_comparison_6cls"
METRICS_PATH = PRED_ROOT / "metrics.json"


def palm_yaw(raw):
    across = raw[17] - raw[5]
    along = raw[9] - raw[0]
    normal = np.cross(across, along)
    normal /= np.linalg.norm(normal) + 1e-8
    return float(np.degrees(np.arctan2(abs(normal[0]), abs(normal[2]) + 1e-8)))


def load_evidence():
    baseline = pd.read_csv(PRED_ROOT / "predictions_raw_baseline_5seed_droh.csv")
    proposed = pd.read_csv(PRED_ROOT / "predictions_final_consistency_5seed_droh.csv")
    baseline = baseline.rename(columns={
        "predicted_class": "baseline_class", "confidence": "baseline_confidence",
        "correct": "baseline_correct",
    })
    proposed = proposed.rename(columns={
        "predicted_class": "proposed_class", "confidence": "proposed_confidence",
        "correct": "proposed_correct",
    })
    keep_b = ["action", "sample_id", "true_class", "baseline_class", "baseline_confidence", "baseline_correct"]
    keep_p = ["action", "sample_id", "proposed_class", "proposed_confidence", "proposed_correct"]
    frame = baseline[keep_b].merge(proposed[keep_p], on=["action", "sample_id"])

    media = pd.read_csv(APRIL_ROOT / "test/mediapipe_metadata.csv")
    media = media[media.status == "ok"].copy()
    media["sample_id"] = media.raw_path.map(lambda value: Path(value).stem)
    media = media.rename(columns={"class": "source_action"})
    frame = frame.merge(media[["sample_id", "source_action", "image_path", "raw_path"]], on="sample_id", how="left")
    manifest = pd.read_csv(APRIL_ROOT / "metadata/droh_baseline.csv")
    frame = frame.merge(manifest[["sample_id", "input_path"]], on="sample_id", how="left")
    frame["image_exists"] = frame.image_path.map(lambda value: isinstance(value, str) and Path(value).exists())
    frame["yaw"] = frame.raw_path.map(
        lambda path: palm_yaw(np.load(path).astype(np.float32))
        if isinstance(path, str) and Path(path).exists() else np.nan
    )
    frame["rescued"] = (~frame.baseline_correct) & frame.proposed_correct
    frame["regressed"] = frame.baseline_correct & (~frame.proposed_correct)
    return frame.sort_values(["rescued", "yaw", "proposed_confidence"], ascending=[False, False, False]).reset_index(drop=True)


def fit_image(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, scale, x, y


def draw_skeleton(canvas, points, rect, color=(55, 215, 255), raw_image_space=False):
    x0, y0, x1, y1 = rect
    points = np.asarray(points, dtype=np.float32)
    if raw_image_space:
        pixels = np.column_stack((x0 + points[:, 0] * (x1 - x0), y0 + points[:, 1] * (y1 - y0))).astype(int)
    else:
        xy = points[:, :2].copy()
        mins, maxs = xy.min(0), xy.max(0)
        span = np.maximum(maxs - mins, 1e-6)
        xy = (xy - mins) / span
        margin = 28
        pixels = np.column_stack((
            x0 + margin + xy[:, 0] * max(1, x1 - x0 - 2 * margin),
            y1 - margin - xy[:, 1] * max(1, y1 - y0 - 2 * margin),
        )).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), color, 3, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(canvas, tuple(point), 4, (35, 70, 255), -1, cv2.LINE_AA)


def draw_evidence(row, position, visual_total, rescued_total, regressed_total, metrics):
    width, height = 1280, 720
    output = np.full((height, width, 3), (12, 15, 21), dtype=np.uint8)
    image = cv2.imread(str(row.image_path))
    image_panel, scale, off_x, off_y = fit_image(image, 550, 570)
    output[95:665, :550] = image_panel

    raw = np.load(row.raw_path).astype(np.float32)
    px = raw.copy()
    px[:, 0] = (off_x + raw[:, 0] * image.shape[1] * scale) / 550.0
    px[:, 1] = (off_y + raw[:, 1] * image.shape[0] * scale) / 570.0
    draw_skeleton(output, px, (0, 95, 550, 665), raw_image_space=True)

    cv2.putText(output, "CURATED RESCUED CASE - LOCKED DrOh TEST", (22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, "Selected because baseline is wrong and final is correct", (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 205, 255), 1, cv2.LINE_AA)

    cv2.rectangle(output, (575, 95), (845, 390), (24, 29, 38), -1)
    cv2.putText(output, "CANONICAL MODEL INPUT", (590, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 225, 230), 1, cv2.LINE_AA)
    canonical = np.load(row.input_path).astype(np.float32)
    draw_skeleton(output, canonical, (585, 140, 835, 380), color=(90, 220, 235))

    cv2.putText(output, f"Ground truth: {DISPLAY_NAMES[row.true_class]}", (875, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, f"Raw palm inclination: {row.yaw:.1f} deg", (875, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 210, 220), 1, cv2.LINE_AA)

    def card(y, title, predicted, confidence, correct, accent):
        cv2.rectangle(output, (875, y), (1250, y + 110), (24, 29, 38), -1)
        cv2.putText(output, title, (895, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent, 2, cv2.LINE_AA)
        result_color = (70, 225, 105) if correct else (70, 85, 245)
        status = "CORRECT" if correct else "WRONG"
        cv2.putText(output, f"{DISPLAY_NAMES[predicted]}  {confidence*100:.1f}%", (895, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.64, result_color, 2, cv2.LINE_AA)
        cv2.putText(output, status, (895, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, result_color, 2, cv2.LINE_AA)

    card(190, "RAW BASELINE - 5 SEEDS", row.baseline_class, row.baseline_confidence, row.baseline_correct, (120, 190, 255))
    card(320, "BLENDER8 + CONSISTENCY - 5 SEEDS", row.proposed_class, row.proposed_confidence, row.proposed_correct, (90, 235, 130))

    cv2.rectangle(output, (575, 465), (1250, 615), (20, 25, 33), -1)
    cv2.putText(output, f"FINAL RESCUED {rescued_total} / 675 TEST SAMPLES", (600, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (90, 235, 130), 2, cv2.LINE_AA)
    cv2.putText(output, f"Reverse cases: {regressed_total} / 675", (600, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (195, 200, 210), 1, cv2.LINE_AA)
    baseline_accuracy = 100 * metrics["baseline"]["accuracy"]
    final_accuracy = 100 * metrics["final"]["accuracy"]
    cv2.putText(output, f"Overall: {baseline_accuracy:.2f}% -> {final_accuracy:.2f}%  ({metrics['difference_pp']:+.2f} pp)", (600, 574), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, f"Hard-angle visual case {position + 1}/{visual_total} | locked test-set replay", (600, 603), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (175, 185, 198), 1, cv2.LINE_AA)
    cv2.putText(output, "E: evidence  L: live webcam  Left/Right: sample  A: autoplay  Q: quit", (22, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 195, 205), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    baseline = EnsembleGestureRecognizer(RAW_BASELINE_5_CHECKPOINTS, 1)
    proposed = EnsembleGestureRecognizer(FINAL_CONSISTENCY_5_CHECKPOINTS, 1)
    extractor = MediaPipeHandExtractor(static_image_mode=False)
    evidence = load_evidence()
    rescued_total = int(evidence.rescued.sum())
    rescued = evidence[evidence.rescued & evidence.image_exists].reset_index(drop=True)
    regressed_total = int(evidence.regressed.sum())

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    mode, index, autoplay = "evidence", 0, False
    last_advance = perf_counter()
    writer = None
    try:
        while True:
            if mode == "live":
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Cannot read webcam frame")
                frame = cv2.flip(frame, 1)
                extraction = extractor.process_bgr(frame)
                bp = pp = None
                if extraction is not None:
                    bp = baseline.predict_skeleton(extraction["skeleton"], smooth=True)
                    pp = proposed.predict_skeleton(extraction["skeleton"], smooth=True)
                display = draw_comparison_result(
                    frame, extraction, bp, pp,
                    baseline_title="RAW SINGLE-VIEW BASELINE (5 seeds)",
                    proposed_title="BLENDER8 + CONSISTENCY (5 seeds)",
                    baseline_accuracy=100 * metrics["baseline"]["accuracy"],
                    proposed_accuracy=100 * metrics["final"]["accuracy"],
                )
                cv2.putText(display, "LIVE PER-FRAME (no smoothing) | E: curated DrOh rescued cases", (18, display.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 235, 130), 2, cv2.LINE_AA)
                display = cv2.resize(display, (1280, 720))
            else:
                if autoplay and perf_counter() - last_advance > 2.8:
                    index = (index + 1) % len(rescued)
                    last_advance = perf_counter()
                display = draw_evidence(rescued.iloc[index], index, len(rescued), rescued_total, regressed_total, metrics)

            if args.record and writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280, 720))
            if writer is not None:
                writer.write(display)
            cv2.imshow("Thesis Defense Showcase", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e"):
                mode = "evidence"
            elif key == ord("l"):
                mode = "live"; baseline.reset(); proposed.reset()
            elif key == ord("a"):
                autoplay = not autoplay; last_advance = perf_counter()
            elif key in (81, ord("[")):
                index = (index - 1) % len(rescued); autoplay = False
            elif key in (83, ord("]")):
                index = (index + 1) % len(rescued); autoplay = False
    finally:
        capture.release()
        extractor.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
