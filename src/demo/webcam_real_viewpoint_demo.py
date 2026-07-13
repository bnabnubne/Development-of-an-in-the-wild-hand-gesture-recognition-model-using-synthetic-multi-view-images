"""Fair live comparison using only real webcam viewpoints.

No synthetic view is created at inference. Both models receive the same
MediaPipe skeleton and the same DrOh preprocessing. The presenter physically
rotates the hand; results are accumulated over observed palm-yaw bins.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np

from inference_core import (
    CLASS_NAMES,
    CONNECTIONS,
    DISPLAY_NAMES,
    PROJECT_ROOT,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
)


ANGLE_EDGES = np.array([-90, -60, -30, 0, 30, 60, 90], dtype=np.float32)
ANGLE_LABELS = ["-75", "-45", "-15", "+15", "+45", "+75"]


def estimate_palm_yaw(raw_landmarks):
    """Estimate camera-relative palm yaw from the unaligned MediaPipe hand."""
    points = np.asarray(raw_landmarks, dtype=np.float32)
    wrist = points[0]
    across = points[17] - points[5]
    along = points[9] - wrist
    normal = np.cross(across, along)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return 0.0
    normal /= norm
    # Frontal palm is near 0; left/right turns have opposite signs.
    return float(np.degrees(np.arctan2(normal[0], abs(normal[2]) + 1e-6)))


def angle_bin(angle):
    return int(np.clip(np.digitize(angle, ANGLE_EDGES) - 1, 0, len(ANGLE_LABELS) - 1))


@dataclass
class ViewpointStats:
    # Latest stable observation per real angle bin avoids over-counting easy frontal frames.
    baseline_correct: list = field(default_factory=lambda: [None] * 6)
    proposed_correct: list = field(default_factory=lambda: [None] * 6)
    baseline_class: list = field(default_factory=lambda: [None] * 6)
    proposed_class: list = field(default_factory=lambda: [None] * 6)
    observations: list = field(default_factory=lambda: [0] * 6)

    def update(self, index, ground_truth, baseline_prediction, proposed_prediction):
        self.baseline_correct[index] = baseline_prediction["class"] == ground_truth
        self.proposed_correct[index] = proposed_prediction["class"] == ground_truth
        self.baseline_class[index] = baseline_prediction["class"]
        self.proposed_class[index] = proposed_prediction["class"]
        self.observations[index] += 1

    def reset(self):
        self.__dict__.update(ViewpointStats().__dict__)

    def score(self, values):
        observed = [value for value in values if value is not None]
        return sum(observed), len(observed)


def draw_demo(frame, extraction, ground_truth, baseline_prediction, proposed_prediction, yaw, stats, fps):
    height, width = frame.shape[:2]
    panel_width = max(500, int(width * 0.49))
    camera_width = width - panel_width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:, :camera_width] = cv2.resize(frame, (camera_width, height))
    x0 = camera_width
    cv2.rectangle(output, (x0, 0), (width, height), (12, 15, 21), -1)

    if extraction is not None:
        points = extraction["raw_landmarks"]
        pixels = np.column_stack((points[:, 0] * camera_width, points[:, 1] * height)).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(output, tuple(pixels[a]), tuple(pixels[b]), (55, 215, 255), 3, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(output, (int(x), int(y)), 5, (35, 70, 255), -1, cv2.LINE_AA)

    cv2.putText(output, "REAL-WEBCAM VIEWPOINT TEST", (x0 + 20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, "Same input + same preprocessing | no synthetic inference", (x0 + 20, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (185, 195, 208), 1, cv2.LINE_AA)
    cv2.putText(output, f"Ground truth [{CLASS_NAMES.index(ground_truth)+1}]: {DISPLAY_NAMES[ground_truth]}", (x0 + 20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 220, 230), 1, cv2.LINE_AA)

    if extraction is None:
        cv2.putText(output, "Show the full hand, then rotate it slowly", (x0 + 20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 190, 255), 2, cv2.LINE_AA)
    else:
        current_bin = angle_bin(yaw)
        cv2.putText(output, f"Observed palm yaw: {yaw:+.1f} deg", (x0 + 20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (225, 225, 225), 1, cv2.LINE_AA)

        def current_row(y, title, prediction, accent):
            correct = prediction["class"] == ground_truth
            status = "CORRECT" if correct else "WRONG"
            status_color = (75, 225, 110) if correct else (75, 90, 245)
            text = f"{DISPLAY_NAMES[prediction['class']]}  {prediction['confidence']*100:.1f}%  {status}"
            cv2.putText(output, title, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent, 2, cv2.LINE_AA)
            cv2.putText(output, text, (x0 + 190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2, cv2.LINE_AA)

        current_row(151, "Raw baseline", baseline_prediction, (120, 190, 255))
        current_row(184, "Blender8 MV", proposed_prediction, (90, 235, 130))

        cv2.putText(output, "LATEST RESULT IN EACH REAL VIEWPOINT BIN", (x0 + 20, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cell_width = (panel_width - 40) // 6
        for index, label in enumerate(ANGLE_LABELS):
            left = x0 + 20 + index * cell_width
            right = left + cell_width - 6
            border = (90, 235, 230) if index == current_bin else (85, 90, 100)
            cv2.rectangle(output, (left, 240), (right, 342), border, 2)
            cv2.putText(output, label, (left + 7, 259), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 215), 1, cv2.LINE_AA)

            def mark(y, value, prefix):
                if value is None:
                    text, color = f"{prefix}: --", (130, 135, 145)
                else:
                    text = f"{prefix}: {'OK' if value else 'X'}"
                    color = (70, 220, 105) if value else (70, 85, 245)
                cv2.putText(output, text, (left + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

            mark(292, stats.baseline_correct[index], "B")
            mark(322, stats.proposed_correct[index], "MV")

        b_ok, observed = stats.score(stats.baseline_correct)
        p_ok, _ = stats.score(stats.proposed_correct)
        cv2.putText(output, f"Observed viewpoint coverage: {observed}/6 bins", (x0 + 20, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 215, 225), 1, cv2.LINE_AA)
        cv2.putText(output, f"Baseline: {b_ok}/{observed} bins correct", (x0 + 20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (120, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"Multiview: {p_ok}/{observed} bins correct", (x0 + 20, 443), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (90, 235, 130), 2, cv2.LINE_AA)
        difference = p_ok - b_ok
        color = (90, 235, 130) if difference > 0 else ((120, 190, 255) if difference < 0 else (215, 215, 215))
        cv2.putText(output, f"Measured live difference: {difference:+d} viewpoint bins", (x0 + 20, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.putText(output, "Rotate slowly left/right | R: reset | 1-6: GT | Q: quit", (x0 + 20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (185, 190, 200), 1, cv2.LINE_AA)
    cv2.putText(output, f"FPS {fps:.1f}", (width - 78, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 155, 165), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", default=None)
    parser.add_argument("--sample-interval", type=float, default=0.20)
    args = parser.parse_args()

    root = PROJECT_ROOT / "model/results/defense_experiments_6cls"
    baseline = EnsembleGestureRecognizer([root / f"gru_wrist_middle_views0_lambda0p0_seed{s}/best.pt" for s in [0, 1, 42]], 1)
    proposed = EnsembleGestureRecognizer([root / f"gru_wrist_middle_views8_lambda0p0_seed{s}/best.pt" for s in [0, 1, 42]], 1)
    extractor = MediaPipeHandExtractor(static_image_mode=False)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ground_truth = CLASS_NAMES[0]
    stats = ViewpointStats()
    last_sample = 0.0
    last_time = perf_counter()
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            extraction = extractor.process_bgr(frame)
            baseline_prediction = proposed_prediction = None
            yaw = 0.0
            now = perf_counter()
            if extraction is not None:
                # Identical deployment input for both models; no generated camera view.
                skeleton = extraction["skeleton"]
                baseline_prediction = baseline.predict_skeleton(skeleton, smooth=False)
                proposed_prediction = proposed.predict_skeleton(skeleton, smooth=False)
                yaw = estimate_palm_yaw(extraction["raw_landmarks"])
                if now - last_sample >= args.sample_interval:
                    stats.update(angle_bin(yaw), ground_truth, baseline_prediction, proposed_prediction)
                    last_sample = now
            fps = 1.0 / max(now - last_time, 1e-8)
            last_time = now
            display = draw_demo(frame, extraction, ground_truth, baseline_prediction, proposed_prediction, yaw, stats, fps)

            if args.record and writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (display.shape[1], display.shape[0]))
            if writer is not None:
                writer.write(display)
            cv2.imshow("Fair Real-Viewpoint Comparison - press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                ground_truth = CLASS_NAMES[key - ord("1")]
                stats.reset()
            elif key == ord("r"):
                stats.reset()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
