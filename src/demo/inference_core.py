
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hand-gesture-demo")

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model/results/paper_controlled_raw_mv_6cls/best.pt"
RAW_BASELINE_ROOT = PROJECT_ROOT / "model/results/defense_experiments_6cls"
FINAL_CONSISTENCY_ROOT = PROJECT_ROOT / "model/results/consistency_finetune_6cls"
FINAL_SEEDS = [0, 1, 7, 21, 42]
RAW_BASELINE_5_CHECKPOINTS = [
    RAW_BASELINE_ROOT / f"gru_wrist_middle_views0_lambda0p0_seed{seed}/best.pt"
    for seed in FINAL_SEEDS
]
FINAL_CONSISTENCY_5_CHECKPOINTS = [
    FINAL_CONSISTENCY_ROOT / f"lambda0p3_lr1em4_seed{seed}/best.pt"
    for seed in FINAL_SEEDS
]
CAMERA_SINGLE_ROOT = PROJECT_ROOT / "model/results/clean_viewpoint_factorial_v2_6cls"
CAMERA_SINGLE_5_CHECKPOINTS = [
    CAMERA_SINGLE_ROOT / f"camera_single_seed{seed}/best.pt"
    for seed in FINAL_SEEDS
]
CLASS_NAMES = ["ok", "paper", "rock", "scissors", "the-finger", "thumb"]
DISPLAY_NAMES = {
    "ok": "OK",
    "paper": "Paper",
    "rock": "Rock",
    "scissors": "Scissors",
    "the-finger": "The-Finger",
    "thumb": "Thumb",
}
CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)
CAMERA_ANGLES_DEG = [-45, 0, 30, 45, 60, 90, 120, 135]
CAMERA_RADIUS = 3.0
CAMERA_HEIGHT = 0.8


class GRU6(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(3, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(CLASS_NAMES))

    def forward(self, x):
        output, _ = self.gru(x)
        feature = output[:, -1]
        return self.fc(feature), feature


def canonicalize_and_align(landmarks, handedness):
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3).copy()
    if str(handedness).lower() == "left":
        points[:, 0] = 1.0 - points[:, 0]
        points[:, 2] *= -1.0

    points -= points[0]
    scale = float(np.linalg.norm(points[9] - points[0]))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("Degenerate wrist-to-middle-MCP scale")
    points /= scale

    wrist, index, middle, pinky = points[0], points[5], points[9], points[17]
    y_axis = middle - wrist
    y_axis /= np.linalg.norm(y_axis) + 1e-6
    x_axis = pinky - index
    x_axis /= np.linalg.norm(x_axis) + 1e-6
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-6
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-6
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    points = points @ rotation

    x_check = points[17] - points[5]
    x_check /= np.linalg.norm(x_check) + 1e-6
    y_check = points[9] - points[0]
    y_check /= np.linalg.norm(y_check) + 1e-6
    if np.cross(x_check, y_check)[2] < 0:
        points[:, 0] *= -1
        points[:, 2] *= -1
    return points.astype(np.float32)


def minimal_camera_preprocess(landmarks, handedness):
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3).copy()
    if str(handedness).lower() == "left":
        points[:, 0] = 1.0 - points[:, 0]
        points[:, 2] *= -1.0
    points -= points[0:1]
    scale = float(np.linalg.norm(points[9] - points[0]))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("Degenerate wrist-to-middle-MCP scale")
    return (points / scale).astype(np.float32)


def model_normalize(points):
    points = np.asarray(points, dtype=np.float32).reshape(21, 3).copy()
    points -= points[0:1]
    scale = float(np.linalg.norm(points[9] - points[0]))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("Invalid model normalization scale")
    return points / scale


def generate_blender8_views(skeleton):
    world = model_normalize(skeleton)
    world -= world.mean(axis=0, keepdims=True)
    views = []
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for angle_deg in CAMERA_ANGLES_DEG:
        angle = np.deg2rad(angle_deg)
        camera = np.array(
            [CAMERA_RADIUS * np.sin(angle), -CAMERA_RADIUS * np.cos(angle), CAMERA_HEIGHT],
            dtype=np.float32,
        )
        z_axis = camera / (np.linalg.norm(camera) + 1e-8)
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-8
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis) + 1e-8
        world_to_camera = np.stack([x_axis, y_axis, z_axis], axis=0)
        view = (world_to_camera @ (world - camera).T).T
        view -= view[0:1]
        views.append(view.astype(np.float32))
    return views


def evaluate_virtual_views(recognizer, skeleton, ground_truth):
    results = [recognizer.predict_skeleton(view, smooth=False) for view in generate_blender8_views(skeleton)]
    correct = sum(result["class"] == ground_truth for result in results)
    majority = max(CLASS_NAMES, key=lambda name: sum(result["class"] == name for result in results))
    agreement = sum(result["class"] == majority for result in results)
    return {"results": results, "correct": correct, "majority": majority, "agreement": agreement}


class GestureRecognizer:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, smoothing_window=7):
        self.checkpoint_path = Path(checkpoint)
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model = GRU6(hidden_dim=128)
        state = checkpoint_data["model_state_dict"]
        if any(key.startswith("encoder.") for key in state):
            state = {
                key.replace("encoder.", "gru.", 1).replace("classifier.", "fc.", 1): value
                for key, value in state.items()
            }
        self.model.load_state_dict(state)
        self.model.eval()
        self.history = deque(maxlen=max(1, int(smoothing_window)))

    def reset(self):
        self.history.clear()

    def predict_skeleton(self, skeleton, smooth=False):
        normalized = model_normalize(skeleton)
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        start = perf_counter()
        with torch.no_grad():
            logits, _ = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].numpy()
        latency_ms = (perf_counter() - start) * 1000.0
        if smooth:
            self.history.append(probabilities)
            probabilities = np.mean(np.stack(self.history), axis=0)
        order = np.argsort(probabilities)[::-1]
        return {
            "class": CLASS_NAMES[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "probabilities": probabilities,
            "top3": [
                (CLASS_NAMES[int(index)], float(probabilities[index]))
                for index in order[:3]
            ],
            "latency_ms": latency_ms,
            "normalized_skeleton": normalized,
        }


class EnsembleGestureRecognizer:

    def __init__(self, checkpoints=None, smoothing_window=7):
        if checkpoints is None:
            root = PROJECT_ROOT / "model/results/defense_experiments_6cls"
            checkpoints = [
                root / f"gru_wrist_middle_views8_lambda0p0_seed{seed}/best.pt"
                for seed in [0, 1, 42]
            ]
        self.members = [GestureRecognizer(path, smoothing_window=1) for path in checkpoints]
        self.history = deque(maxlen=max(1, int(smoothing_window)))

    def reset(self):
        self.history.clear()

    def predict_skeleton(self, skeleton, smooth=False):
        member_results = [member.predict_skeleton(skeleton, smooth=False) for member in self.members]
        probabilities = np.mean([result["probabilities"] for result in member_results], axis=0)
        if smooth:
            self.history.append(probabilities)
            probabilities = np.mean(np.stack(self.history), axis=0)
        order = np.argsort(probabilities)[::-1]
        return {
            "class": CLASS_NAMES[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "probabilities": probabilities,
            "top3": [(CLASS_NAMES[int(index)], float(probabilities[index])) for index in order[:3]],
            "latency_ms": float(sum(result["latency_ms"] for result in member_results)),
            "normalized_skeleton": member_results[0]["normalized_skeleton"],
        }


class MediaPipeHandExtractor:
    def __init__(self, static_image_mode, min_detection_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self.hands.close()

    def process_bgr(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        handedness = "Right"
        handedness_score = 0.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = classification.label
            handedness_score = float(classification.score)
        raw = np.array([[point.x, point.y, point.z] for point in hand.landmark], dtype=np.float32)
        camera_skeleton = minimal_camera_preprocess(raw, handedness)
        skeleton = canonicalize_and_align(raw, handedness)
        return {
            "raw_landmarks": raw,
            "camera_skeleton": camera_skeleton,
            "skeleton": skeleton,
            "handedness": handedness,
            "handedness_score": handedness_score,
        }


def draw_result(image_bgr, extraction, prediction=None, confidence_threshold=0.55):
    output = image_bgr.copy()
    height, width = output.shape[:2]
    if extraction is not None:
        points = extraction["raw_landmarks"]
        pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(output, tuple(pixels[a]), tuple(pixels[b]), (60, 210, 255), 3, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(output, (int(x), int(y)), 5, (30, 60, 255), -1, cv2.LINE_AA)

    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (min(width - 12, 540), 142), (10, 10, 10), -1)
    output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0)
    if prediction is None:
        cv2.putText(output, "No hand detected", (28, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 180, 255), 2)
        return output

    confidence = prediction["confidence"]
    class_name = DISPLAY_NAMES[prediction["class"]]
    label = class_name if confidence >= confidence_threshold else f"Uncertain: {class_name}"
    color = (80, 230, 110) if confidence >= confidence_threshold else (80, 190, 255)
    cv2.putText(output, label, (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
    cv2.putText(output, f"Confidence: {confidence * 100:.1f}%", (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, f"Model latency: {prediction['latency_ms']:.2f} ms", (28, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1, cv2.LINE_AA)
    return output


def draw_comparison_result(
    image_bgr,
    extraction,
    baseline_prediction=None,
    proposed_prediction=None,
    confidence_threshold=0.55,
    baseline_title="BASELINE (raw)",
    proposed_title="PROPOSED (8-view MV)",
    baseline_accuracy=76.44,
    proposed_accuracy=84.00,
):
    output = image_bgr.copy()
    height, width = output.shape[:2]
    if extraction is not None:
        points = extraction["raw_landmarks"]
        pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(output, tuple(pixels[a]), tuple(pixels[b]), (60, 210, 255), 3, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(output, (int(x), int(y)), 5, (30, 60, 255), -1, cv2.LINE_AA)

    panel_width = min(width - 24, 760)
    panel_height = 252
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width, 12 + panel_height), (8, 8, 8), -1)
    output = cv2.addWeighted(overlay, 0.78, output, 0.22, 0)

    if baseline_prediction is None or proposed_prediction is None:
        cv2.putText(output, "No hand detected", (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (80, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(output, "Show the full hand inside the camera frame", (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)
        return output

    cv2.line(output, (28, 132), (12 + panel_width - 16, 132), (150, 150, 150), 1, cv2.LINE_AA)

    def draw_card(y, title, evaluated_accuracy, prediction, accent):
        confidence = prediction["confidence"]
        gesture = DISPLAY_NAMES[prediction["class"]]
        display = gesture if confidence >= confidence_threshold else f"Uncertain: {gesture}"
        cv2.putText(output, title, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, accent, 2, cv2.LINE_AA)
        cv2.putText(output, f"Prediction: {display}", (30, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(output, f"Conf: {confidence * 100:.1f}%", (30, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(output, f"Test: {evaluated_accuracy:.2f}% | {prediction['latency_ms']:.1f} ms", (30, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (195, 195, 195), 1, cv2.LINE_AA)

    draw_card(39, baseline_title, baseline_accuracy, baseline_prediction, (120, 190, 255))
    draw_card(158, proposed_title, proposed_accuracy, proposed_prediction, (90, 235, 130))
    return output


def draw_multiview_stress_result(image_bgr, extraction, ground_truth, baseline_eval, proposed_eval):
    output = image_bgr.copy()
    height, width = output.shape[:2]
    shade = output.copy()
    cv2.rectangle(shade, (0, 0), (width, height), (8, 10, 14), -1)
    output = cv2.addWeighted(shade, 0.90, output, 0.10, 0)
    gt = DISPLAY_NAMES[ground_truth]
    cv2.putText(output, "8-VIEW VIRTUAL-CAMERA STRESS TEST", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, f"Captured pose | Ground truth: {gt}", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (205, 215, 225), 1, cv2.LINE_AA)

    def row(y, title, evaluation, accent):
        score = evaluation["correct"]
        majority = DISPLAY_NAMES[evaluation["majority"]]
        cv2.putText(output, title, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, accent, 2, cv2.LINE_AA)
        cv2.putText(output, f"Correct: {score}/8   Majority: {majority}   Agreement: {evaluation['agreement']}/8", (24, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
        cell_w = max(72, (width - 48) // 8)
        for index, (angle, result) in enumerate(zip(CAMERA_ANGLES_DEG, evaluation["results"])):
            x = 24 + index * cell_w
            good = result["class"] == ground_truth
            color = (75, 220, 110) if good else (80, 95, 245)
            cv2.rectangle(output, (x, y + 45), (min(x + cell_w - 7, width - 10), y + 112), color, 2)
            cv2.putText(output, f"{angle:+d} deg", (x + 5, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 215, 215), 1, cv2.LINE_AA)
            short = DISPLAY_NAMES[result["class"]][:7]
            cv2.putText(output, short, (x + 5, y + 89), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            cv2.putText(output, f"{result['confidence'] * 100:.0f}%", (x + 5, y + 107), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1, cv2.LINE_AA)

    row(120, "BASELINE (raw training)", baseline_eval, (120, 190, 255))
    row(275, "PROPOSED (Blender8 MV training)", proposed_eval, (90, 235, 130))
    delta = proposed_eval["correct"] - baseline_eval["correct"]
    delta_color = (90, 235, 130) if delta > 0 else ((120, 190, 255) if delta < 0 else (220, 220, 220))
    cv2.putText(output, f"View-robustness difference: {delta:+d}/8", (24, min(height - 52, 442)), cv2.FONT_HERSHEY_SIMPLEX, 0.66, delta_color, 2, cv2.LINE_AA)
    cv2.putText(output, "R: return to live   |   1-6: change ground truth   |   Q: quit", (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 195, 205), 1, cv2.LINE_AA)
    return output
