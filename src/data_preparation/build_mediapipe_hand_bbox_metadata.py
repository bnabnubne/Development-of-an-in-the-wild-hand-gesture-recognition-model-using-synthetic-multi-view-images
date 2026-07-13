import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parent
RGB_MV_CSV = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview.csv"
SALUX_CSV = MODEL_ROOT / "metadata" / "salux_original_rgb_5cls.csv"
DROH_CSV = MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv"

OUT_RGB_MV_CSV = MODEL_ROOT / "metadata" / "rgb_skeleton_multiview_mphand_bbox.csv"
OUT_SALUX_CSV = MODEL_ROOT / "metadata" / "salux_original_rgb_5cls_mphand_bbox.csv"
OUT_DROH_CSV = MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls_mphand_bbox.csv"
OUT_SUMMARY = MODEL_ROOT / "metadata" / "mediapipe_hand_bbox_summary.json"


def discover_cam_cols(df):
    return sorted(
        [c for c in df.columns if c.startswith("rgb_cam_") and c.endswith("_path")],
        key=lambda c: int(c.split("_")[2]),
    )


class HandBBoxDetector:
    def __init__(self, min_detection_confidence=0.25):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self.hands.close()

    def detect_bbox(self, image_path):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        hand = result.multi_hand_landmarks[0]
        xs = np.array([lm.x for lm in hand.landmark], dtype=np.float32)
        ys = np.array([lm.y for lm in hand.landmark], dtype=np.float32)
        xs = np.clip(xs, 0.0, 1.0)
        ys = np.clip(ys, 0.0, 1.0)
        return {
            "x1": float(xs.min()),
            "y1": float(ys.min()),
            "x2": float(xs.max()),
            "y2": float(ys.max()),
        }


def add_external_bboxes(df, detector, path_col="image_path", progress_name="external"):
    rows = []
    detected = 0
    for idx, row in df.iterrows():
        out = row.to_dict()
        bbox = detector.detect_bbox(row[path_col])
        if bbox is not None:
            detected += 1
            for key, value in bbox.items():
                out[f"mphand_bbox_{key}"] = value
        else:
            for key in ["x1", "y1", "x2", "y2"]:
                out[f"mphand_bbox_{key}"] = np.nan
        rows.append(out)
        if (idx + 1) % 100 == 0:
            print(f"{progress_name}: {idx + 1}/{len(df)} detected={detected}", flush=True)
    return pd.DataFrame(rows), detected


def add_multiview_bboxes(df, detector):
    cam_cols = discover_cam_cols(df)
    rows = []
    total = 0
    detected = 0
    for idx, row in df.iterrows():
        out = row.to_dict()
        for cam_col in cam_cols:
            cam_id = int(cam_col.split("_")[2])
            bbox = detector.detect_bbox(row[cam_col])
            total += 1
            if bbox is not None:
                detected += 1
                for key, value in bbox.items():
                    out[f"mphand_bbox_cam_{cam_id}_{key}"] = value
            else:
                for key in ["x1", "y1", "x2", "y2"]:
                    out[f"mphand_bbox_cam_{cam_id}_{key}"] = np.nan
        rows.append(out)
        if (idx + 1) % 50 == 0:
            print(f"render: rows={idx + 1}/{len(df)} images={total} detected={detected}", flush=True)
    return pd.DataFrame(rows), detected, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-detection-confidence", type=float, default=0.25)
    parser.add_argument(
        "--datasets",
        default="render,salux,droh",
        help="Comma-separated datasets to process: render,salux,droh.",
    )
    args = parser.parse_args()
    selected = {part.strip() for part in args.datasets.split(",") if part.strip()}

    detector = HandBBoxDetector(args.min_detection_confidence)
    summary = {
        "min_detection_confidence": args.min_detection_confidence,
        "outputs": {},
    }

    try:
        if "render" in selected:
            df = pd.read_csv(RGB_MV_CSV)
            out_df, detected, total = add_multiview_bboxes(df, detector)
            out_df.to_csv(OUT_RGB_MV_CSV, index=False)
            summary["outputs"]["render"] = {
                "rows": len(out_df),
                "images": total,
                "detected": detected,
                "detection_rate": detected / total if total else 0.0,
                "path": str(OUT_RGB_MV_CSV),
            }

        if "salux" in selected:
            df = pd.read_csv(SALUX_CSV)
            out_df, detected = add_external_bboxes(df, detector, progress_name="salux")
            out_df.to_csv(OUT_SALUX_CSV, index=False)
            summary["outputs"]["salux"] = {
                "rows": len(out_df),
                "detected": detected,
                "detection_rate": detected / len(out_df) if len(out_df) else 0.0,
                "path": str(OUT_SALUX_CSV),
            }

        if "droh" in selected:
            df = pd.read_csv(DROH_CSV)
            out_df, detected = add_external_bboxes(df, detector, progress_name="droh")
            out_df.to_csv(OUT_DROH_CSV, index=False)
            summary["outputs"]["droh"] = {
                "rows": len(out_df),
                "detected": detected,
                "detection_rate": detected / len(out_df) if len(out_df) else 0.0,
                "path": str(OUT_DROH_CSV),
            }
    finally:
        detector.close()

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
