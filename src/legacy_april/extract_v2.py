import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_ROOT = Path("./data/raw_droh")   # ảnh gốc theo class
RAW_OUT_ROOT = Path("./test/skeleton_raw")
VIS_OUT_ROOT = Path("./test/visualize")
META_CSV = Path("./test/mediapipe_metadata.csv")

RAW_OUT_ROOT.mkdir(parents=True, exist_ok=True)
VIS_OUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_NUM_HANDS = 1
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5

DRAW_VIS = True

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)

rows = []
saved = 0
failed = 0

for class_dir in sorted(INPUT_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    raw_class_dir = RAW_OUT_ROOT / class_dir.name
    vis_class_dir = VIS_OUT_ROOT / class_dir.name
    raw_class_dir.mkdir(parents=True, exist_ok=True)
    vis_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(class_dir.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            rows.append({
                "class": class_dir.name,
                "image_path": str(img_path),
                "status": "read_fail",
                "handedness": "",
                "score": -1.0,
                "raw_path": ""
            })
            failed += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            rows.append({
                "class": class_dir.name,
                "image_path": str(img_path),
                "status": "no_hand",
                "handedness": "",
                "score": -1.0,
                "raw_path": ""
            })
            failed += 1
            continue

        hand_landmarks = result.multi_hand_landmarks[0]

        handedness_label = ""
        handedness_score = -1.0
        if result.multi_handedness:
            handedness_label = result.multi_handedness[0].classification[0].label
            handedness_score = float(result.multi_handedness[0].classification[0].score)

        kp = []
        for lm in hand_landmarks.landmark:
            kp.append([lm.x, lm.y, lm.z])
        kp = np.array(kp, dtype=np.float32)   # shape (21,3)

        raw_path = raw_class_dir / f"{img_path.stem}.npy"
        np.save(raw_path, kp)

        if DRAW_VIS:
            vis_img = img_bgr.copy()
            mp_draw.draw_landmarks(
                vis_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            text = f"{handedness_label} {handedness_score:.2f}"
            cv2.putText(vis_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(str(vis_class_dir / f"{img_path.stem}_vis.jpg"), vis_img)

        rows.append({
            "class": class_dir.name,
            "image_path": str(img_path),
            "status": "ok",
            "handedness": handedness_label,
            "score": handedness_score,
            "raw_path": str(raw_path)
        })
        saved += 1

df = pd.DataFrame(rows)
df.to_csv(META_CSV, index=False)

print("DONE")
print("Saved raw skeletons:", saved)
print("Failed samples:", failed)
print("Metadata:", META_CSV)