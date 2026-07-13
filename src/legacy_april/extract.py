import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

mp_hands = mp.solutions.hands

INPUT_ROOT = Path("./data/raw_droh")
OUTPUT_ROOT = Path("./test/skeleton_raw")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_keypoints(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    keypoints = []
    for lm in hand.landmark:
        keypoints.append([lm.x, lm.y, lm.z])
    return np.array(keypoints, dtype=np.float32)   # (21,3)

n_ok = 0
n_fail = 0

for class_dir in INPUT_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    out_class = OUTPUT_ROOT / class_dir.name
    out_class.mkdir(parents=True, exist_ok=True)

    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        kp = extract_hand_keypoints(img_path)
        if kp is None:
            n_fail += 1
            continue

        np.save(out_class / f"{img_path.stem}.npy", kp)
        n_ok += 1

print("Extracted:", n_ok)
print("Failed:", n_fail)