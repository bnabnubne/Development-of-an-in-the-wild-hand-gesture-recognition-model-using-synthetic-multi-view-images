import numpy as np
import pandas as pd
from pathlib import Path

RAW_ROOT = Path("./test/skeleton_raw")
META_CSV = Path("./test/mediapipe_metadata.csv")
FINAL_ROOT = Path("./test/skeleton_final")
FINAL_ROOT.mkdir(parents=True, exist_ok=True)

WRIST_IDX = 0
INDEX_MCP_IDX = 5
MIDDLE_MCP_IDX = 9
PINKY_MCP_IDX = 17

def canonicalize_hand(kp: np.ndarray, handedness: str) -> np.ndarray:
    kp = kp.copy()
    if handedness.lower() == "left":
        kp[:, 0] = 1.0 - kp[:, 0]   # vì MediaPipe x đang ở image normalized [0,1]
        kp[:, 2] = -kp[:, 2]        # flip z để giữ orientation hợp lý hơn
    return kp


def align_hand(kp: np.ndarray) -> np.ndarray:
    wrist = kp[WRIST_IDX]
    index = kp[INDEX_MCP_IDX]
    middle = kp[MIDDLE_MCP_IDX]
    pinky = kp[PINKY_MCP_IDX]

    y = middle - wrist
    y = y / (np.linalg.norm(y) + 1e-6)

    x = pinky - index
    x = x / (np.linalg.norm(x) + 1e-6)

    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-6)

    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-6)

    R = np.stack([x, y, z], axis=1)
    kp_aligned = kp @ R

    wrist2 = kp_aligned[WRIST_IDX]
    index2 = kp_aligned[INDEX_MCP_IDX]
    middle2 = kp_aligned[MIDDLE_MCP_IDX]
    pinky2 = kp_aligned[PINKY_MCP_IDX]

    x2 = pinky2 - index2
    x2 = x2 / (np.linalg.norm(x2) + 1e-6)

    y2 = middle2 - wrist2
    y2 = y2 / (np.linalg.norm(y2) + 1e-6)

    z2 = np.cross(x2, y2)
    z2 = z2 / (np.linalg.norm(z2) + 1e-6)

    if z2[2] < 0:
        kp_aligned[:, 0] *= -1
        kp_aligned[:, 2] *= -1

    return kp_aligned.astype(np.float32)


def normalize_hand(kp: np.ndarray, handedness: str) -> np.ndarray:
    kp = kp.copy().astype(np.float32)

    kp = canonicalize_hand(kp, handedness)

    kp = kp - kp[WRIST_IDX]

    scale = np.linalg.norm(kp[MIDDLE_MCP_IDX] - kp[WRIST_IDX]) + 1e-6
    kp = kp / scale

    kp = align_hand(kp)

    return kp.astype(np.float32)


df = pd.read_csv(META_CSV)

saved = 0
skipped = 0
rows_out = []

for _, row in df.iterrows():
    if row["status"] != "ok":
        skipped += 1
        continue

    raw_path = Path(row["raw_path"])
    if not raw_path.exists():
        skipped += 1
        continue

    kp = np.load(raw_path).astype(np.float32)
    if kp.shape != (21, 3):
        skipped += 1
        continue

    handedness = str(row["handedness"])
    kp_final = normalize_hand(kp, handedness)

    class_name = row["class"]
    out_class_dir = FINAL_ROOT / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_class_dir / raw_path.name
    np.save(out_path, kp_final)

    rows_out.append({
        "class": class_name,
        "image_path": row["image_path"],
        "handedness": handedness,
        "raw_path": str(raw_path),
        "final_path": str(out_path)
    })
    saved += 1

out_csv = FINAL_ROOT.parent / "skeleton_final_metadata.csv"
pd.DataFrame(rows_out).to_csv(out_csv, index=False)

print("DONE")
print("Saved final skeletons:", saved)
print("Skipped:", skipped)
print("Metadata:", out_csv)