import numpy as np
from pathlib import Path

INPUT_ROOT = Path("./dataset/skeleton_raw")
OUTPUT_ROOT = Path("./dataset/skeleton_final")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def align_hand(kp):
    wrist = kp[0]
    index = kp[5]
    middle = kp[9]
    pinky = kp[17]

    y = (middle - wrist)
    y /= np.linalg.norm(y) + 1e-6

    x = (pinky - index)
    x /= np.linalg.norm(x) + 1e-6

    z = np.cross(x, y)
    z /= np.linalg.norm(z) + 1e-6

    if z[2] < 0:
        x = -x
        z = -z

    x = np.cross(y, z)

    R = np.stack([x, y, z], axis=1)
    kp = kp @ R

    return kp

def normalize_hand(kp: np.ndarray) -> np.ndarray:
    kp = kp.copy().astype(np.float32)

    kp = kp - kp[0]

    scale = np.linalg.norm(kp[9] - kp[0]) + 1e-6
    kp = kp / scale
    
    kp = align_hand(kp)

    return kp

count = 0

for class_dir in sorted(INPUT_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    out_class = OUTPUT_ROOT / class_dir.name
    out_class.mkdir(parents=True, exist_ok=True)

    for npy_path in sorted(class_dir.glob("*.npy")):
        kp = np.load(npy_path).astype(np.float32)
        if kp.shape != (21, 3):
            print(f"[SKIP] {npy_path} shape={kp.shape}")
            continue

        kp_norm = normalize_hand(kp)
        np.save(out_class / npy_path.name, kp_norm)
        count += 1

print(f"Normalized {count} skeleton files.")
print(f"Saved to: {OUTPUT_ROOT}")