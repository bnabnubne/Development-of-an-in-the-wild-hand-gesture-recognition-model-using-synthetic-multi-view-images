import numpy as np
from pathlib import Path

NPY_PATH = "./dataset/skeleton_raw/ok/ok (36).npy"
OUT_PATH = "./2d/ok/ok (36).npy"

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

kp = np.load(NPY_PATH).astype(np.float32)

if kp.shape != (21, 3):
    raise ValueError(f"Expected (21,3), got {kp.shape}")

xy = kp[:, :2].copy()

np.save(OUT_PATH, xy)

print("Saved:", OUT_PATH)
print("Shape:", xy.shape)
print("Min:", xy.min(axis=0))
print("Max:", xy.max(axis=0))
print("Wrist:", xy[0])
print("Thumb tip:", xy[4])
print("Index tip:", xy[8])