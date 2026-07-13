import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
VIS_ROOT = Path("./test/visualize")
RAW_ROOT = Path("./test/skeleton_raw")
META_CSV = Path("./test/mediapipe_metadata.csv")

OUT_ROOT = Path("./test/droh_real2d_wristnorm")
OUT_META_DIR = Path("./metadata")
OUT_META_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_META_DIR / "droh_real2d_wristnorm.csv"

REMOVE_OLD_OUTPUT = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

# =========================
# PREPARE OUTPUT
# =========================
if REMOVE_OLD_OUTPUT and OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD HANDEDNESS META
# =========================
stem_to_handedness = {}

if META_CSV.exists():
    meta_df = pd.read_csv(META_CSV)

    for _, row in meta_df.iterrows():
        handedness = str(row.get("handedness", "")).strip()

        raw_path = str(row.get("raw_path", "")).strip()
        image_path = str(row.get("image_path", "")).strip()

        if raw_path:
            stem_to_handedness[Path(raw_path).stem] = handedness
        if image_path:
            stem_to_handedness[Path(image_path).stem] = handedness

# =========================
# FUNCTIONS
# =========================
def normalize_2d(coords_2d: np.ndarray) -> np.ndarray:
    coords_2d = coords_2d.astype(np.float32).copy()

    # center theo wrist
    wrist = coords_2d[WRIST_IDX].copy()
    coords_2d = coords_2d - wrist

    # scale theo wrist -> middle_mcp
    scale = np.linalg.norm(coords_2d[MIDDLE_MCP_IDX]) + 1e-6
    coords_2d = coords_2d / scale

    return coords_2d

# =========================
# MAIN
# =========================
rows = []
saved = 0
missing = 0
skipped = 0

for vis_class_dir in sorted(VIS_ROOT.iterdir()):
    if not vis_class_dir.is_dir():
        continue

    class_name = vis_class_dir.name
    raw_class_dir = RAW_ROOT / class_name
    out_class_dir = OUT_ROOT / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    if not raw_class_dir.exists():
        print(f"[WARN] Missing raw class folder: {raw_class_dir}")
        continue

    for vis_path in sorted(vis_class_dir.iterdir()):
        if not vis_path.is_file():
            continue
        if vis_path.suffix.lower() not in IMAGE_EXTS:
            skipped += 1
            continue

        # ví dụ:
        # IMG_6289_vis.jpg -> IMG_6289
        # IMG_6289.jpg     -> IMG_6289
        stem = vis_path.stem
        if stem.endswith("_vis"):
            base_name = stem[:-4]
        else:
            base_name = stem

        raw_npy = raw_class_dir / f"{base_name}.npy"

        if not raw_npy.exists():
            print(f"[MISS] {raw_npy}")
            missing += 1
            continue

        kp = np.load(raw_npy).astype(np.float32)   # shape (21,3)
        if kp.shape != (21, 3):
            print(f"[SKIP] Invalid shape {kp.shape}: {raw_npy}")
            skipped += 1
            continue

        # lấy 2D thật từ DrOh, KHÔNG qua Blender
        coords_2d = kp[:, :2].copy()

        # canonicalize nếu sample là Left
        handedness = stem_to_handedness.get(base_name, "")
        if handedness.lower() == "left":
            # mediapipe x trong ảnh là normalized [0,1]
            coords_2d[:, 0] = 1.0 - coords_2d[:, 0]

        coords_2d = normalize_2d(coords_2d)

        out_path = out_class_dir / f"{base_name}.npy"
        np.save(out_path, coords_2d)

        rows.append({
            "action": class_name,
            "sample_id": base_name,
            "input_path": str(out_path),
            "split": "test",
            "handedness": handedness
        })

        saved += 1

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\nDONE")
print("Saved real 2D files :", saved)
print("Missing raw files   :", missing)
print("Skipped files       :", skipped)
print("Output folder       :", OUT_ROOT)
print("Metadata CSV        :", OUT_CSV)