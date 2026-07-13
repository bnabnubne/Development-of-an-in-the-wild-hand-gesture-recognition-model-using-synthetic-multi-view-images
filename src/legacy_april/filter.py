import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
VIS_ROOT = Path("./test/visualize")
SRC_ROOT = Path("./test/skeleton_final")
OUT_ROOT = Path("./test/skeleton_final_v2")

# nếu muốn xóa folder output cũ trước khi chạy
REMOVE_OLD_OUTPUT = True

# các đuôi ảnh chấp nhận trong visualize
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# PREPARE OUTPUT
# =========================
if REMOVE_OLD_OUTPUT and OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# MAIN
# =========================
copied = 0
missing = 0
skipped = 0

for vis_class_dir in sorted(VIS_ROOT.iterdir()):
    if not vis_class_dir.is_dir():
        continue

    class_name = vis_class_dir.name
    src_class_dir = SRC_ROOT / class_name
    out_class_dir = OUT_ROOT / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    if not src_class_dir.exists():
        print(f"[WARN] Missing class folder in skeleton_final: {src_class_dir}")
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

        src_npy = src_class_dir / f"{base_name}.npy"
        dst_npy = out_class_dir / f"{base_name}.npy"

        if src_npy.exists():
            shutil.copy2(src_npy, dst_npy)
            copied += 1
        else:
            print(f"[MISS] {src_npy}")
            missing += 1

print("\nDONE")
print("Copied skeleton files :", copied)
print("Missing skeleton files:", missing)
print("Skipped non-image     :", skipped)
print("Saved to:", OUT_ROOT)