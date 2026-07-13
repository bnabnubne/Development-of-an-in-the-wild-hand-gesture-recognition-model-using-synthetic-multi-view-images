import random
import shutil
from pathlib import Path

random.seed(42)

SRC_ROOT = Path("./dataset/skeleton_raw")
DST_ROOT = Path("./dataset/skeleton")
N_PER_CLASS = 70

DST_ROOT.mkdir(parents=True, exist_ok=True)

for class_dir in sorted(SRC_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    files = sorted(class_dir.glob("*.npy"))
    sampled = random.sample(files, min(N_PER_CLASS, len(files)))

    out_class = DST_ROOT / class_dir.name
    out_class.mkdir(parents=True, exist_ok=True)

    for f in sampled:
        shutil.copy2(f, out_class / f.name)

    print(f"{class_dir.name:12s} selected={len(sampled)}")