import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = Path("./metadata/droh_baseline.csv")
OUT_ROOT = Path("./test/droh_aligned3d")
OUT_CSV = Path("./metadata/droh_aligned3d.csv")

REMOVE_OLD = True

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# =========================================================
# ALIGN
# =========================================================
def align_hand_3d(kp: np.ndarray) -> np.ndarray:
    """
    Input:  kp shape (21,3)
    Output: kp aligned shape (21,3)
    Quy ước:
      - wrist về gốc
      - trục y: wrist -> middle_mcp
      - trục x: ngang lòng bàn tay (index <-> pinky), trực giao với y
      - trục z: pháp tuyến lòng bàn tay
      - ép z cùng chiều dương
    """
    kp = kp.astype(np.float32).copy()

    # 1) center wrist
    kp = kp - kp[WRIST:WRIST+1]

    index = kp[INDEX_MCP]
    middle = kp[MIDDLE_MCP]
    pinky = kp[PINKY_MCP]

    # 2) y-axis = wrist -> middle_mcp
    y = middle.copy()
    y_norm = np.linalg.norm(y) + 1e-8
    y = y / y_norm

    # 3) x-axis = pinky - index, rồi trực giao với y
    x = pinky - index
    x = x - np.dot(x, y) * y
    x_norm = np.linalg.norm(x) + 1e-8
    x = x / x_norm

    # 4) z-axis = x cross y
    z = np.cross(x, y)
    z_norm = np.linalg.norm(z) + 1e-8
    z = z / z_norm

    # 5) ép z cùng chiều dương để nhất quán
    if z[2] < 0:
        x = -x
        z = -z

    # 6) re-orthogonalize lại cho chắc
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-8)

    # 7) rotation matrix: columns = [x y z]
    R = np.stack([x, y, z], axis=1)   # world -> canonical via kp @ R
    kp_aligned = kp @ R

    # 8) scale theo wrist -> middle_mcp = 1
    scale = np.linalg.norm(kp_aligned[MIDDLE_MCP]) + 1e-8
    kp_aligned = kp_aligned / scale

    return kp_aligned.astype(np.float32)

# =========================================================
# MAIN
# =========================================================
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing INPUT_CSV: {INPUT_CSV}")

    if REMOVE_OLD and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(INPUT_CSV)

    rows = []
    saved = 0
    skipped = 0

    for _, row in df.iterrows():
        action = row["action"]
        sample_id = row["sample_id"]
        in_path = Path(row["input_path"])

        if not in_path.exists():
            print(f"[MISS] {in_path}")
            skipped += 1
            continue

        kp = np.load(in_path).astype(np.float32)
        if kp.shape != (21, 3):
            print(f"[SKIP] bad shape {kp.shape}: {in_path}")
            skipped += 1
            continue

        kp_aligned = align_hand_3d(kp)

        out_dir = OUT_ROOT / action
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sample_id}.npy"
        np.save(out_path, kp_aligned)

        rows.append({
            "action": action,
            "sample_id": sample_id,
            "input_path": str(out_path),
            "split": "test"
        })
        saved += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print("\nDONE")
    print("Saved   :", saved)
    print("Skipped :", skipped)
    print("OUT_ROOT:", OUT_ROOT)
    print("OUT_CSV :", OUT_CSV)

if __name__ == "__main__":
    main()