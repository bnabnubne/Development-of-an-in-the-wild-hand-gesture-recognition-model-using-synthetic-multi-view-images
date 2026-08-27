import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

IMAGE_PATH = "./data/raw/FullLabelled/ok/ok (111).jpg"
XY_PATH = "./batch_ok/2d/ok (111).npy"
OUT_PATH = "./batch_ok/overlays/ok (111).png"

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
xy = np.load(XY_PATH).astype(np.float32)

h, w = img.shape[:2]

pts = xy.copy()
pts[:, 0] *= w
pts[:, 1] *= h

plt.figure(figsize=(7, 7))
plt.imshow(img)

for a, b in CONNECTIONS:
    plt.plot(
        [pts[a, 0], pts[b, 0]],
        [pts[a, 1], pts[b, 1]],
        linewidth=2,
        color="lime"
    )

for i, p in enumerate(pts):
    plt.scatter(p[0], p[1], s=25, color="red")
    plt.text(p[0] + 3, p[1] - 3, str(i), color="blue", fontsize=9)

plt.title("Salux MediaPipe 2D Overlay - ok_6990")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.show()

print("Saved:", OUT_PATH)