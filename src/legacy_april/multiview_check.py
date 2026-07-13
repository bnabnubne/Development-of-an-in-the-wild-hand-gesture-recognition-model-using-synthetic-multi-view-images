import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SAMPLE_DIR = Path("./test/multiview_blender/ok/IMG_6289")

FILES = ["Cam_0.npy", "Cam_2.npy", "Cam_4.npy", "Cam_5.npy"]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

for fname in FILES:
    pts = np.load(SAMPLE_DIR / fname).astype(np.float32)

    plt.figure(figsize=(5,5))
    plt.scatter(pts[:,0], pts[:,1], s=30)

    for a, b in CONNECTIONS:
        plt.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]])

    for i, (x, y) in enumerate(pts):
        plt.text(x, y, str(i), fontsize=8)

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.title(fname)
    plt.show()