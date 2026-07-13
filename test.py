import numpy as np
import matplotlib.pyplot as plt

methods = [
    "3D GRU Baseline",
    "ROT3D",
    "Transformer",
    "Transformer+\nCtr+Triplet",
    "MV-Con\n4cam",
    "MV-SupCon\n4cam",
    "MV-Con\n8cam",
    "MV-SupCon\n8cam"
]

s1_acc = [94.20, 95.73, 96.00, 96.00, 91.56, 93.90, 94.61, 95.32]
s2_acc = [62.96, 69.78, 55.00, 66.00, 67.70, 64.44, 75.85, 72.29]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(11, 4))

bars1 = plt.bar(x - width/2, s1_acc, width, label="S1 / CanonicalSet", color="darkblue")
bars2 = plt.bar(x + width/2, s2_acc, width, label="S2 / HandinWildSet", color="lightblue")

plt.ylabel("Accuracy (%)")
plt.xlabel("Method")
plt.xticks(x, methods, rotation=25, ha="right")
plt.ylim(0, 105)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()
plt.savefig("method_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()