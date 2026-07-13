import numpy as np

pts = np.load(
"./templates/ok_template_joints.npy"
)

CHAINS = {
    "thumb":[1,2,3,4],
    "index":[5,6,7,8],
    "middle":[9,10,11,12],
    "ring":[13,14,15,16],
    "pinky":[17,18,19,20]
}

for name, chain in CHAINS.items():

    print()
    print(name)

    prev = 0

    for j in chain:

        L = np.linalg.norm(
            pts[j] - pts[prev]
        )

        print(
            f"{prev}->{j}",
            round(float(L),4)
        )

        prev = j