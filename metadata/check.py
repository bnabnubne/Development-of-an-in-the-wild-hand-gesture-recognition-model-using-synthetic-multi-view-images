import pandas as pd

a = pd.read_csv("./metadata/droh_baseline.csv")
b = pd.read_csv("./metadata/droh_multiview.csv")

a_test = set(a[a["split"]=="test"]["sample_id"])
b_test = set(b[b["split"]=="test"]["sample_id"])

print("same test set:", a_test == b_test)
print("diff:", len(a_test.symmetric_difference(b_test)))