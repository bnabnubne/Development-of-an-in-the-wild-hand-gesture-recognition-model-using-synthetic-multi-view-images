import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

CKPT_PATH = "./results/rot3d_model_6cls/best_rot3d.pt"
SALUX_CSV = "./metadata/salux_baseline.csv"
DROH_CSV  = "./metadata/droh_baseline.csv"

OUT_DIR = Path("./results/rot3d_test_6cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

class SingleView3DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["input_path"]).astype(np.float32)   # (21,3)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

class SingleViewGRU3D(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=1, num_classes=7, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        feat = out[:, -1, :]
        return self.fc(feat)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds

def save_confmat(cm, class_names, out_path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
label_to_idx = ckpt["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}
cfg = ckpt["config"]

model = SingleViewGRU3D(
    input_dim=cfg["input_dim"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    num_classes=len(label_to_idx),
    dropout=cfg["dropout"]
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

salux_df = pd.read_csv(SALUX_CSV)
droh_df = pd.read_csv(DROH_CSV)

salux_df["action"] = salux_df["action"].replace({
    "thumbup": "thumb",
    "thumbdown": "thumb"
})

droh_df["action"] = droh_df["action"].replace({
    "thumbup": "thumb",
    "thumbdown": "thumb"
})
salux_test_df = salux_df[salux_df["split"] == "test"].copy()

salux_ds = SingleView3DDataset(salux_test_df, label_to_idx=label_to_idx)
droh_ds = SingleView3DDataset(droh_df, label_to_idx=label_to_idx)

salux_loader = DataLoader(salux_ds, batch_size=BATCH_SIZE, shuffle=False)
droh_loader = DataLoader(droh_ds, batch_size=BATCH_SIZE, shuffle=False)

salux_acc, y_true_salux, y_pred_salux = evaluate(model, salux_loader, DEVICE)
droh_acc, y_true_droh, y_pred_droh = evaluate(model, droh_loader, DEVICE)

report_salux = classification_report(y_true_salux, y_pred_salux, target_names=class_names, digits=4, zero_division=0)
report_droh = classification_report(y_true_droh, y_pred_droh, target_names=class_names, digits=4, zero_division=0)

cm_salux = confusion_matrix(y_true_salux, y_pred_salux)
cm_droh = confusion_matrix(y_true_droh, y_pred_droh)

(OUT_DIR / "report_salux_test.txt").write_text(report_salux, encoding="utf-8")
(OUT_DIR / "report_droh_test.txt").write_text(report_droh, encoding="utf-8")

save_confmat(cm_salux, class_names, OUT_DIR / "confmat_salux_test.png", "ROT3D - Salux Test")
save_confmat(cm_droh, class_names, OUT_DIR / "confmat_droh_test.png", "ROT3D - DrOh Test")

summary = {
    "salux_test_acc": salux_acc,
    "droh_test_acc": droh_acc
}
with open(OUT_DIR / "summary_test.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("===== ROT3D TEST SUMMARY =====")
print("Salux test acc:", salux_acc)
print("DrOh test acc :", droh_acc)
print("Saved to:", OUT_DIR)