import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

from gru_models import SingleViewGRUClassifier, MultiViewGRUClassifier

# ===================== CONFIG =====================
INTERNAL_CSV = "./dataset/multiview_blender_metadata.csv"
DATASET2_CSV = "./test/test_metadata.csv"

SINGLE_CKPT = "singleview_gru_best.pth"
MULTI_CKPT = "multiview_gru_best.pth"

SAVE_DIR = Path("./confmat_final")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32


# ===================== DATASET =====================
class InternalSingleDataset(Dataset):
    def __init__(self, csv_path, split="test", cam_col="cam_0_path"):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.cam_col = cam_col

        labels = sorted(self.df["action"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row[self.cam_col]).astype(np.float32).reshape(1, -1)
        y = self.label_to_idx[row["action"]]
        return torch.tensor(x), torch.tensor(y)


class InternalMultiDataset(Dataset):
    def __init__(self, csv_path, split="test"):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        labels = sorted(self.df["action"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        def load(p): return np.load(p).astype(np.float32).reshape(1, -1)

        x0 = load(row["cam_0_path"])
        x2 = load(row["cam_2_path"])
        x4 = load(row["cam_4_path"])
        x5 = load(row["cam_5_path"])
        y = self.label_to_idx[row["action"]]

        return torch.tensor(x0), torch.tensor(x2), torch.tensor(x4), torch.tensor(x5), torch.tensor(y)


class Dataset2Multi(Dataset):
    def __init__(self, csv_path, label_to_idx):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        def load(p): return np.load(p).astype(np.float32).reshape(1, -1)

        x0 = load(row["cam_0_path"])
        x2 = load(row["cam_2_path"])
        x4 = load(row["cam_4_path"])
        x5 = load(row["cam_5_path"])
        y = self.label_to_idx[row["action"]]

        return torch.tensor(x0), torch.tensor(x2), torch.tensor(x4), torch.tensor(x5), torch.tensor(y)


# ===================== SAVE CONF MATRIX =====================
def save_confmat(y_true, y_pred, labels, title, name):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{name}.png", dpi=200)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    with open(SAVE_DIR / f"{name}.txt", "w") as f:
        f.write(report)

    acc = accuracy_score(y_true, y_pred)

    print(f"\n===== {title} =====")
    print(f"Acc: {acc:.4f}")
    print(report)


# ===================== MAIN =====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== LOAD CKPT =====
    ckpt_s = torch.load(SINGLE_CKPT, map_location=device)
    ckpt_m = torch.load(MULTI_CKPT, map_location=device)

    label_to_idx = ckpt_s["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]

    # ===== MODELS =====
    model_s = SingleViewGRUClassifier(num_classes=len(labels)).to(device)
    model_s.load_state_dict(ckpt_s["model_state_dict"])
    model_s.eval()

    model_m = MultiViewGRUClassifier(num_classes=len(labels)).to(device)
    model_m.load_state_dict(ckpt_m["model_state_dict"])
    model_m.eval()

    # ===================== 1. INTERNAL SINGLE =====================
    ds = InternalSingleDataset(INTERNAL_CSV, "test", cam_col="cam_4_path")
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model_s(x)
            pred = logits.argmax(1)

            y_true += y.tolist()
            y_pred += pred.cpu().tolist()

    save_confmat(y_true, y_pred, labels,
                 "Single-view: Cam_0 -> Cam_4",
                 "single_internal")

    # ===================== 2. INTERNAL MULTI =====================
    ds = InternalMultiDataset(INTERNAL_CSV, "test")
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x0, x2, x4, x5, y in loader:
            x0, x2, x4, x5 = x0.to(device), x2.to(device), x4.to(device), x5.to(device)

            _, _, _, _, l0, l2, l4, l5 = model_m(x0, x2, x4, x5)
            pred = l4.argmax(1)

            y_true += y.tolist()
            y_pred += pred.cpu().tolist()

    save_confmat(y_true, y_pred, labels,
                 "Multi-view: 4-view -> Cam_4",
                 "multi_internal")

    # ===================== 3. DATASET2 SINGLE =====================
    ds = Dataset2Multi(DATASET2_CSV, label_to_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x0, _, _, _, y in loader:
            x0 = x0.to(device)
            logits, _ = model_s(x0)
            pred = logits.argmax(1)

            y_true += y.tolist()
            y_pred += pred.cpu().tolist()

    save_confmat(y_true, y_pred, labels,
                 "Single-view -> Dataset2",
                 "single_dataset2")

    # ===================== 4. DATASET2 MULTI =====================
    y_true, y_pred = [], []
    with torch.no_grad():
        for x0, x2, x4, x5, y in loader:
            x0, x2, x4, x5 = x0.to(device), x2.to(device), x4.to(device), x5.to(device)

            _, _, _, _, l0, l2, l4, l5 = model_m(x0, x2, x4, x5)
            logits = (l0 + l2 + l4 + l5) / 4.0
            pred = logits.argmax(1)

            y_true += y.tolist()
            y_pred += pred.cpu().tolist()

    save_confmat(y_true, y_pred, labels,
                 "Multi-view -> Dataset2",
                 "multi_dataset2")


if __name__ == "__main__":
    main()