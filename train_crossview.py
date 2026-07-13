import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# CONFIG
# =========================
CSV_PATH = "./metadata/salux_multiview.csv"
OUT_ROOT = Path("./results/crossview")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4 schema cross-view
EXPERIMENTS = [
    {
        "name": "train_cam024_test_cam5",
        "train_views": ["cam_0_path", "cam_2_path", "cam_4_path"],
        "test_view": "cam_5_path",
    },
    {
        "name": "train_cam025_test_cam4",
        "train_views": ["cam_0_path", "cam_2_path", "cam_5_path"],
        "test_view": "cam_4_path",
    },
    {
        "name": "train_cam045_test_cam2",
        "train_views": ["cam_0_path", "cam_4_path", "cam_5_path"],
        "test_view": "cam_2_path",
    },
    {
        "name": "train_cam245_test_cam0",
        "train_views": ["cam_2_path", "cam_4_path", "cam_5_path"],
        "test_view": "cam_0_path",
    },
]

# =========================
# SEED
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================
# DATASET
# =========================
class CrossViewTrainDataset(Dataset):
    """
    Train trên 3 view. Mỗi sample sẽ bung thành 3 sample con:
    cùng label, khác input view.
    """
    def __init__(self, df: pd.DataFrame, label_to_idx=None, train_views=None):
        self.samples = []
        self.train_views = train_views

        if label_to_idx is None:
            labels = sorted(df["action"].unique().tolist())
            self.label_to_idx = {lb: i for i, lb in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        for _, row in df.iterrows():
            for view_col in train_views:
                self.samples.append({
                    "action": row["action"],
                    "sample_id": row["sample_id"],
                    "input_path": row[view_col],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        x = np.load(row["input_path"]).astype(np.float32)   # (21,2)
        x = x.reshape(-1)                                   # (42,)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,42)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y


class CrossViewEvalDataset(Dataset):
    """
    Eval chỉ trên 1 view target.
    """
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, view_col: str):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.view_col = view_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row[self.view_col]).astype(np.float32)
        x = x.reshape(-1)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)
        return x, y

# =========================
# MODEL
# =========================
class SingleViewGRU2D(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, num_layers=1, num_classes=7, dropout=0.0):
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
        logits = self.fc(feat)
        return logits

# =========================
# UTILS
# =========================
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

    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

# =========================
# RUN EXPERIMENTS
# =========================
all_results = {}

for exp in EXPERIMENTS:
    exp_name = exp["name"]
    train_views = exp["train_views"]
    test_view = exp["test_view"]

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("train_views:", train_views)
    print("test_view  :", test_view)
    print("=" * 80)

    out_dir = OUT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CrossViewTrainDataset(train_df, train_views=train_views)
    label_to_idx = train_ds.label_to_idx

    val_ds = CrossViewEvalDataset(val_df, label_to_idx=label_to_idx, view_col=test_view)
    test_ds = CrossViewEvalDataset(test_df, label_to_idx=label_to_idx, view_col=test_view)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SingleViewGRU2D(
        input_dim=42,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=len(label_to_idx),
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_path = out_dir / "best_crossview.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc, _, _ = evaluate(model, val_loader, DEVICE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "config": {
                    "input_dim": 42,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "train_views": train_views,
                    "test_view": test_view,
                }
            }, best_path)

        print(f"Epoch {epoch:03d} | loss={running_loss:.4f} | val_acc={val_acc:.4f}")

    print("Best val acc:", best_val_acc)

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_acc, _, _ = evaluate(model, test_loader, DEVICE)

    summary = {
        "experiment": exp_name,
        "train_views": train_views,
        "test_view": test_view,
        "best_val_acc": best_val_acc,
        "salux_test_acc": test_acc,
    }

    with open(out_dir / "summary_train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    all_results[exp_name] = summary

with open(OUT_ROOT / "summary_all_train.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\nDONE")
print("Saved to:", OUT_ROOT)