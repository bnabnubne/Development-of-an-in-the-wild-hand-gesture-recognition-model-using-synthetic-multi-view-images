import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CSV_PATH = Path("./metadata/rgb_skeleton_multiview.csv")
CKPT_PATH = Path("./results/rgb_skeleton_fusion/best_rgb_skeleton_fusion.pt")
OUT_DIR = Path("./results/rgb_skeleton_fusion_eval")

BATCH_SIZE = 32
NUM_WORKERS = 0

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

EVAL_MODE = "avg_all_cams"
SINGLE_CAM_ID = 0


class RGBSkeletonEvalDataset(Dataset):
    def __init__(self, df, cam_cols, label_to_idx, image_size, skeleton_mode):
        self.df = df.reset_index(drop=True)
        self.cam_cols = cam_cols
        self.label_to_idx = label_to_idx
        self.skeleton_mode = skeleton_mode
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = [self.load_image(row[col]) for col in self.cam_cols]
        images = torch.stack(images, dim=0)

        if self.skeleton_mode == "view_aligned":
            skeletons = []
            for cam_idx in range(len(self.cam_cols)):
                skeleton = np.load(row[f"skeleton_cam_{cam_idx}_path"]).astype(np.float32)
                skeletons.append(torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32))
            skeletons = torch.stack(skeletons, dim=0)
        elif self.skeleton_mode == "original":
            skeleton_path = row["skeleton_orig_path"] if "skeleton_orig_path" in row else row["skeleton_path"]
            skeleton = np.load(skeleton_path).astype(np.float32)
            skeleton = torch.tensor(skeleton.reshape(21, 3), dtype=torch.float32)
            skeletons = skeleton.unsqueeze(0).repeat(len(self.cam_cols), 1, 1)
        else:
            raise ValueError(f"Unknown skeleton_mode: {self.skeleton_mode}")

        label = torch.tensor(self.label_to_idx[row["action"]], dtype=torch.long)

        return images, skeletons, label


class SkeletonEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]


class RGBEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class RGBSkeletonFusionNet(nn.Module):
    def __init__(self, num_classes, rgb_feature_dim, skel_hidden_dim, fusion_dim, dropout):
        super().__init__()
        self.rgb_encoder = RGBEncoder(rgb_feature_dim)
        self.skel_encoder = SkeletonEncoder(3, skel_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feature_dim + skel_hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image, skeleton):
        rgb_feat = self.rgb_encoder(image)
        skel_feat = self.skel_encoder(skeleton)
        fused = torch.cat([rgb_feat, skel_feat], dim=1)
        return self.classifier(fused)


def evaluate(model, loader, eval_mode, single_cam_id):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, skeletons, label in loader:
            images = images.to(DEVICE)
            skeletons = skeletons.to(DEVICE)
            label = label.to(DEVICE)

            if eval_mode == "single_cam":
                logits = model(images[:, single_cam_id], skeletons[:, single_cam_id])
            elif eval_mode == "avg_all_cams":
                logits_sum = None
                for cam_idx in range(images.shape[1]):
                    logits_cam = model(images[:, cam_idx], skeletons[:, cam_idx])
                    logits_sum = logits_cam if logits_sum is None else logits_sum + logits_cam
                logits = logits_sum / images.shape[1]
            else:
                raise ValueError(f"Unknown EVAL_MODE: {eval_mode}")

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())

    return accuracy_score(all_targets, all_preds), all_targets, all_preds


def save_confmat(cm, class_names, out_path):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            if value == 0:
                continue
            ax.text(j, i, f"{value:.1f}", ha="center", va="center")

    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    cfg = ckpt["config"]

    df = pd.read_csv(CSV_PATH)
    if cfg.get("merge_thumb", False):
        df["action"] = df["action"].replace({"thumbup": "thumb", "thumbdown": "thumb"})

    test_df = df[df["split"] == "test"].copy()
    cam_cols = cfg["cam_cols"]
    skeleton_mode = cfg.get("skeleton_mode", "original")

    ds = RGBSkeletonEvalDataset(test_df, cam_cols, label_to_idx, cfg["image_size"], skeleton_mode)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = RGBSkeletonFusionNet(
        num_classes=len(label_to_idx),
        rgb_feature_dim=cfg["rgb_feature_dim"],
        skel_hidden_dim=cfg["skel_hidden_dim"],
        fusion_dim=cfg["fusion_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, y_true, y_pred = evaluate(model, loader, EVAL_MODE, SINGLE_CAM_ID)
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    suffix = EVAL_MODE if EVAL_MODE == "avg_all_cams" else f"cam{SINGLE_CAM_ID}"
    (OUT_DIR / f"report_{suffix}.txt").write_text(report, encoding="utf-8")
    save_confmat(cm, class_names, OUT_DIR / f"confmat_{suffix}.png")

    summary = {
        "eval_mode": EVAL_MODE,
        "single_cam_id": SINGLE_CAM_ID,
        "test_acc": acc,
        "classes": label_to_idx,
        "csv_path": str(CSV_PATH),
        "ckpt_path": str(CKPT_PATH),
        "skeleton_mode": skeleton_mode,
    }
    with open(OUT_DIR / f"summary_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== RGB+SKELETON TEST SUMMARY =====")
    print("Eval mode:", EVAL_MODE)
    print("Test acc :", acc)
    print("Saved to :", OUT_DIR)


if __name__ == "__main__":
    main()
