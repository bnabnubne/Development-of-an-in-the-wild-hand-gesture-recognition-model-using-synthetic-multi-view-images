import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from train_rgb_improved import DEVICE, RGBOnlyNet, build_transforms
from test_mvconsistency_skeleton_5cls import SingleViewGRU3D


MODEL_ROOT = Path(__file__).resolve().parent
APRIL_ROOT = MODEL_ROOT.parent.parent / "April"
SKELETON_CKPT = (
    APRIL_ROOT / "results" / "mv_consistency_anchor_8cam_model_lambda0.3"
    / "best_mv_consistency_anchor.pt"
)
DATASETS = {
    "salux": MODEL_ROOT / "metadata" / "salux_original_rgb_5cls.csv",
    "droh": MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv",
}
RGB_VARIANT = "pretrained_handcentric"
SKELETON_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


class MultimodalDataset(Dataset):
    def __init__(self, df, label_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        normal = self.transform(image)
        flipped = self.transform(ImageOps.mirror(image))
        skeleton = np.load(row["skeleton_path"]).astype(np.float32).reshape(21, 3)
        return normal, flipped, torch.from_numpy(skeleton), self.label_to_idx[row["action"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    args = parser.parse_args()

    rgb_dir = MODEL_ROOT / "results" / f"rgb_{RGB_VARIANT}"
    rgb_ckpt = torch.load(rgb_dir / "best.pt", map_location=DEVICE)
    labels5 = rgb_ckpt["label_to_idx"]
    _, transform = build_transforms(rgb_ckpt["config"]["variant"])
    rgb_model = RGBOnlyNet(len(labels5)).to(DEVICE)
    rgb_model.load_state_dict(rgb_ckpt["model_state_dict"])
    rgb_model.eval()

    sk_ckpt = torch.load(SKELETON_CKPT, map_location=DEVICE)
    cfg = sk_ckpt["config"]
    labels7 = sk_ckpt["label_to_idx"]
    sk_model = SingleViewGRU3D(
        input_dim=int(cfg.get("input_dim", 3)), hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]), num_classes=len(labels7),
        dropout=float(cfg["dropout"]),
    ).to(DEVICE)
    sk_model.load_state_dict(sk_ckpt["model_state_dict"])
    sk_model.eval()

    df = pd.read_csv(DATASETS[args.dataset])
    df = df[df["action"].isin(labels5)].copy()
    loader = DataLoader(MultimodalDataset(df, labels5, transform), batch_size=32,
                        shuffle=False, num_workers=0)
    keep7 = torch.tensor([labels7[name] for name, _ in sorted(labels5.items(), key=lambda x: x[1])],
                         device=DEVICE)
    targets, rgb_probs_all, sk_probs_all, sk_full_preds = [], [], [], []
    with torch.no_grad():
        for normal, flipped, skeleton, label in loader:
            rgb_logits = (rgb_model(normal.to(DEVICE)) + rgb_model(flipped.to(DEVICE))) / 2
            sk_logits, _ = sk_model(skeleton.to(DEVICE))
            rgb_probs_all.append(torch.softmax(rgb_logits, 1).cpu())
            sk_probs_all.append(torch.softmax(sk_logits.index_select(1, keep7), 1).cpu())
            sk_full_preds.extend(sk_logits.argmax(1).cpu().tolist())
            targets.extend(label.tolist())
    rgb_probs = torch.cat(rgb_probs_all)
    sk_probs = torch.cat(sk_probs_all)

    results = {
        "skeleton_full_7class_head": accuracy_score(targets, sk_full_preds),
        "skeleton_restricted_5class": accuracy_score(targets, sk_probs.argmax(1).tolist()),
        "rgb_flip_tta": accuracy_score(targets, rgb_probs.argmax(1).tolist()),
        "late_fusion": {},
    }
    predictions = {}
    for weight in SKELETON_WEIGHTS:
        pred = (weight * sk_probs + (1 - weight) * rgb_probs).argmax(1).tolist()
        key = f"skeleton_{weight:.2f}"
        results["late_fusion"][key] = accuracy_score(targets, pred)
        predictions[key] = pred

    class_names = [x for x, _ in sorted(labels5.items(), key=lambda x: x[1])]
    best_key = max(results["late_fusion"], key=results["late_fusion"].get)
    report = classification_report(
        targets, predictions[best_key], labels=range(len(class_names)),
        target_names=class_names, digits=4, zero_division=0,
    )
    out_dir = MODEL_ROOT / "results" / "robust_late_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report_{args.dataset}_{best_key}.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / f"confmat_{args.dataset}_{best_key}.npy", confusion_matrix(
        targets, predictions[best_key], labels=range(len(class_names))
    ))
    summary = {
        "dataset": args.dataset, "rows": len(df), "rgb_variant": RGB_VARIANT,
        "best_late_fusion_weight": best_key, "results": results,
        "note": "Weight sweep is an ablation on external labels, not a deployment-tuned parameter.",
    }
    (out_dir / f"summary_{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
