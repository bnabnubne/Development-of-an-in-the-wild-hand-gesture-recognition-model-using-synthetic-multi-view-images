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


MODEL_ROOT = Path(__file__).resolve().parent
DATASETS = {
    "salux": MODEL_ROOT / "metadata" / "salux_original_rgb_5cls.csv",
    "droh": MODEL_ROOT / "metadata" / "droh_rgb_skeleton_5cls.csv",
}


class ExternalRGBDataset(Dataset):
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
        label = self.label_to_idx[row["action"]]
        return normal, flipped, label


def evaluate(model, loader, tta):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for normal, flipped, label in loader:
            logits = model(normal.to(DEVICE))
            if tta:
                logits = (logits + model(flipped.to(DEVICE))) / 2
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(label.tolist())
    return accuracy_score(targets, preds), targets, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="salux")
    args = parser.parse_args()
    out_dir = MODEL_ROOT / "results" / f"rgb_{args.variant}"
    ckpt = torch.load(out_dir / "best.pt", map_location=DEVICE)
    variant = ckpt["config"]["variant"]
    label_to_idx = ckpt["label_to_idx"]
    _, eval_tf = build_transforms(variant)
    df = pd.read_csv(DATASETS[args.dataset])
    df = df[df["action"].isin(label_to_idx)].copy()
    loader = DataLoader(
        ExternalRGBDataset(df, label_to_idx, eval_tf), batch_size=32,
        shuffle=False, num_workers=0,
    )
    model = RGBOnlyNet(len(label_to_idx)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = [x for x, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    results = {}
    for name, tta in [("plain", False), ("flip_tta", True)]:
        acc, y_true, y_pred = evaluate(model, loader, tta)
        report = classification_report(
            y_true, y_pred, labels=range(len(class_names)), target_names=class_names,
            digits=4, zero_division=0,
        )
        (out_dir / f"report_{args.dataset}_{name}.txt").write_text(report, encoding="utf-8")
        np.save(out_dir / f"confmat_{args.dataset}_{name}.npy", confusion_matrix(
            y_true, y_pred, labels=range(len(class_names))
        ))
        results[name] = acc
    summary = {
        "variant": variant,
        "dataset": args.dataset,
        "rows": len(df),
        "accuracy": results,
    }
    (out_dir / f"summary_{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
