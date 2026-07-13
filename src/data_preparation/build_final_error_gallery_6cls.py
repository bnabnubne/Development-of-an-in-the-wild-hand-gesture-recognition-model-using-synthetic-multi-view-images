"""Export real RGB failure cases of the final five-seed model for defense slides."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "results/droh_postfilter_audit_605_6cls/predictions_low_lr_consistency_5ensemble.csv"
MEDIA_METADATA = Path("./test/mediapipe_metadata.csv")
OUT = ROOT / "results/final_5seed_error_gallery_postfilter_605_6cls"
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def safe_name(value):
    return str(value).lower().replace("the-finger", "the_finger").replace(" ", "_")


def annotate(image, landmarks, true_class, predicted_class, confidence, sample_id, rgb_available=True):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), (40, 220, 255),
                 max(2, width // 420), cv2.LINE_AA)
    for point in pixels:
        cv2.circle(canvas, tuple(point), max(3, width // 180), (40, 65, 255), -1, cv2.LINE_AA)

    panel_height = max(86, int(height * 0.16))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (10, 12, 18), -1)
    canvas = cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0)
    scale = max(0.55, min(1.05, width / 900))
    cv2.putText(canvas, f"GT: {true_class}  |  Pred: {predicted_class}",
                (18, int(panel_height * 0.43)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (70, 90, 255), 2, cv2.LINE_AA)
    source_note = "RGB + MediaPipe" if rgb_available else "Skeleton only - source RGB unavailable"
    cv2.putText(canvas, f"Confidence: {confidence * 100:.1f}%  |  {sample_id}  |  {source_note}",
                (18, int(panel_height * 0.78)), cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.72, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def thumbnail(image, width=300, height=230):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    output = np.full((height, width, 3), 20, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def contact_sheet(images, title, columns=4, limit=12):
    selected = images[:limit]
    if not selected:
        return None
    tile_w, tile_h, header = 300, 230, 58
    rows = int(np.ceil(len(selected) / columns))
    sheet = np.full((header + rows * tile_h, columns * tile_w, 3), 18, dtype=np.uint8)
    cv2.putText(sheet, title, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (245, 245, 245), 2, cv2.LINE_AA)
    for index, image in enumerate(selected):
        row, col = divmod(index, columns)
        sheet[header + row * tile_h:header + (row + 1) * tile_h,
              col * tile_w:(col + 1) * tile_w] = thumbnail(image, tile_w, tile_h)
    return sheet


def main():
    predictions = pd.read_csv(PREDICTIONS)
    media = pd.read_csv(MEDIA_METADATA)
    media = media[media.status == "ok"].copy()
    media["sample_id"] = media.raw_path.map(lambda value: Path(value).stem)
    frame = predictions.merge(
        media[["sample_id", "image_path", "raw_path", "handedness"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    errors = frame[~frame.correct.astype(bool)].copy()
    if len(predictions) != 605 or len(errors) != 74:
        raise ValueError(f"Expected 605 predictions and 74 final errors, got {len(predictions)} and {len(errors)}")
    if errors[["image_path", "raw_path"]].isna().any().any():
        raise ValueError("Some failure cases are missing RGB or MediaPipe paths")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    pair_counts = (
        errors.groupby(["true_class", "predicted_class"]).size()
        .rename("count").reset_index().sort_values("count", ascending=False)
    )
    pair_counts["share_of_74_errors"] = pair_counts["count"] / len(errors)
    pair_counts.to_csv(OUT / "confusion_pairs.csv", index=False)

    exported_rows = []
    all_pair_sheets = []
    for pair_rank, pair in enumerate(pair_counts.itertuples(index=False), start=1):
        pair_name = f"{pair_rank:02d}_{safe_name(pair.true_class)}_to_{safe_name(pair.predicted_class)}__n{pair.count}"
        pair_dir = OUT / pair_name
        original_dir = pair_dir / "original"
        annotated_dir = pair_dir / "annotated"
        original_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        subset = errors[
            (errors.true_class == pair.true_class) &
            (errors.predicted_class == pair.predicted_class)
        ].sort_values("confidence", ascending=False)
        annotated_images = []
        for index, row in enumerate(subset.itertuples(index=False), start=1):
            source = Path(row.image_path)
            raw_path = Path(row.raw_path)
            if not raw_path.is_file():
                raise FileNotFoundError(f"Cannot read skeleton {raw_path}")
            image = cv2.imread(str(source)) if source.is_file() else None
            rgb_available = image is not None
            if not rgb_available:
                image = np.full((700, 900, 3), (22, 24, 30), dtype=np.uint8)
            suffix = "" if rgb_available else "__skeleton_only"
            stem = f"{index:02d}_{safe_name(row.sample_id)}__conf_{row.confidence * 100:.1f}{suffix}"
            original_path = original_dir / f"{stem}{source.suffix.lower()}" if rgb_available else None
            annotated_path = annotated_dir / f"{stem}.jpg"
            if rgb_available:
                shutil.copy2(source, original_path)
            annotated = annotate(
                image, np.load(raw_path, allow_pickle=False), row.true_class,
                row.predicted_class, row.confidence, row.sample_id, rgb_available,
            )
            cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
            annotated_images.append(annotated)
            exported_rows.append({
                "pair_rank": pair_rank,
                "true_class": row.true_class,
                "predicted_class": row.predicted_class,
                "confidence": row.confidence,
                "sample_id": row.sample_id,
                "rgb_available": rgb_available,
                "source_image": str(source),
                "original_copy": str(original_path) if original_path else "",
                "annotated_image": str(annotated_path),
            })
        sheet = contact_sheet(
            annotated_images,
            f"GT {pair.true_class} -> Pred {pair.predicted_class} | {pair.count} errors",
        )
        if sheet is not None:
            sheet_path = pair_dir / "contact_sheet.jpg"
            cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if pair_rank <= 6:
                all_pair_sheets.append((pair, sheet))

    pd.DataFrame(exported_rows).to_csv(OUT / "all_errors.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    if all_pair_sheets:
        max_width = max(sheet.shape[1] for _, sheet in all_pair_sheets)
        total_height = sum(sheet.shape[0] for _, sheet in all_pair_sheets)
        overview = np.full((total_height, max_width, 3), 18, dtype=np.uint8)
        y = 0
        for _, sheet in all_pair_sheets:
            overview[y:y + sheet.shape[0], :sheet.shape[1]] = sheet
            y += sheet.shape[0]
        cv2.imwrite(str(OUT / "top_6_confusion_pairs_contact_sheet.jpg"), overview,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])

    # Compact 16:9 overview for direct insertion into a defense slide.
    slide = np.full((1080, 1920, 3), (15, 17, 22), dtype=np.uint8)
    cv2.putText(slide, "Representative errors from the final model (top confusion pairs)",
                (42, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (245, 245, 245), 2, cv2.LINE_AA)
    for panel_index, pair in enumerate(pair_counts.head(6).itertuples(index=False)):
        row, col = divmod(panel_index, 3)
        x0, y0 = 30 + col * 630, 88 + row * 490
        cv2.rectangle(slide, (x0, y0), (x0 + 610, y0 + 465), (31, 35, 44), -1)
        title = f"GT {pair.true_class} -> Pred {pair.predicted_class} ({pair.count})"
        cv2.putText(slide, title, (x0 + 15, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.61, (235, 235, 240), 2, cv2.LINE_AA)
        pair_dir = OUT / f"{panel_index + 1:02d}_{safe_name(pair.true_class)}_to_{safe_name(pair.predicted_class)}__n{pair.count}" / "annotated"
        examples = sorted(pair_dir.glob("*.jpg"))[:4]
        for example_index, example_path in enumerate(examples):
            image = cv2.imread(str(example_path))
            tile = thumbnail(image, 285, 195)
            tile_row, tile_col = divmod(example_index, 2)
            left, top = x0 + 13 + tile_col * 295, y0 + 50 + tile_row * 202
            slide[top:top + 195, left:left + 285] = tile
    cv2.imwrite(str(OUT / "slide_top_6_confusions_16x9.jpg"), slide,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    readme = f"""# Final-model DrOh Error Gallery

- Model: five-seed Blender8 + consistency lambda=0.3 probability ensemble.
- DrOh post-filter result: 531/605 correct (87.77%); 74 misclassified samples.
- Folders are ranked by confusion-pair frequency and named `ground_truth_to_prediction`.
- Each pair contains available untouched RGB copies, annotated copies, and a contact sheet.
- Every sample in this post-filter gallery has its current source RGB image.
- `confusion_pairs.csv` contains pair counts; `all_errors.csv` maps every exported image.
- `top_6_confusion_pairs_contact_sheet.jpg` is a ready-to-use visual overview for slides.
- `slide_top_6_confusions_16x9.jpg` is the compact 1920x1080 slide-ready version.

The folder ranking and `confusion_pairs.csv` are the source of truth for the largest
remaining confusion pairs after filtering.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Exported {len(exported_rows)} errors across {len(pair_counts)} confusion pairs to {OUT}")


if __name__ == "__main__":
    main()
