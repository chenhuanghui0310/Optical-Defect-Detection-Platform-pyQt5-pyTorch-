# -*- coding: utf-8 -*-
"""
Merge real labeled data into dataset/images/Real_Data.

Changes in this version:
1. Keep Roboflow dust/hair local boxes.
2. Auto-localize real scratch regions instead of writing full-image weak labels.
3. Deduplicate label lines before saving.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOFLOW_DIR = PROJECT_ROOT / "dataset" / "My First Project.yolov5pytorch" / "train"
SCRATCH_DIRS = [
    PROJECT_ROOT / "dataset" / "real_data" / "20260127-划痕",
    PROJECT_ROOT / "dataset" / "real_data" / "20260211-划痕",
]
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "images" / "Real_Data"
ROBOFLOW_CLS_MAP = {0: 3, 1: 4}
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def ensure_dirs():
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)


def _deduplicate_yolo_lines(lines: list[str], decimals: int = 6) -> list[str]:
    unique = []
    seen = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        key = (
            int(float(parts[0])),
            round(float(parts[1]), decimals),
            round(float(parts[2]), decimals),
            round(float(parts[3]), decimals),
            round(float(parts[4]), decimals),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            f"{key[0]} {key[1]:.{decimals}f} {key[2]:.{decimals}f} {key[3]:.{decimals}f} {key[4]:.{decimals}f}"
        )
    return unique


def _load_image_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _localize_real_scratch_boxes(img: np.ndarray) -> list[str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, np.ones((17, 17), np.uint8))
    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blackhat, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))

    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    labels = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 80:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect < 2.0 and area < 600:
            continue
        pad = max(6, int(round(max(bw, bh) * 0.08)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        xc = (x1 + x2) / 2.0 / w
        yc = (y1 + y2) / 2.0 / h
        wn = (x2 - x1) / float(w)
        hn = (y2 - y1) / float(h)
        labels.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return _deduplicate_yolo_lines(labels)


def process_roboflow_data():
    images_dir = ROBOFLOW_DIR / "images"
    labels_dir = ROBOFLOW_DIR / "labels"
    if not images_dir.exists():
        print(f"[WARN] Roboflow images dir not found: {images_dir}")
        return 0

    count = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        remapped_lines = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            old_cls = int(parts[0])
            new_cls = ROBOFLOW_CLS_MAP.get(old_cls)
            if new_cls is None:
                continue
            remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        remapped_lines = _deduplicate_yolo_lines(remapped_lines)
        if not remapped_lines:
            continue

        out_name = f"real_dusthair_{count:04d}"
        out_img = OUTPUT_DIR / "images" / f"{out_name}{img_path.suffix}"
        out_lbl = OUTPUT_DIR / "labels" / f"{out_name}.txt"
        shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

        cls_counts = {}
        for line in remapped_lines:
            cls_id = int(line.split()[0])
            cls_counts[cls_id] = cls_counts.get(cls_id, 0) + 1
        main_cls = max(cls_counts, key=cls_counts.get)
        defect_name = {3: "灰尘污染", 4: "头发丝遮挡"}.get(main_cls, "灰尘污染")
        meta = {
            "defect_type": defect_name,
            "sim_type": "real",
            "source": "roboflow_labeled",
            "original_file": img_path.name,
        }
        (OUTPUT_DIR / "images" / f"{out_name}_参数.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1

    print(f"[OK] Roboflow dust/hair: {count}")
    return count


def process_scratch_data():
    count = 0
    weak_fallback = 0
    for scratch_dir in SCRATCH_DIRS:
        if not scratch_dir.exists():
            print(f"[WARN] Scratch dir not found: {scratch_dir}")
            continue

        for img_path in sorted(scratch_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            img = _load_image_unicode(img_path)
            if img is None:
                continue

            out_name = f"real_scratch_{count:04d}"
            out_img = OUTPUT_DIR / "images" / f"{out_name}{img_path.suffix}"
            out_lbl = OUTPUT_DIR / "labels" / f"{out_name}.txt"
            shutil.copy2(img_path, out_img)

            labels = _localize_real_scratch_boxes(img)
            if not labels:
                h, w = img.shape[:2]
                labels = [f"0 0.500000 0.500000 1.000000 1.000000"]
                weak_fallback += 1

            out_lbl.write_text("\n".join(labels) + "\n", encoding="utf-8")
            meta = {
                "defect_type": "表面划痕",
                "sim_type": "real",
                "source": "real_photo",
                "original_file": img_path.name,
                "scratch_boxes_auto": labels[0] != "0 0.500000 0.500000 1.000000 1.000000",
            }
            (OUTPUT_DIR / "images" / f"{out_name}_参数.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            count += 1

    print(f"[OK] Scratch real photos: {count}, weak fallback: {weak_fallback}")
    return count


def main():
    print("=" * 60)
    print("Merge real data into Real_Data")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")

    ensure_dirs()
    total = 0
    total += process_roboflow_data()
    total += process_scratch_data()

    print(f"Total processed: {total}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
