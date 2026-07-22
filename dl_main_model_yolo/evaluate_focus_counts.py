"""Evaluate class, localization and exact-count performance on focus_v2 eval data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {0: "scratch", 1: "bump", 2: "dent", 3: "dust", 4: "hair"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local-defect counts and localization.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True, help="focus_v2/eval_yolo directory")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def load_labels(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id, xc, yc, bw, bh = map(float, parts[:5])
        boxes.append((int(cls_id), [(xc - bw / 2) * width, (yc - bh / 2) * height,
                                   (xc + bw / 2) * width, (yc + bh / 2) * height]))
    return boxes


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    model = YOLO(args.weights)
    totals = defaultdict(lambda: {"gt": 0, "pred": 0, "matched": 0, "count_exact": 0, "images": 0})
    images = []

    for split in ("train", "val", "test"):
        label_dir = dataset / "labels" / split
        image_dir = dataset / "images" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = next((image_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTENSIONS
                               if (image_dir / f"{label_path.stem}{ext}").exists()), None)
            if image_path is None:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            gt = load_labels(label_path, width, height)
            result = model.predict(str(image_path), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                   device=args.device, verbose=False)[0]
            predictions = [(int(box.cls.item()), box.xyxy[0].tolist()) for box in result.boxes or []]

            per_image = {"image": str(image_path), "classes": {}}
            for cls_id, name in CLASS_NAMES.items():
                gt_boxes = [box for label, box in gt if label == cls_id]
                pred_boxes = [box for label, box in predictions if label == cls_id]
                if not gt_boxes and not pred_boxes:
                    continue
                unmatched = set(range(len(pred_boxes)))
                matched = 0
                for gt_box in gt_boxes:
                    best = max(unmatched, key=lambda idx: iou(gt_box, pred_boxes[idx]), default=None)
                    if best is not None and iou(gt_box, pred_boxes[best]) >= args.match_iou:
                        unmatched.remove(best)
                        matched += 1
                row = totals[name]
                row["gt"] += len(gt_boxes)
                row["pred"] += len(pred_boxes)
                row["matched"] += matched
                row["images"] += 1
                row["count_exact"] += int(len(gt_boxes) == len(pred_boxes))
                per_image["classes"][name] = {"gt": len(gt_boxes), "pred": len(pred_boxes), "matched": matched}
            images.append(per_image)

    summary = {
        name: {
            **values,
            "recall": values["matched"] / values["gt"] if values["gt"] else 0.0,
            "precision": values["matched"] / values["pred"] if values["pred"] else 0.0,
            "count_exact_rate": values["count_exact"] / values["images"] if values["images"] else 0.0,
        }
        for name, values in totals.items()
    }
    report = {"settings": vars(args), "summary": summary, "images": images}
    output = Path(args.output_json).resolve() if args.output_json else dataset / "focus_count_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
