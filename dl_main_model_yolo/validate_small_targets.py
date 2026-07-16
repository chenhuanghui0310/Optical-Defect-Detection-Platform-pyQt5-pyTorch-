import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAME_TO_ID = {"bump": 1, "dent": 2}
CLASS_ID_TO_NAME = {value: key for key, value in CLASS_NAME_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate small bump/dent targets from manifest.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.10,
        help="Minimum IoU required to count a prediction as a matched small target",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help="Limit exported error records; 0 keeps all errors",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the validation report as UTF-8 JSON",
    )
    return parser.parse_args()


def load_yolo_boxes(label_path: Path) -> list[tuple[int, list[float]]]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, wn, hn = map(float, parts[1:5])
        x1 = xc - wn / 2.0
        y1 = yc - hn / 2.0
        x2 = xc + wn / 2.0
        y2 = yc + hn / 2.0
        boxes.append((cls_id, [x1, y1, x2, y2]))
    return boxes


def bbox_iou(box1, box2) -> float:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def classify_error(expected_cls_id: int, pred_boxes: list[dict], best_same_class_iou: float, match_iou: float) -> str:
    if not pred_boxes:
        return "missed_all"
    if best_same_class_iou >= match_iou:
        return "matched"

    predicted_cls_ids = {int(pred["cls_id"]) for pred in pred_boxes}
    other_small_target_ids = set(CLASS_ID_TO_NAME) - {expected_cls_id}
    if predicted_cls_ids & other_small_target_ids:
        return "confused_small_target"
    if expected_cls_id in predicted_cls_ids:
        return "localized_poorly"
    return "wrong_other_class"


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dataset_root = Path(args.manifest).resolve().parent.parent

    total = 0
    class_correct = 0
    matched = 0
    count_match = 0
    ious = []
    confidences = []
    errors = []
    confusion = {
        "matched": 0,
        "missed_all": 0,
        "confused_small_target": 0,
        "localized_poorly": 0,
        "wrong_other_class": 0,
    }

    for item in manifest:
        image_path = dataset_root / item["image"]
        label_path = dataset_root / item["label"]
        expected_class_name = item["class_name"]
        expected_cls_id = CLASS_NAME_TO_ID[expected_class_name]
        expected_count = int(item.get("expected_count", 1))
        gt_boxes = [box for cls_id, box in load_yolo_boxes(label_path) if cls_id == expected_cls_id]
        image_shape = cv2.imread(str(image_path))
        if image_shape is None:
            errors.append(
                {
                    "image": str(image_path),
                    "expected": expected_class_name,
                    "predicted_labels": [],
                    "error_type": "image_read_failed",
                }
            )
            continue
        h, w = image_shape.shape[:2]
        total += 1

        try:
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
            )
        except Exception:
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device="cpu",
                verbose=False,
                save=False,
            )

        result = results[0]
        names = result.names if isinstance(result.names, dict) else {}
        pred_boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, f"class_{cls_id}"))
                pred_boxes.append(
                    {
                        "cls_id": cls_id,
                        "label": label,
                        "conf": float(box.conf.item()),
                        "box": box.xyxy[0].tolist(),
                    }
                )

        same_class_preds = [p for p in pred_boxes if p["cls_id"] == expected_cls_id]
        if len(same_class_preds) == expected_count:
            count_match += 1

        best_iou = 0.0
        best_conf = 0.0
        for gt_box in gt_boxes:
            for pred in same_class_preds:
                pred_norm = [
                    pred["box"][0] / w,
                    pred["box"][1] / h,
                    pred["box"][2] / w,
                    pred["box"][3] / h,
                ]
                current_iou = bbox_iou(gt_box, pred_norm)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_conf = pred["conf"]

        error_type = classify_error(expected_cls_id, pred_boxes, best_iou, args.match_iou)
        confusion[error_type] = confusion.get(error_type, 0) + 1

        if best_iou >= args.match_iou:
            class_correct += 1
            matched += 1
            ious.append(best_iou)
            confidences.append(best_conf)
        else:
            errors.append(
                {
                    "image": str(image_path),
                    "expected": expected_class_name,
                    "best_same_class_iou": best_iou,
                    "expected_count": expected_count,
                    "predicted_count": len(pred_boxes),
                    "predicted_labels": [p["label"] for p in pred_boxes],
                    "predictions": pred_boxes,
                    "error_type": error_type,
                }
            )

    report = {
        "total": total,
        "class_accuracy": class_correct / total if total else 0.0,
        "count_match_rate": count_match / total if total else 0.0,
        "matched_rate": matched / total if total else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "median_confidence": float(np.median(confidences)) if confidences else 0.0,
        "match_iou": args.match_iou,
        "confusion_breakdown": confusion,
        "confidence_distribution": confidences,
        "errors": errors if args.max_errors <= 0 else errors[: args.max_errors],
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"Saved report to: {output_path}")
    print(report_text)


if __name__ == "__main__":
    main()
