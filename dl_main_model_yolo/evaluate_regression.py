"""Evaluate a YOLO weight against a fixed, deployment-facing regression set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate a weight on fixed OpticalLab regression images.")
    parser.add_argument("--weights", required=True, help="Candidate or baseline YOLO .pt file.")
    parser.add_argument(
        "--manifest",
        default=str(root / "dataset" / "regression" / "optical_lab_v1" / "manifest.json"),
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--output-dir",
        default=str(root / "dl_main_model_yolo" / "runs" / "regression"),
    )
    return parser.parse_args()


def predict(model: YOLO, image_path: Path, args: argparse.Namespace):
    try:
        return model.predict(
            source=str(image_path), conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]
    except Exception:
        return model.predict(
            source=str(image_path), conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device="cpu", verbose=False,
        )[0]


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve() / weights.stem
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = YOLO(str(weights))
    results = []

    for item in manifest["items"]:
        image_path = manifest_path.parent / item["image"]
        result = predict(model, image_path, args)
        names = result.names if isinstance(result.names, dict) else {}
        predictions = []
        for box in result.boxes or []:
            cls_id = int(box.cls.item())
            predictions.append(
                {
                    "label": str(names.get(cls_id, f"class_{cls_id}")),
                    "confidence": round(float(box.conf.item()), 6),
                    "bbox_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )
        predictions.sort(key=lambda value: value["confidence"], reverse=True)
        top_label = predictions[0]["label"] if predictions else None
        expected = item.get("expected_label")
        rendered_path = image_dir / image_path.name
        ok, buffer = cv2.imencode(".png", result.plot())
        if ok:
            buffer.tofile(str(rendered_path))
        results.append(
            {
                "image": item["image"],
                "expected_label": expected,
                "top_label": top_label,
                "top1_match": top_label == expected if expected else None,
                "prediction_counts": dict(Counter(prediction["label"] for prediction in predictions)),
                "predictions": predictions,
                "visualization": str(rendered_path),
                "note": item.get("note", ""),
            }
        )

    labelled = [item for item in results if item["top1_match"] is not None]
    report = {
        "weights": str(weights),
        "manifest": str(manifest_path),
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou},
        "top1_accuracy_labelled": sum(item["top1_match"] for item in labelled) / len(labelled) if labelled else None,
        "items": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Regression report: {report_path}")
    for item in results:
        print(f"{item['image']}: expected={item['expected_label']}, top={item['top_label']}, counts={item['prediction_counts']}")


if __name__ == "__main__":
    main()
