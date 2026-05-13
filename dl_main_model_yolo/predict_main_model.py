import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained YOLO defect model")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "train" / "yolo11s_final_split2" / "weights" / "best.pt"),
        help="Path to trained weight, e.g. runs/train/yolo11s_final_split2/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "images" / "val"),
        help="Image/video/folder source",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or cpu")
    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "predict"),
        help="Output folder",
    )
    parser.add_argument("--name", type=str, default="main_model_pred", help="Run name")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt results")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence in txt results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=True,
    )


if __name__ == "__main__":
    main()
