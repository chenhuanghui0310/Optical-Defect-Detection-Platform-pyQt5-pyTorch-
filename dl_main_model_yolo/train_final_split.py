import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO using auto-split Final_Dataset")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Pretrained YOLO checkpoint")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "final_split_yolo" / "dataset.yaml"),
        help="Dataset yaml produced by split_final_dataset.py",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "train"),
    )
    parser.add_argument("--name", type=str, default="yolo11s_final_split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        fliplr=0.0,
        flipud=0.0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
