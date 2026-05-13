import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO main model for defect detection")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Pretrained YOLO checkpoint")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "defect_competition.yaml"),
        help="Dataset yaml path",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, e.g. 0, or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--patience", type=int, default=40, help="Early stop patience")
    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "train"),
        help="Output project folder",
    )
    parser.add_argument("--name", type=str, default="yolo11s_defect_main", help="Run name")
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
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
