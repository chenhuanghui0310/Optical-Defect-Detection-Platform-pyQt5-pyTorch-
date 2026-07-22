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
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=str, default="disk", choices=["False", "ram", "disk"], help="YOLO cache mode")
    parser.add_argument("--amp", type=lambda x: str(x).lower() == "true", default=True, help="Enable AMP mixed precision")
    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "train"),
    )
    parser.add_argument("--name", type=str, default="yolo11s_final_split")
    parser.add_argument(
        "--small-target-preset",
        action="store_true",
        help="Enable a bump/dent oriented training preset with larger imgsz and reduced heavy augmentations",
    )
    parser.add_argument(
        "--local-defect-preset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Protect thin/local fringe defects from destructive image augmentations.",
    )
    parser.add_argument(
        "--fine-tune-preset",
        action="store_true",
        help="Use a conservative learning rate when continuing from a trained defect model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cache_mode = False if args.cache == "False" else args.cache
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache_mode,
        amp=args.amp,
        project=args.project,
        name=args.name,
        pretrained=True,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.05,
        close_mosaic=20,
        verbose=True,
    )
    if args.local_defect_preset:
        train_kwargs.update(
            degrees=0.0,
            translate=0.02,
            scale=0.15,
            perspective=0.0,
            shear=0.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment=None,
            erasing=0.0,
        )
    if args.small_target_preset:
        train_kwargs.update(
            imgsz=max(args.imgsz, 1280),
            mosaic=0.05,
            close_mosaic=20,
            degrees=0.0,
            translate=0.02,
            scale=0.15,
            perspective=0.0,
            shear=0.0,
            mixup=0.0,
            copy_paste=0.0,
        )
    if args.fine_tune_preset:
        train_kwargs.update(
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=1.0,
            mosaic=0.0,
            close_mosaic=0,
        )
    model.train(
        **train_kwargs,
    )


if __name__ == "__main__":
    main()
