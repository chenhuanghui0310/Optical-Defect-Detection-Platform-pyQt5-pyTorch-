import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained YOLO defect model")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained weight, e.g. runs/train/yolo11s_defect_main/weights/best.pt",
    )
    parser.add_argument("--data", type=str, default="defect_competition.yaml", help="Dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or cpu")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.val(data=args.data, imgsz=args.imgsz, device=args.device, split=args.split, verbose=True)


if __name__ == "__main__":
    main()
