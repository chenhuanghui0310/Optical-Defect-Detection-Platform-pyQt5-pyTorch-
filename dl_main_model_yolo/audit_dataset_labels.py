"""Create deterministic, class-stratified label audit contact sheets for YOLO data."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "scratch", 1: "bump", 2: "dent", 3: "dust", 4: "hair"}
COLORS = {0: (0, 165, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255), 4: (255, 0, 255)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Audit local YOLO labels before training.")
    parser.add_argument("--dataset", default=str(root / "dataset" / "final_split_yolo"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--output-dir", default=str(root / "dl_main_model_yolo" / "runs" / "label_audit"))
    return parser.parse_args()


def load_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        try:
            boxes.append((int(values[0]), *map(float, values[1:5])))
        except ValueError:
            continue
    return boxes


def render(image_path: Path, boxes: list[tuple[int, float, float, float, float]], target_cls: int, tile_width: int) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {image_path}")
    height, width = image.shape[:2]
    for cls_id, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[cls_id], 2)
    cv2.putText(image, f"{CLASS_NAMES[target_cls]} | {image_path.stem}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    scale = tile_width / width
    return cv2.resize(image, (tile_width, max(1, int(height * scale))))


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    output_dir = Path(args.output_dir).resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records: dict[int, list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]]] = {key: [] for key in CLASS_NAMES}

    for label_path in sorted(label_dir.glob("*.txt")):
        boxes = load_boxes(label_path)
        image_matches = [path for path in image_dir.glob(f"{label_path.stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES]
        if not image_matches:
            continue
        for cls_id in {box[0] for box in boxes}:
            if cls_id in records:
                records[cls_id].append((image_matches[0], label_path, boxes))

    summary = {"dataset": str(dataset), "split": args.split, "samples": {}}
    for cls_id, candidates in records.items():
        rng.shuffle(candidates)
        chosen = candidates[: args.samples_per_class]
        tiles = [render(image, boxes, cls_id, args.tile_width) for image, _label, boxes in chosen]
        if not tiles:
            continue
        tile_height = max(tile.shape[0] for tile in tiles)
        columns = 4
        rows = math.ceil(len(tiles) / columns)
        sheet = np.zeros((rows * tile_height, columns * args.tile_width, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row, col = divmod(index, columns)
            sheet[row * tile_height:row * tile_height + tile.shape[0], col * args.tile_width:col * args.tile_width + tile.shape[1]] = tile
        output_path = output_dir / f"{CLASS_NAMES[cls_id]}_audit.png"
        ok, encoded = cv2.imencode(".png", sheet)
        if ok:
            encoded.tofile(str(output_path))
        summary["samples"][CLASS_NAMES[cls_id]] = {"available": len(candidates), "shown": len(chosen), "sheet": str(output_path)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
