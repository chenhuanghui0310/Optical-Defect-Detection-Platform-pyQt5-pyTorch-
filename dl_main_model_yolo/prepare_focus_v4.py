"""Build the V4 fine-tuning dataset from the proven Focus-V2 baseline.

V4 keeps every original final_split_yolo training image, including mixed
defect scenes.  It only *adds* audited wedge images for weak local defects,
complete hair strands, and hard-negative clean fringe patterns.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auto_dataset_generator import DatasetGenerator, WedgeParameters  # noqa: E402


FIELD_W = 0.030
FIELD_H = 0.015
CLASS_IDS = {"scratch": 0, "bump": 1, "dent": 2, "dust": 3, "hair": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited Focus-V4 additions on top of the full original dataset.")
    parser.add_argument("--output-root", default=str(ROOT / "dataset" / "focus_v4"))
    parser.add_argument("--train-local", type=int, default=300, help="Weak bump images and weak dent images each.")
    parser.add_argument("--train-hair", type=int, default=320, help="Images with one or more fully labeled hair strands.")
    parser.add_argument("--train-clean", type=int, default=260, help="Hard-negative clean fringe images.")
    parser.add_argument("--eval-local", type=int, default=80)
    parser.add_argument("--eval-hair", type=int, default=80)
    parser.add_argument("--eval-clean", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def encode_png(path: Path, image_bgr: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buffer.tofile(str(path))


def to_bgr(pattern: np.ndarray, wavelength_nm: int) -> np.ndarray:
    # Platform image export uses matplotlib origin='lower'.
    cmap = DatasetGenerator._get_color_mapping(None, wavelength_nm)
    rgb = (cmap(np.flipud(np.clip(pattern, 0.0, 1.0)))[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def configure_optics(params: WedgeParameters, rng: random.Random) -> None:
    """Include difficult but defect-free fringe variation in every class."""
    params.lambda_nm = rng.choice((589, 610, 622, 650, 680))
    params.theta_rad = rng.uniform(4e-5, 14e-5)
    params.d0_um = rng.uniform(2.0, 22.0)
    params.n1 = rng.uniform(1.2, 2.0)
    params.n2 = rng.uniform(1.2, 2.0)
    params.n3 = rng.uniform(1.2, 2.0)


def position_template(count: int) -> list[tuple[float, float]]:
    templates = {
        1: [(0.015, 0.0075)],
        2: [(0.009, 0.0075), (0.021, 0.0075)],
        3: [(0.010, 0.0038), (0.010, 0.0112), (0.022, 0.0075)],
        5: [(0.007, 0.0035), (0.015, 0.0035), (0.023, 0.0035), (0.011, 0.011), (0.019, 0.011)],
    }
    return templates[count]


def jittered_positions(count: int, radius_mm: float, rng: random.Random) -> list[tuple[float, float]]:
    """Avoid teaching the detector a few fixed locations for weak defects."""
    radius = radius_mm * 1e-3
    if count == 1:
        return [(rng.uniform(radius, FIELD_W - radius), rng.uniform(radius, FIELD_H - radius))]
    positions: list[tuple[float, float]] = []
    for x, y in position_template(count):
        x = min(FIELD_W - radius, max(radius, x + rng.uniform(-0.0011, 0.0011)))
        y = min(FIELD_H - radius, max(radius, y + rng.uniform(-0.0008, 0.0008)))
        positions.append((x, y))
    return positions


def yolo_box_for_position(cx: float, cy: float, radius_mm: float, class_id: int) -> str:
    radius = radius_mm * 1e-3
    x1, x2 = max(0.0, cx - radius), min(FIELD_W, cx + radius)
    y1, y2 = max(0.0, cy - radius), min(FIELD_H, cy + radius)
    return (
        f"{class_id} {(x1 + x2) / 2 / FIELD_W:.6f} "
        f"{1.0 - ((y1 + y2) / 2 / FIELD_H):.6f} "
        f"{(x2 - x1) / FIELD_W:.6f} {(y2 - y1) / FIELD_H:.6f}"
    )


def bbox_from_component(component: np.ndarray, class_id: int, pad: int = 5) -> str:
    ys, xs = np.where(component)
    if not len(xs):
        raise RuntimeError("Empty component")
    height, width = component.shape
    x1, x2 = max(0, xs.min() - pad), min(width, xs.max() + 1 + pad)
    y1, y2 = max(0, ys.min() - pad), min(height, ys.max() + 1 + pad)
    return f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def build_local(params: WedgeParameters, label: str, count: int, rng: random.Random) -> tuple[np.ndarray, list[str], dict]:
    params.defect_type = "小突起" if label == "bump" else "小凹陷"
    params.bump_count = count
    # Most samples target the requested 0.02 um and 2.9/3.0 mm weak shapes;
    # nearby values prevent memorising two exact parameter pairs.
    params.bump_height = 0.02 if rng.random() < 0.60 else rng.uniform(0.014, 0.030)
    params.bump_radius_mm = rng.choice((2.9, 3.0)) if rng.random() < 0.60 else rng.uniform(2.7, 3.2)
    params.bump_position_mode = "固定位置"
    params.fixed_bump_positions = jittered_positions(count, params.bump_radius_mm, rng)
    _curve, pattern = params.calculate_defect_interference()
    lines = [
        yolo_box_for_position(x, y, params.bump_radius_mm, CLASS_IDS[label])
        for x, y in params.generated_bump_positions
    ]
    if len(lines) != count:
        raise RuntimeError(f"Expected {count} local labels, got {len(lines)}")
    return to_bgr(pattern, params.lambda_nm), lines, {
        "class": label,
        "count": count,
        "height_um": params.bump_height,
        "radius_mm": params.bump_radius_mm,
    }


def build_hair(params: WedgeParameters, rng: random.Random) -> tuple[np.ndarray, list[str], dict]:
    params.defect_type = "头发丝遮挡"
    params.hair_count = rng.choices((1, 2, 3), weights=(0.60, 0.30, 0.10), k=1)[0]
    _curve, pattern = params.calculate_defect_interference()
    raw_mask = getattr(params, "hair_mask", None)
    if raw_mask is None:
        raise RuntimeError("Hair generator did not return a mask")

    # The mask contains full physical strands.  Label connected components,
    # not small fragments, so each box covers a complete visible strand.
    mask = np.flipud(raw_mask.astype(np.uint8))
    total, components, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lines: list[str] = []
    for component_id in range(1, total):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < 12:
            continue
        lines.append(bbox_from_component(components == component_id, CLASS_IDS["hair"], pad=5))
    if not lines:
        raise RuntimeError("No visible hair components were generated")
    return to_bgr(pattern, params.lambda_nm), lines, {
        "class": "hair",
        "count": len(lines),
        "requested_hair_count": params.hair_count,
    }


def build_clean(params: WedgeParameters) -> tuple[np.ndarray, list[str], dict]:
    # These are deliberate hard negatives: they have the same wavelength and
    # fringe curvature variation as local-defect images but have no defects.
    params.defect_type = "无瑕疵"
    _curve, pattern = params.calculate_ideal_interference()
    return to_bgr(pattern, params.lambda_nm), [], {"class": "clean", "count": 0}


def save_sample(image_dir: Path, label_dir: Path, sample_id: int, name: str, image: np.ndarray, labels: list[str], meta: dict) -> None:
    stem = f"{sample_id:05d}_{name}"
    encode_png(image_dir / f"{stem}.png", image)
    (label_dir / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    (image_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_split(root: Path, split: str, counts: dict[str, int], seed: int) -> dict[str, int]:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    sample_id = 1
    written = {name: 0 for name in counts}
    for label in ("bump", "dent", "hair", "clean"):
        for index in range(counts.get(label, 0)):
            params = WedgeParameters()
            configure_optics(params, rng)
            if label in ("bump", "dent"):
                image, lines, meta = build_local(params, label, (1, 2, 3, 5)[index % 4], rng)
            elif label == "hair":
                image, lines, meta = build_hair(params, rng)
            else:
                image, lines, meta = build_clean(params)
            save_sample(image_dir, label_dir, sample_id, label, image, lines, meta)
            sample_id += 1
            written[label] += 1
    return written


def copy_full_base(output_root: Path) -> Path:
    """Copy all original images and labels, preserving mixed-defect scenes."""
    source = ROOT / "dataset" / "final_split_yolo"
    target = output_root / "base_full"
    image_dir, label_dir = target / "images" / "train", target / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for source_label in (source / "labels" / "train").glob("*.txt"):
        source_image = next(
            (p for p in (source / "images" / "train").glob(f"{source_label.stem}.*") if p.suffix.lower() in image_extensions),
            None,
        )
        if source_image is None:
            continue
        shutil.copy2(source_label, label_dir / source_label.name)
        shutil.copy2(source_image, image_dir / source_image.name)
        copied += 1
    if not copied:
        raise RuntimeError("No original training samples were copied")
    return target


def write_yaml(output_root: Path) -> Path:
    base = copy_full_base(output_root)
    focused = output_root / "train"
    path = output_root / "focus_v4_training.yaml"
    posix = lambda item: item.resolve().as_posix()
    path.write_text(
        f"path: {posix(ROOT)}\n"
        "train:\n"
        f"  - {posix(base / 'images' / 'train')}\n"
        f"  - {posix(focused / 'images' / 'train')}\n"
        f"val: {posix(ROOT / 'dataset' / 'final_split_yolo' / 'images' / 'val')}\n"
        f"test: {posix(ROOT / 'dataset' / 'final_split_yolo' / 'images' / 'test')}\n\n"
        "names:\n  0: scratch\n  1: bump\n  2: dent\n  3: dust\n  4: hair\n",
        encoding="utf-8",
    )
    return path


def validate_generated_split(root: Path, split: str) -> dict[str, int]:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    verified, class_boxes = 0, {name: 0 for name in CLASS_IDS}
    for meta_path in image_dir.glob("*.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lines = [line.split() for line in (label_dir / f"{meta_path.stem}.txt").read_text(encoding="utf-8").splitlines() if line.split()]
        if len(lines) != int(meta["count"]):
            raise RuntimeError(f"{meta_path.name}: expected {meta['count']} labels, got {len(lines)}")
        for parts in lines:
            if len(parts) != 5:
                raise RuntimeError(f"{meta_path.name}: malformed YOLO label")
            class_boxes[next(name for name, value in CLASS_IDS.items() if value == int(parts[0]))] += 1
            values = [float(value) for value in parts[1:]]
            if not all(0.0 <= value <= 1.0 for value in values[:2]) or not all(0.0 < value <= 1.0 for value in values[2:]):
                raise RuntimeError(f"{meta_path.name}: out-of-range label")
        verified += 1
    return {"images": verified, **class_boxes}


def main() -> None:
    args = parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"{output} exists. Use --replace to rebuild it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    train_counts = {"bump": args.train_local, "dent": args.train_local, "hair": args.train_hair, "clean": args.train_clean}
    eval_counts = {"bump": args.eval_local, "dent": args.eval_local, "hair": args.eval_hair, "clean": args.eval_clean}
    train_summary = generate_split(output / "train", "train", train_counts, args.seed)
    eval_summary = generate_split(output / "eval", "test", eval_counts, args.seed + 1)
    yaml_path = write_yaml(output)
    quality = {"train": validate_generated_split(output / "train", "train"), "eval": validate_generated_split(output / "eval", "test")}
    (output / "summary.json").write_text(json.dumps({"generated": {"train": train_summary, "eval": eval_summary}, "quality": quality}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"focus_v4 ready: {output}")
    print(f"Training YAML: {yaml_path}")
    print(f"Held-out evaluation dataset: {output / 'eval'}")
    print("Original mixed-defect training images were retained in base_full.")


if __name__ == "__main__":
    main()
