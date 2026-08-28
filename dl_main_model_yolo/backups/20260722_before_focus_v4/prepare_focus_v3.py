"""Create a label-audited, wedge-focused dataset for local optical defects.

This builder intentionally does not reuse focus_v2.  In particular, it creates
one hair strand per image and one tight annotation per physical instance.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))
from auto_dataset_generator import DatasetGenerator, WedgeParameters  # noqa: E402


FIELD_W = 0.030
FIELD_H = 0.015
CLASS_IDS = {"scratch": 0, "bump": 1, "dent": 2, "dust": 3, "hair": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build focus_v3 training and held-out evaluation data.")
    parser.add_argument("--output-root", default=str(ROOT / "dataset" / "focus_v3"))
    parser.add_argument("--train-local", type=int, default=240, help="Bump images and dent images each.")
    parser.add_argument("--train-hair", type=int, default=240)
    parser.add_argument("--train-scratch", type=int, default=240)
    parser.add_argument("--train-clean", type=int, default=120)
    parser.add_argument("--eval-local", type=int, default=60, help="Bump images and dent images each.")
    parser.add_argument("--eval-hair", type=int, default=60)
    parser.add_argument("--eval-scratch", type=int, default=60)
    parser.add_argument("--eval-clean", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def encode_png(path: Path, image_bgr: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buffer.tofile(str(path))


def to_bgr(pattern: np.ndarray, wavelength_nm: int) -> np.ndarray:
    # Match the platform export: imshow(..., origin='lower') flips the array.
    cmap = DatasetGenerator._get_color_mapping(None, wavelength_nm)
    rgb = (cmap(np.flipud(np.clip(pattern, 0.0, 1.0)))[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def configure_optics(params: WedgeParameters, rng: random.Random) -> None:
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


def yolo_box_for_position(cx: float, cy: float, radius_mm: float) -> str:
    radius = radius_mm * 1e-3
    x1 = max(0.0, cx - radius)
    x2 = min(FIELD_W, cx + radius)
    y1 = max(0.0, cy - radius)
    y2 = min(FIELD_H, cy + radius)
    # The exported image has origin='lower', so convert physical y to image y.
    yc = 1.0 - ((y1 + y2) / 2.0 / FIELD_H)
    xc = (x1 + x2) / 2.0 / FIELD_W
    width = (x2 - x1) / FIELD_W
    height = (y2 - y1) / FIELD_H
    return f"{xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"


def bbox_from_mask(mask: np.ndarray, class_id: int, pad: int = 4) -> str:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("Empty ground-truth mask")
    height, width = mask.shape
    x1, x2 = max(0, xs.min() - pad), min(width, xs.max() + 1 + pad)
    y1, y2 = max(0, ys.min() - pad), min(height, ys.max() + 1 + pad)
    return f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def save_sample(image_dir: Path, label_dir: Path, sample_id: int, name: str, image: np.ndarray, labels: list[str], meta: dict) -> None:
    stem = f"{sample_id:05d}_{name}"
    encode_png(image_dir / f"{stem}.png", image)
    (label_dir / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    (image_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_local(params: WedgeParameters, label: str, count: int, rng: random.Random) -> tuple[np.ndarray, list[str], dict]:
    params.defect_type = "小突起" if label == "bump" else "小凹陷"
    params.bump_count = count
    params.bump_height = 0.02 if rng.random() < 0.50 else rng.uniform(0.014, 0.030)
    params.bump_radius_mm = rng.choice((2.9, 3.0)) if rng.random() < 0.50 else rng.uniform(2.7, 3.2)
    params.bump_position_mode = "固定位置"
    params.fixed_bump_positions = position_template(count)
    _curve, pattern = params.calculate_defect_interference()
    lines = [f"{CLASS_IDS[label]} {yolo_box_for_position(x, y, params.bump_radius_mm)}" for x, y in params.generated_bump_positions]
    if len(lines) != count:
        raise RuntimeError(f"Expected {count} local labels, got {len(lines)}")
    return to_bgr(pattern, params.lambda_nm), lines, {
        "class": label,
        "count": count,
        "height_um": params.bump_height,
        "radius_mm": params.bump_radius_mm,
        "positions": params.generated_bump_positions,
    }


def build_hair(params: WedgeParameters, rng: random.Random) -> tuple[np.ndarray, list[str], dict]:
    params.defect_type = "头发丝遮挡"
    params.hair_count = 1
    _curve, pattern = params.calculate_defect_interference()
    mask = getattr(params, "hair_mask", None)
    if mask is None:
        raise RuntimeError("Hair generator did not return a mask")
    # Match the origin-lower image written by to_bgr.
    line = bbox_from_mask(np.flipud(mask.astype(bool)), CLASS_IDS["hair"], pad=5)
    return to_bgr(pattern, params.lambda_nm), [line], {"class": "hair", "count": 1}


def build_scratch(params: WedgeParameters, helper: DatasetGenerator) -> tuple[np.ndarray, list[str], dict]:
    params.defect_type = "表面划痕"
    params.use_manual_scratches = True
    params.manual_scratch_type = "凹陷划痕"
    mask = None
    for _ in range(8):
        candidate = helper._generate_random_scratch_mask(params.simulation_size, "wedge")
        if np.count_nonzero(candidate > 0.01) >= 20:
            mask = candidate
            break
    if mask is None:
        # A rare random stroke may fall outside the sampled canvas.  Keep the
        # sample usable with a deterministic thin zig-zag rather than writing
        # an empty or full-image scratch label.
        size = params.simulation_size
        mask = np.zeros((size, size), dtype=np.float32)
        points = np.array(
            [(size // 8, size * 3 // 4), (size * 3 // 8, size // 3),
             (size * 5 // 8, size * 2 // 3), (size * 7 // 8, size // 4)],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(mask, [points], isClosed=False, color=1.0, thickness=max(3, size // 90), lineType=cv2.LINE_AA)
    params.manual_scratch_mask = mask
    _curve, pattern = params.calculate_defect_interference()
    # add_scratch_defect() maps the source mask onto the simulation with its
    # own vertical coordinate conversion; to_bgr() then flips the rendered
    # pattern for the origin='lower' export.  Those two transforms cancel for
    # a scratch, so the exported-image box is derived from the source mask
    # directly.  Flipping here shifts labels vertically away from the mark.
    line = bbox_from_mask(mask > 0.01, CLASS_IDS["scratch"], pad=5)
    return to_bgr(pattern, params.lambda_nm), [line], {"class": "scratch", "count": 1}


def build_clean(params: WedgeParameters) -> tuple[np.ndarray, list[str], dict]:
    params.defect_type = "无瑕疵"
    _curve, pattern = params.calculate_ideal_interference()
    return to_bgr(pattern, params.lambda_nm), [], {"class": "clean", "count": 0}


def generate_split(root: Path, split: str, counts: dict[str, int], seed: int) -> dict[str, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    helper = DatasetGenerator.__new__(DatasetGenerator)
    sample_id = 1
    written = {name: 0 for name in counts}

    for label in ("bump", "dent", "hair", "scratch", "clean"):
        for index in range(counts.get(label, 0)):
            params = WedgeParameters()
            configure_optics(params, rng)
            if label in ("bump", "dent"):
                count = (1, 2, 3, 5)[index % 4]
                image, lines, meta = build_local(params, label, count, rng)
            elif label == "hair":
                image, lines, meta = build_hair(params, rng)
            elif label == "scratch":
                image, lines, meta = build_scratch(params, helper)
            else:
                image, lines, meta = build_clean(params)
            save_sample(image_dir, label_dir, sample_id, label, image, lines, meta)
            sample_id += 1
            written[label] += 1
    return written


def copy_safe_base(output_root: Path) -> Path:
    source = ROOT / "dataset" / "final_split_yolo"
    target = output_root / "base_safe"
    image_dir = target / "images" / "train"
    label_dir = target / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    excluded = {CLASS_IDS["scratch"], CLASS_IDS["hair"]}
    kept = 0
    for source_label in (source / "labels" / "train").glob("*.txt"):
        lines = source_label.read_text(encoding="utf-8").splitlines()
        classes = {int(line.split()[0]) for line in lines if line.split()}
        if classes & excluded:
            continue
        source_image = next((candidate for candidate in (source / "images" / "train").glob(f"{source_label.stem}.*")), None)
        if source_image is None:
            continue
        shutil.copy2(source_image, image_dir / source_image.name)
        shutil.copy2(source_label, label_dir / source_label.name)
        kept += 1
    if not kept:
        raise RuntimeError("No safe base images were copied")
    return target


def write_yaml(output_root: Path) -> Path:
    path = output_root / "focus_v3_training.yaml"
    base = copy_safe_base(output_root)
    focused = output_root / "train"

    def posix(item: Path) -> str:
        return item.resolve().as_posix()

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
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    verified = 0
    class_counts: dict[str, int] = {}
    for meta_path in image_dir.glob("*.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        class_name = str(meta["class"])
        expected = int(meta["count"])
        label_path = label_dir / f"{meta_path.stem}.txt"
        lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.split()]
        if len(lines) != expected:
            raise RuntimeError(f"{meta_path.name}: expected {expected} labels, found {len(lines)}")
        for parts in lines:
            if len(parts) != 5:
                raise RuntimeError(f"{label_path.name}: malformed YOLO label")
            values = [float(value) for value in parts[1:]]
            if not all(0.0 < value <= 1.0 for value in values[2:]) or not all(0.0 <= value <= 1.0 for value in values[:2]):
                raise RuntimeError(f"{label_path.name}: out-of-range YOLO box")
        class_counts[class_name] = class_counts.get(class_name, 0) + len(lines)
        verified += 1
    return {"images": verified, **class_counts}


def main() -> None:
    args = parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"{output} exists. Use --replace to rebuild it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    train_counts = {"bump": args.train_local, "dent": args.train_local, "hair": args.train_hair, "scratch": args.train_scratch, "clean": args.train_clean}
    eval_counts = {"bump": args.eval_local, "dent": args.eval_local, "hair": args.eval_hair, "scratch": args.eval_scratch, "clean": args.eval_clean}
    train_summary = generate_split(output / "train", "train", train_counts, args.seed)
    eval_summary = generate_split(output / "eval", "test", eval_counts, args.seed + 1)
    yaml_path = write_yaml(output)
    quality = {
        "train": validate_generated_split(output / "train", "train"),
        "eval": validate_generated_split(output / "eval", "test"),
    }
    (output / "summary.json").write_text(
        json.dumps({"generated": {"train": train_summary, "eval": eval_summary}, "quality": quality}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"focus_v3 ready: {output}")
    print(f"Training YAML: {yaml_path}")
    print(f"Held-out evaluation dataset: {output / 'eval'}")


if __name__ == "__main__":
    main()
