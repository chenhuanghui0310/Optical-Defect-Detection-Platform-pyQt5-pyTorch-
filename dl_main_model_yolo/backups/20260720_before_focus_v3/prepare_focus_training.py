"""Build a clean, deployment-like focus dataset without contaminating validation.

The generated train images are added only to YOLO's training paths.  The
separate evaluation images are never referenced by the training YAML.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from auto_dataset_generator import DatasetGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare focused train/eval datasets for local defects.")
    parser.add_argument("--output-root", default=str(ROOT / "dataset" / "focus_v2"))
    parser.add_argument("--train-count", type=int, default=600)
    parser.add_argument("--eval-count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--replace", action="store_true", help="Replace an existing output-root.")
    return parser.parse_args()


def generate(output_dir: Path, count: int, seed: int, count_choices: tuple[int, ...]) -> None:
    random.seed(seed)
    np.random.seed(seed)
    generator = DatasetGenerator(
        output_dir=output_dir,
        num_samples=count,
        weak_shape_focus=True,
        focused_defects=True,
        focus_count_choices=count_choices,
    )
    stats = generator.generate_dataset()
    if stats["failed"]:
        raise RuntimeError(f"Generation failed for {stats['failed']} samples in {output_dir}")


def make_yolo_split(source_dir: Path, output_dir: Path, seed: int) -> None:
    command = [
        sys.executable,
        str(ROOT / "dl_main_model_yolo" / "split_final_dataset.py"),
        "--source-dir", str(source_dir),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--no-balance-train",
        "--clean",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def write_training_yaml(output_root: Path) -> Path:
    base = ROOT / "dataset" / "final_split_yolo"
    focused = output_root / "train_yolo"
    yaml_path = output_root / "focus_training.yaml"

    def posix(path: Path) -> str:
        return path.resolve().as_posix()

    yaml_path.write_text(
        "path: " + posix(ROOT) + "\n"
        "train:\n"
        f"  - {posix(base / 'images' / 'train')}\n"
        f"  - {posix(focused / 'images' / 'train')}\n"
        f"val: {posix(base / 'images' / 'val')}\n"
        f"test: {posix(base / 'images' / 'test')}\n\n"
        "names:\n"
        "  0: scratch\n"
        "  1: bump\n"
        "  2: dent\n"
        "  3: dust\n"
        "  4: hair\n",
        encoding="utf-8",
    )
    return yaml_path


def verify_labels(dataset_dir: Path) -> dict[str, int]:
    counts = {"bump": 0, "dent": 0, "hair": 0}
    ids = {1: "bump", 2: "dent", 4: "hair"}
    for label_path in (dataset_dir / "labels").glob("*/*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(float(parts[0])) in ids:
                counts[ids[int(float(parts[0]))]] += 1
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"Focused data has no usable labels for: {', '.join(missing)}")
    (dataset_dir / "label_summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return counts


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if not args.replace:
            raise FileExistsError(f"Output already exists: {output_root}. Re-run with --replace to rebuild it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    train_source = output_root / "train_source"
    eval_source = output_root / "eval_source"
    generate(train_source, args.train_count, args.seed, (1, 2, 3, 4, 5, 6, 8))
    generate(eval_source, args.eval_count, args.seed + 1, (1, 2, 3, 5, 8))

    make_yolo_split(train_source, output_root / "train_yolo", args.seed)
    make_yolo_split(eval_source, output_root / "eval_yolo", args.seed + 1)
    yaml_path = write_training_yaml(output_root)
    train_labels = verify_labels(output_root / "train_yolo")
    eval_labels = verify_labels(output_root / "eval_yolo")

    print("Focus dataset ready")
    print(f"Training YAML: {yaml_path}")
    print(f"Never-train evaluation data: {output_root / 'eval_yolo'}")
    print(f"Training labels: {train_labels}")
    print(f"Evaluation labels: {eval_labels}")


if __name__ == "__main__":
    main()
