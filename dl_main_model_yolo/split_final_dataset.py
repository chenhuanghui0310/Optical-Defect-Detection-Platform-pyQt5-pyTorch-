import argparse
import json
import random
import shutil
from pathlib import Path
import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Final_Dataset to YOLO train/val/test and generate weak labels"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "images" / "Final_Dataset"),
        help="Directory containing *_参数.json and *_有瑕疵干涉图.png",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "final_split_yolo"),
        help="Output YOLO dataset root",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default 0.1 → 8:1:1 split)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default 0.1 → 8:1:1 split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output-dir before generating",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for p in [
        root / "images" / "train",
        root / "images" / "val",
        root / "images" / "test",
        root / "labels" / "train",
        root / "labels" / "val",
        root / "labels" / "test",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def defect_to_cls(defect_type: str) -> int:
    mapping = {
        "表面划痕": 0,
        "小突起": 1,
        "小凸起": 1,   # 统一"凸"字命名
        "小凹陷": 2,
        "灰尘污染": 3,
        "头发丝遮挡": 4,
    }
    if defect_type not in mapping:
        raise ValueError(f"Unknown defect_type: {defect_type}")
    return mapping[defect_type]


def collect_samples(source_dir: Path) -> list[tuple[Path, Path, int]]:
    samples = []
    for json_path in sorted(source_dir.glob("*_参数.json")):
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        defect_type = meta.get("defect_type")
        if not defect_type:
            continue

        sample_prefix = json_path.stem.replace("_参数", "")
        image_path = source_dir / f"{sample_prefix}_有瑕疵干涉图.png"
        ideal_path = source_dir / f"{sample_prefix}_理想干涉图.png"
        if not image_path.exists() or not ideal_path.exists():
            continue

        try:
            cls_id = defect_to_cls(defect_type)
        except ValueError:
            continue

        samples.append((image_path, ideal_path, cls_id))
    return samples


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> tuple[set[int], set[int], set[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and (val_ratio + test_ratio) < 1):
        raise ValueError("val-ratio and test-ratio must be >=0 and val+test < 1")

    val_n = int(round(n * val_ratio))
    test_n = int(round(n * test_ratio))

    if n >= 3:
        val_n = max(1, val_n)
        test_n = max(1, test_n)

    if val_n + test_n >= n:
        overflow = val_n + test_n - (n - 1)
        if test_n >= overflow:
            test_n -= overflow
        else:
            val_n -= (overflow - test_n)
            test_n = 0

    val_n = max(0, val_n)
    test_n = max(0, test_n)

    val_ids = set(indices[:val_n])
    test_ids = set(indices[val_n: val_n + test_n])
    train_ids = set(indices[val_n + test_n:])
    return train_ids, val_ids, test_ids


def write_weak_label(label_path: Path, cls_id: int) -> None:
    # Weak label: whole image is treated as the defect box.
    label_path.write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")

def generate_yolo_labels(defect_img_path: Path, ideal_img_path: Path, cls_id: int, label_path: Path) -> None:
    try:
        # OpenCV不支持中文路径，使用 numpy + imdecode 读取
        defect_data = np.fromfile(str(defect_img_path), dtype=np.uint8)
        ideal_data = np.fromfile(str(ideal_img_path), dtype=np.uint8)
        
        defect_img = cv2.imdecode(defect_data, cv2.IMREAD_COLOR)
        ideal_img = cv2.imdecode(ideal_data, cv2.IMREAD_COLOR)
        
        if defect_img is None or ideal_img is None:
            write_weak_label(label_path, cls_id)
            return

        # 智能提取绘图区（Axes）：过滤标题变长导致的图像挤压错位
        def find_plot_bbox(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = (gray < 80).astype(np.uint8)
            h, w = gray.shape
            row_sums = np.sum(mask, axis=1)
            col_sums = np.sum(mask, axis=0)
            
            # 黑框脊线长度至少占 40%
            row_peaks = np.where(row_sums > w * 0.4)[0]
            col_peaks = np.where(col_sums > h * 0.4)[0]
            
            if len(row_peaks) >= 2 and len(col_peaks) >= 2:
                return col_peaks[0], row_peaks[0], col_peaks[-1], row_peaks[-1]
            return 0, 0, w, h
            
        x1_d, y1_d, x2_d, y2_d = find_plot_bbox(defect_img)
        x1_i, y1_i, x2_i, y2_i = find_plot_bbox(ideal_img)
        
        defect_plot = defect_img[y1_d:y2_d, x1_d:x2_d]
        ideal_plot = ideal_img[y1_i:y2_i, x1_i:x2_i]
        
        if defect_plot.shape[0] == 0 or defect_plot.shape[1] == 0:
            write_weak_label(label_path, cls_id)
            return

        # 强制对齐绘图区
        ideal_plot_resized = cv2.resize(ideal_plot, (defect_plot.shape[1], defect_plot.shape[0]))

        # 只在纯净的干涉图区域做差分
        diff = cv2.absdiff(defect_plot, ideal_plot_resized)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 降低二值化阈值，为了捕捉到小突起和小凹陷引起的微小条纹畸变(相位移动)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 形态学操作：先连接相近的断裂差异区域
        kernel_close = np.ones((20, 20), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # 较大扩张，为了将条纹畸变的周边正常条纹也包含进瑕疵框，提供更多上下文供YOLO区分凹凸方向
        kernel_dilate = np.ones((30, 30), np.uint8)
        thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_d, w_d = defect_img.shape[:2]
        labels = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 过滤极小的干扰噪点
            if w > 8 and h > 8:
                # 映射回原图坐标
                x_full = x + x1_d
                y_full = y + y1_d
                
                # 给瑕疵框明显的留白（Padding），对于判断光干涉边缘的曲向(小凸起和小凹陷)极其重要
                padding = 24
                bx1 = max(0, x_full - padding)
                by1 = max(0, y_full - padding)
                bx2 = min(w_d, x_full + w + padding)
                by2 = min(h_d, y_full + h + padding)
                
                w_box = bx2 - bx1
                h_box = by2 - by1
                
                x_center = (bx1 + w_box / 2.0) / w_d
                y_center = (by1 + h_box / 2.0) / h_d
                w_norm = w_box / float(w_d)
                h_norm = h_box / float(h_d)
                
                labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if not labels:
            write_weak_label(label_path, cls_id)
        else:
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Error processing {defect_img_path.name}: {e}")
        write_weak_label(label_path, cls_id)


def copy_and_label(
    samples: list[tuple[Path, Path, int]],
    output_dir: Path,
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
) -> None:
    for idx, (img_path, ideal_path, cls_id) in enumerate(samples):
        if idx in train_ids:
            split = "train"
        elif idx in val_ids:
            split = "val"
        else:
            split = "test"

        out_img = output_dir / "images" / split / img_path.name
        out_lbl = output_dir / "labels" / split / f"{img_path.stem}.txt"

        shutil.copy2(img_path, out_img)
        generate_yolo_labels(img_path, ideal_path, cls_id, out_lbl)


def write_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "dataset.yaml"
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: scratch\n"
        "  1: bump\n"
        "  2: dent\n"
        "  3: dust\n"
        "  4: hair\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"source-dir not found: {source_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    ensure_dirs(output_dir)

    samples = collect_samples(source_dir)
    if len(samples) < 2:
        raise RuntimeError("Not enough samples to split. Check source-dir contents.")

    train_ids, val_ids, test_ids = split_indices(len(samples), args.val_ratio, args.test_ratio, args.seed)
    copy_and_label(samples, output_dir, train_ids, val_ids, test_ids)
    write_yaml(output_dir)

    print("Split finished")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Total: {len(samples)}")
    print(f"Train: {len(train_ids)}")
    print(f"Val: {len(val_ids)}")
    print(f"Test: {len(test_ids)}")
    print(f"YAML: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
