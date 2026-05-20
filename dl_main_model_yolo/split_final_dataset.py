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
    skipped_unknown = 0
    for json_path in sorted(source_dir.glob("*_参数.json")):
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # 过滤旧版数据（sim_type=unknown 的样本格式不一致，会污染训练集）
        sim_type = meta.get("sim_type", "unknown")
        if sim_type == "unknown":
            skipped_unknown += 1
            continue

        defect_type = meta.get("defect_type")
        if not defect_type:
            continue

        sample_prefix = json_path.stem.replace("_参数", "")
        image_path = source_dir / f"{sample_prefix}_有瑕疵干涉图.png"

        # 真实数据：图片文件名不带"_有瑕疵干涉图"后缀，尝试多种扩展名
        if not image_path.exists() and sim_type == "real":
            for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif"]:
                candidate = source_dir / f"{sample_prefix}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

        ideal_path = source_dir / f"{sample_prefix}_理想干涉图.png"
        if not image_path.exists():
            continue

        try:
            cls_id = defect_to_cls(defect_type)
        except ValueError:
            continue

        samples.append((image_path, ideal_path, cls_id))

    if skipped_unknown > 0:
        print(f"  [INFO] 跳过 {skipped_unknown} 个旧版样本（sim_type=unknown）")
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


def generate_analytic_labels(
    defect_img_path: Path,
    meta: dict,
    cls_id: int,
    label_path: Path,
) -> None:
    """
    解析标注：直接从 JSON 参数计算 YOLO 格式的边界框，无需差分图。
    适用于 小凸起(cls=1)、小凹陷(cls=2)、头发丝遮挡(cls=4)。

    坐标转换公式（以牛顿环为例）：
        物理坐标范围 [-L/2, L/2]（米）→ 像素坐标 [0, img_size]
        pixel = (phys / L + 0.5) * img_size
    """
    # 读取图像尺寸
    img_data = np.fromfile(str(defect_img_path), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        write_weak_label(label_path, cls_id)
        return

    img_h, img_w = img.shape[:2]
    sim_type = meta.get("sim_type", "wedge")

    # 物理视场尺寸（米）
    if sim_type == "newton":
        field_x = 0.006   # ±3mm → 6mm 总宽
        field_y = 0.006
    else:
        field_x = 0.030   # 0~30mm
        field_y = 0.015   # 0~15mm

    def phys_to_norm_newton(cx_m, cy_m, r_m):
        """牛顿环：原点在视场中心"""
        px = (cx_m / field_x + 0.5)
        py = 1.0 - (cy_m / field_y + 0.5)   # imshow origin='lower' → y 翻转
        pw = r_m * 2 / field_x
        ph = r_m * 2 / field_y
        return px, py, pw, ph

    def phys_to_norm_wedge(cx_m, cy_m, r_m):
        """劈尖：原点在视场左下角"""
        px = cx_m / field_x
        py = 1.0 - cy_m / field_y
        pw = r_m * 2 / field_x
        ph = r_m * 2 / field_y
        return px, py, pw, ph

    labels = []

    if cls_id in (1, 2):
        # 小凸起 / 小凹陷：从参数直接算每个 bump 的位置和半径
        bump_count = int(meta.get("bump_count", 1))
        bump_radius_mm = float(meta.get("bump_radius_mm", 2.4))
        bump_position_mode = meta.get("bump_position_mode", "网格排列")

        # 限制半径不超过视场 48%（与 parameters.py 一致）
        max_r_m = min(field_x, field_y) * 0.48
        r_m = float(np.clip(bump_radius_mm * 1e-3, 0.05e-3, max_r_m))

        # 重建位置列表（与 parameters.py 的 _generate_*_positions 逻辑一致）
        if bump_position_mode == "固定位置":
            raw_positions = meta.get("fixed_bump_positions", [])
            positions = [(float(p[0]), float(p[1])) for p in raw_positions[:bump_count]]
        elif bump_position_mode == "随机分布":
            # 随机位置无法从 JSON 精确重建，退回差分图标注
            generate_yolo_labels_diff(defect_img_path, None, cls_id, label_path, meta)
            return
        else:
            # 网格排列：重建网格
            grid_size = int(np.ceil(np.sqrt(bump_count)))
            positions = []
            if sim_type == "newton":
                sx, sy = field_x, field_y
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) >= bump_count:
                            break
                        cx = -sx / 2 + (j + 1) * sx / (grid_size + 1)
                        base_sp = sy / (grid_size + 1)
                        cy = -sy / 2 + (i + 1) * base_sp + i * base_sp * 0.3
                        cy = float(np.clip(cy, -0.45 * sy, 0.45 * sy))
                        positions.append((cx, cy))
            else:
                sx, sy = field_x, field_y
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) >= bump_count:
                            break
                        cx = (j + 1) * sx / (grid_size + 1)
                        cy = (i + 1) * sy / (grid_size + 1)
                        positions.append((cx, cy))

        for cx_m, cy_m in positions[:bump_count]:
            if sim_type == "newton":
                px, py, pw, ph = phys_to_norm_newton(cx_m, cy_m, r_m)
            else:
                px, py, pw, ph = phys_to_norm_wedge(cx_m, cy_m, r_m)

            # 适度留白 5%（相对于 bump 直径）
            pad = max(pw, ph) * 0.05
            pw = min(1.0, pw + pad * 2)
            ph = min(1.0, ph + pad * 2)

            # 裁剪到 [0,1]
            px = float(np.clip(px, pw / 2, 1 - pw / 2))
            py = float(np.clip(py, ph / 2, 1 - ph / 2))
            pw = float(np.clip(pw, 0.01, 1.0))
            ph = float(np.clip(ph, 0.01, 1.0))

            labels.append(f"{cls_id} {px:.6f} {py:.6f} {pw:.6f} {ph:.6f}")

    elif cls_id == 4:
        # 头发丝遮挡：路径随机，无法解析，用全图弱标签
        write_weak_label(label_path, cls_id)
        return

    if not labels:
        write_weak_label(label_path, cls_id)
    else:
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

def generate_yolo_labels_diff(defect_img_path: Path, ideal_img_path: Path | None, cls_id: int, label_path: Path, meta: dict | None = None) -> None:
    """差分图标注：用于表面划痕（cls=0）。理想图路径可选，为 None 时用弱标签。"""
    if ideal_img_path is None or not ideal_img_path.exists():
        write_weak_label(label_path, cls_id)
        return
    try:
        # OpenCV不支持中文路径，使用 numpy + imdecode 读取
        defect_data = np.fromfile(str(defect_img_path), dtype=np.uint8)
        ideal_data = np.fromfile(str(ideal_img_path), dtype=np.uint8)
        
        defect_img = cv2.imdecode(defect_data, cv2.IMREAD_COLOR)
        ideal_img = cv2.imdecode(ideal_data, cv2.IMREAD_COLOR)
        
        if defect_img is None or ideal_img is None:
            write_weak_label(label_path, cls_id)
            return

        # 智能提取绘图区（Axes）：兼容有/无坐标轴标签的图像
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
            # 无明显边框时（无坐标轴的纯净图），直接使用全图
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
        
        # 灰尘亮度差更弱，单独降低阈值
        thresh_val = 3 if cls_id == 3 else 8
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # 根据图像尺寸自适应形态学核大小
        # 牛顿环图像约 750×750，劈尖图像约 750×375，核大小按比例缩放
        img_short = min(defect_img.shape[:2])
        # 闭运算：连接断裂的差异区域，核大小约为图像短边的 1.5%
        close_k = max(5, int(img_short * 0.015))
        kernel_close = np.ones((close_k, close_k), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        # 膨胀：适度扩大检测区域，核大小约为图像短边的 2%（比之前的 30×30 小很多）
        dilate_k = max(6, int(img_short * 0.02))
        kernel_dilate = np.ones((dilate_k, dilate_k), np.uint8)
        thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_d, w_d = defect_img.shape[:2]
        labels = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 过滤极小的干扰噪点（面积过滤更可靠）
            if w > 6 and h > 6 and w * h > 100:
                # 映射回原图坐标
                x_full = x + x1_d
                y_full = y + y1_d

                # 适度留白：约为图像短边的 1%，比之前的固定 24px 更合理
                padding = max(8, int(img_short * 0.01))
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

        # 读取对应的 JSON meta（用于解析标注）
        json_path = img_path.parent / img_path.name.replace("_有瑕疵干涉图.png", "_参数.json")
        # 真实数据的 JSON 文件名格式不同
        if not json_path.exists():
            json_path = img_path.parent / f"{img_path.stem}_参数.json"
        meta: dict = {}
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        sim_type = meta.get("sim_type", "wedge")

        # 真实数据：检查是否有预生成的标注文件（来自 merge_real_data.py）
        if sim_type == "real":
            pre_label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if pre_label.exists():
                shutil.copy2(pre_label, out_lbl)
            else:
                write_weak_label(out_lbl, cls_id)
        elif cls_id in (0, 3):
            # 表面划痕/灰尘污染：差分图标注
            generate_yolo_labels_diff(img_path, ideal_path, cls_id, out_lbl, meta)
        else:
            # 小凸起/小凹陷/头发丝：解析标注
            generate_analytic_labels(img_path, meta, cls_id, out_lbl)


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
