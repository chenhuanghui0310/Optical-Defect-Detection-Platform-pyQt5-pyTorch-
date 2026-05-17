# -*- coding: utf-8 -*-
"""
合并真实数据到 Final_Dataset 目录。

功能：
1. 将 Roboflow 导出的灰尘+头发丝标注数据（类别ID重映射：0→3, 1→4）
2. 将划痕真实图片（自动生成全图弱标签 cls=0）
3. 统一复制到 dataset/images/Final_Dataset/ 并生成对应的 _参数.json

用法：
    python dl_main_model_yolo/merge_real_data.py
"""

import json
import shutil
from pathlib import Path


# ─── 路径配置 ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Roboflow 导出目录（灰尘+头发丝，已标注）
ROBOFLOW_DIR = PROJECT_ROOT / "dataset" / "My First Project.yolov5pytorch" / "train"

# 划痕真实图片目录
SCRATCH_DIRS = [
    PROJECT_ROOT / "dataset" / "real_data" / "20260127-划痕",
    PROJECT_ROOT / "dataset" / "real_data" / "20260211-划痕",
]

# 输出目标目录
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "images" / "Real_Data"

# Roboflow 类别映射：Roboflow 的 0=dust→我们的3, 1=hair→我们的4
ROBOFLOW_CLS_MAP = {0: 3, 1: 4}

# 图片扩展名
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def ensure_dirs():
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)


def process_roboflow_data():
    """处理 Roboflow 导出的灰尘+头发丝数据"""
    images_dir = ROBOFLOW_DIR / "images"
    labels_dir = ROBOFLOW_DIR / "labels"

    if not images_dir.exists():
        print(f"[WARN] Roboflow images dir not found: {images_dir}")
        return 0

    count = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # 找对应的标注文件
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  [SKIP] No label for {img_path.name}")
            continue

        # 读取标注并重映射类别ID
        lines = label_path.read_text(encoding="utf-8").strip().split("\n")
        remapped_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            old_cls = int(parts[0])
            new_cls = ROBOFLOW_CLS_MAP.get(old_cls)
            if new_cls is None:
                print(f"  [WARN] Unknown class {old_cls} in {label_path.name}, skipping line")
                continue
            remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        if not remapped_lines:
            continue

        # 生成统一文件名
        out_name = f"real_dusthair_{count:04d}"
        out_img = OUTPUT_DIR / "images" / f"{out_name}{img_path.suffix}"
        out_lbl = OUTPUT_DIR / "labels" / f"{out_name}.txt"

        # 复制图片
        shutil.copy2(img_path, out_img)

        # 写入重映射后的标注
        out_lbl.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

        # 生成 _参数.json（用于 split_final_dataset.py 识别）
        # 确定主要类别（取标注中出现最多的类别）
        cls_counts = {}
        for line in remapped_lines:
            cls_id = int(line.split()[0])
            cls_counts[cls_id] = cls_counts.get(cls_id, 0) + 1
        main_cls = max(cls_counts, key=cls_counts.get)
        defect_name = {3: "灰尘污染", 4: "头发丝遮挡"}.get(main_cls, "灰尘污染")

        meta = {
            "defect_type": defect_name,
            "sim_type": "real",
            "source": "roboflow_labeled",
            "original_file": img_path.name,
        }
        meta_path = OUTPUT_DIR / "images" / f"{out_name}_参数.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        count += 1

    print(f"  [OK] Roboflow dust/hair: {count} samples processed")
    return count


def process_scratch_data():
    """处理划痕真实图片（全图弱标签）"""
    count = 0

    for scratch_dir in SCRATCH_DIRS:
        if not scratch_dir.exists():
            print(f"  [WARN] Scratch dir not found: {scratch_dir}")
            continue

        for img_path in sorted(scratch_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            out_name = f"real_scratch_{count:04d}"
            out_img = OUTPUT_DIR / "images" / f"{out_name}{img_path.suffix}"
            out_lbl = OUTPUT_DIR / "labels" / f"{out_name}.txt"

            # 复制图片
            shutil.copy2(img_path, out_img)

            # 全图弱标签：cls=0 (scratch), 中心(0.5,0.5), 全图(1.0,1.0)
            out_lbl.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

            # 生成 _参数.json
            meta = {
                "defect_type": "表面划痕",
                "sim_type": "real",
                "source": "real_photo",
                "original_file": img_path.name,
            }
            meta_path = OUTPUT_DIR / "images" / f"{out_name}_参数.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            count += 1

    print(f"  [OK] Scratch real photos: {count} samples processed")
    return count


def main():
    print("=" * 60)
    print("  合并真实数据到训练集")
    print("=" * 60)
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    ensure_dirs()

    total = 0
    total += process_roboflow_data()
    total += process_scratch_data()

    print()
    print(f"  总计: {total} 张真实图片已处理")
    print(f"  输出: {OUTPUT_DIR}")
    print()
    print("  下一步:")
    print("  1. 先运行 auto_dataset_generator.py 生成仿真数据到 Final_Dataset")
    print("  2. 将 Real_Data/images/ 和 Real_Data/labels/ 的内容复制到 Final_Dataset/")
    print("  3. 运行 split_final_dataset.py --clean 重新划分数据集")
    print("=" * 60)


if __name__ == "__main__":
    main()
