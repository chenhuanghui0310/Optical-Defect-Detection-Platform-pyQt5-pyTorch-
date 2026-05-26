import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Final_Dataset into YOLO train/val/test and build local labels."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "images" / "Final_Dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "final_split_yolo"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--preprocess-real", action="store_true")
    parser.add_argument("--max-real-train-ratio", type=float, default=1.0)
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for p in [
        root / "images" / "train",
        root / "images" / "val",
        root / "images" / "test",
        root / "labels" / "train",
        root / "labels" / "val",
        root / "labels" / "test",
        root / "metadata",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def defect_to_cls(defect_type: str) -> int:
    mapping = {
        "表面划痕": 0,
        "小突起": 1,
        "小凸起": 1,
        "小凹陷": 2,
        "灰尘污染": 3,
        "头发丝遮挡": 4,
    }
    if defect_type not in mapping:
        raise ValueError(f"Unknown defect_type: {defect_type}")
    return mapping[defect_type]


def collect_samples(source_dir: Path) -> list[tuple[Path, Path, int, str]]:
    samples: list[tuple[Path, Path, int, str]] = []
    skipped_unknown = 0
    for json_path in sorted(source_dir.glob("*_参数.json")):
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        sim_type = meta.get("sim_type", "unknown")
        if sim_type == "unknown":
            skipped_unknown += 1
            continue

        defect_type = meta.get("defect_type")
        if not defect_type:
            continue

        sample_prefix = json_path.stem.replace("_参数", "")
        image_path = source_dir / f"{sample_prefix}_有瑕疵干涉图.png"
        if not image_path.exists() and sim_type == "real":
            for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
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

        samples.append((image_path, ideal_path, cls_id, sim_type))

    if skipped_unknown > 0:
        print(f"[INFO] skipped unknown-format samples: {skipped_unknown}")
    return samples


def split_indices(
    samples: list[tuple[Path, Path, int, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_real_train_ratio: float = 1.0,
) -> tuple[set[int], set[int], set[int]]:
    n = len(samples)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and (val_ratio + test_ratio) < 1):
        raise ValueError("val/test ratio invalid")

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
            val_n -= overflow - test_n
            test_n = 0

    val_ids = set(indices[:val_n])
    test_ids = set(indices[val_n: val_n + test_n])
    train_ids = set(indices[val_n + test_n:])

    max_real_train_ratio = float(np.clip(max_real_train_ratio, 0.0, 1.0))
    if max_real_train_ratio < 1.0 and train_ids:
        real_train = [idx for idx in train_ids if samples[idx][3] == "real"]
        sim_train = [idx for idx in train_ids if samples[idx][3] != "real"]
        max_real_count = int(len(train_ids) * max_real_train_ratio)
        if sim_train and len(real_train) > max_real_count:
            rng.shuffle(real_train)
            keep_real = set(real_train[:max_real_count])
            move_real = real_train[max_real_count:]
            train_ids = set(sim_train) | keep_real
            for idx in move_real:
                if len(val_ids) <= len(test_ids):
                    val_ids.add(idx)
                else:
                    test_ids.add(idx)
    return train_ids, val_ids, test_ids


def write_weak_label(label_path: Path, cls_id: int) -> None:
    label_path.write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def _load_image_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _find_plot_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 80).astype(np.uint8)
    h, w = gray.shape
    row_sums = np.sum(mask, axis=1)
    col_sums = np.sum(mask, axis=0)
    row_peaks = np.where(row_sums > w * 0.35)[0]
    col_peaks = np.where(col_sums > h * 0.35)[0]
    if len(row_peaks) >= 2 and len(col_peaks) >= 2:
        x1 = int(col_peaks[0]) + 1
        y1 = int(row_peaks[0]) + 1
        x2 = int(col_peaks[-1]) - 1
        y2 = int(row_peaks[-1]) - 1
        if (x2 - x1) > w * 0.2 and (y2 - y1) > h * 0.2:
            return x1, y1, x2, y2
    return 0, 0, w, h


def _deduplicate_yolo_lines(lines: list[str], decimals: int = 6) -> list[str]:
    unique: list[str] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        key = (
            int(float(parts[0])),
            round(float(parts[1]), decimals),
            round(float(parts[2]), decimals),
            round(float(parts[3]), decimals),
            round(float(parts[4]), decimals),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            f"{key[0]} {key[1]:.{decimals}f} {key[2]:.{decimals}f} {key[3]:.{decimals}f} {key[4]:.{decimals}f}"
        )
    return unique


def _mask_to_yolo_boxes(mask: np.ndarray, cls_id: int, min_area: int = 20) -> list[str]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    labels: list[str] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < min_area:
            continue
        xc = (x + bw / 2.0) / w
        yc = (y + bh / 2.0) / h
        wn = bw / float(w)
        hn = bh / float(h)
        labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return _deduplicate_yolo_lines(labels)


def _safe_mask(mask_data, mask_shape) -> np.ndarray | None:
    if mask_data is None or mask_shape is None:
        return None
    try:
        mask = np.array(mask_data, dtype=np.float32)
        if mask.ndim == 1 and len(mask_shape) == 2:
            mask = mask.reshape(mask_shape)
        if mask.ndim != 2:
            return None
        return mask
    except Exception:
        return None


def _refine_local_bbox_from_diff(
    defect_img: np.ndarray,
    ideal_img: np.ndarray | None,
    approx_bbox_xyxy: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if ideal_img is None:
        return None

    dx1, dy1, dx2, dy2 = _find_plot_bbox(defect_img)
    ix1, iy1, ix2, iy2 = _find_plot_bbox(ideal_img)
    defect_plot = defect_img[dy1:dy2, dx1:dx2]
    ideal_plot = ideal_img[iy1:iy2, ix1:ix2]
    if defect_plot.size == 0 or ideal_plot.size == 0:
        return None

    ideal_plot = cv2.resize(ideal_plot, (defect_plot.shape[1], defect_plot.shape[0]))
    diff = cv2.absdiff(defect_plot, ideal_plot)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    ax1, ay1, ax2, ay2 = approx_bbox_xyxy
    ax1 = max(dx1, ax1)
    ay1 = max(dy1, ay1)
    ax2 = min(dx2, ax2)
    ay2 = min(dy2, ay2)
    if ax2 <= ax1 or ay2 <= ay1:
        return None

    lx1 = max(0, ax1 - dx1)
    ly1 = max(0, ay1 - dy1)
    lx2 = min(diff_gray.shape[1], ax2 - dx1)
    ly2 = min(diff_gray.shape[0], ay2 - dy1)
    roi = diff_gray[ly1:ly2, lx1:lx2]
    if roi.size == 0:
        return None

    max_val = float(np.max(roi))
    mean_val = float(np.mean(roi))
    if max_val < 4.0:
        return None

    thresh_val = max(4, int(min(max_val * 0.55, mean_val + max(6.0, (max_val - mean_val) * 0.45))))
    _, binary = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)

    kernel_size = max(3, int(round(min(roi.shape[:2]) * 0.08)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_cx = (lx2 - lx1) / 2.0
    roi_cy = (ly2 - ly1) / 2.0
    best_rect = None
    best_score = -1.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 25:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        dist_score = 1.0 / (1.0 + np.hypot(cx - roi_cx, cy - roi_cy))
        score = area * dist_score
        if score > best_score:
            best_score = score
            best_rect = (x, y, w, h)

    if best_rect is None:
        return None

    x, y, w, h = best_rect
    pad = max(2, int(round(max(w, h) * 0.12)))
    bx1 = max(dx1, dx1 + lx1 + x - pad)
    by1 = max(dy1, dy1 + ly1 + y - pad)
    bx2 = min(dx2, dx1 + lx1 + x + w + pad)
    by2 = min(dy2, dy1 + ly1 + y + h + pad)
    if bx2 <= bx1 or by2 <= by1:
        return None
    return bx1, by1, bx2, by2


def preprocess_real_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    if img.ndim == 2:
        return enhanced
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def copy_image_maybe_preprocess(src: Path, dst: Path, preprocess: bool) -> None:
    if not preprocess:
        shutil.copy2(src, dst)
        return
    img = _load_image_unicode(src)
    if img is None:
        shutil.copy2(src, dst)
        return
    out = preprocess_real_image(img)
    ok, buf = cv2.imencode(dst.suffix or ".png", out)
    if not ok:
        shutil.copy2(src, dst)
        return
    buf.tofile(str(dst))


def _bbox_line_from_xyxy(cls_id: int, img_w: int, img_h: int, box: tuple[int, int, int, int]) -> str | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    wn = bw / float(img_w)
    hn = bh / float(img_h)
    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def generate_analytic_labels(
    defect_img_path: Path,
    ideal_img_path: Path | None,
    meta: dict,
    cls_id: int,
    label_path: Path,
) -> None:
    img = _load_image_unicode(defect_img_path)
    if img is None:
        write_weak_label(label_path, cls_id)
        return

    ideal_img = _load_image_unicode(ideal_img_path) if ideal_img_path and ideal_img_path.exists() else None
    img_h, img_w = img.shape[:2]
    sim_type = meta.get("sim_type", "wedge")
    labels: list[str] = []

    if cls_id in (1, 2):
        field_x = 0.006 if sim_type == "newton" else 0.030
        field_y = 0.006 if sim_type == "newton" else 0.015
        bump_count = int(meta.get("bump_count", 1))
        bump_radius_mm = float(meta.get("bump_radius_mm", 2.4))
        radius_m = max(0.00005, bump_radius_mm * 1e-3)
        positions = meta.get("generated_bump_positions", [])[:bump_count]
        if not positions:
            positions = meta.get("fixed_bump_positions", [])[:bump_count]

        for cx_m, cy_m in positions:
            cx_m = float(cx_m)
            cy_m = float(cy_m)
            if sim_type == "newton":
                px = (cx_m / field_x) + 0.5
                py = 1.0 - ((cy_m / field_y) + 0.5)
            else:
                px = cx_m / field_x
                py = 1.0 - (cy_m / field_y)
            pw = float(np.clip((2 * radius_m) / field_x, 0.01, 1.0))
            ph = float(np.clip((2 * radius_m) / field_y, 0.01, 1.0))
            px = float(np.clip(px, pw / 2, 1 - pw / 2))
            py = float(np.clip(py, ph / 2, 1 - ph / 2))

            bx1 = int(round((px - pw / 2) * img_w))
            by1 = int(round((py - ph / 2) * img_h))
            bx2 = int(round((px + pw / 2) * img_w))
            by2 = int(round((py + ph / 2) * img_h))
            refined = _refine_local_bbox_from_diff(img, ideal_img, (bx1, by1, bx2, by2))
            line = _bbox_line_from_xyxy(cls_id, img_w, img_h, refined if refined is not None else (bx1, by1, bx2, by2))
            if line:
                labels.append(line)

    elif cls_id == 4:
        hair_mask = _safe_mask(meta.get("hair_mask"), meta.get("hair_mask_shape"))
        if hair_mask is not None:
            if hair_mask.shape != (img_h, img_w):
                hair_mask = cv2.resize(hair_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            labels.extend(_mask_to_yolo_boxes((hair_mask > 0.1).astype(np.uint8), cls_id, min_area=12))

    labels = _deduplicate_yolo_lines(labels)
    if not labels:
        write_weak_label(label_path, cls_id)
        return
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def generate_yolo_labels_diff(
    defect_img_path: Path,
    ideal_img_path: Path | None,
    cls_id: int,
    label_path: Path,
    meta: dict | None = None,
) -> None:
    meta = meta or {}

    if cls_id == 0:
        scratch_mask = _safe_mask(meta.get("manual_scratch_mask"), meta.get("manual_scratch_mask_shape"))
        if scratch_mask is not None:
            defect_img = _load_image_unicode(defect_img_path)
            if defect_img is not None:
                h, w = defect_img.shape[:2]
                if scratch_mask.shape != (h, w):
                    scratch_mask = cv2.resize(scratch_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                labels = _mask_to_yolo_boxes((scratch_mask > 0.08).astype(np.uint8), cls_id, min_area=10)
                if labels:
                    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                    return

    if ideal_img_path is None or not ideal_img_path.exists():
        write_weak_label(label_path, cls_id)
        return

    try:
        defect_img = _load_image_unicode(defect_img_path)
        ideal_img = _load_image_unicode(ideal_img_path)
        if defect_img is None or ideal_img is None:
            write_weak_label(label_path, cls_id)
            return

        x1_d, y1_d, x2_d, y2_d = _find_plot_bbox(defect_img)
        x1_i, y1_i, x2_i, y2_i = _find_plot_bbox(ideal_img)
        defect_plot = defect_img[y1_d:y2_d, x1_d:x2_d]
        ideal_plot = ideal_img[y1_i:y2_i, x1_i:x2_i]
        if defect_plot.size == 0 or ideal_plot.size == 0:
            write_weak_label(label_path, cls_id)
            return

        ideal_plot = cv2.resize(ideal_plot, (defect_plot.shape[1], defect_plot.shape[0]))
        diff = cv2.absdiff(defect_plot, ideal_plot)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_val = 3 if cls_id == 3 else 8
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        img_short = min(defect_img.shape[:2])
        close_k = max(5, int(img_short * 0.015))
        dilate_k = max(6, int(img_short * 0.02))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        thresh = cv2.dilate(thresh, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_d, w_d = defect_img.shape[:2]
        labels: list[str] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w <= 6 or h <= 6 or w * h <= 100:
                continue
            x_full = x + x1_d
            y_full = y + y1_d
            padding = max(8, int(img_short * 0.01))
            box = (
                max(0, x_full - padding),
                max(0, y_full - padding),
                min(w_d, x_full + w + padding),
                min(h_d, y_full + h + padding),
            )
            line = _bbox_line_from_xyxy(cls_id, w_d, h_d, box)
            if line:
                labels.append(line)

        labels = _deduplicate_yolo_lines(labels)
        if not labels:
            write_weak_label(label_path, cls_id)
            return
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Error processing {defect_img_path.name}: {e}")
        write_weak_label(label_path, cls_id)


def _copy_real_label_with_cleanup(src_label: Path, dst_label: Path) -> None:
    try:
        lines = _deduplicate_yolo_lines(src_label.read_text(encoding="utf-8").splitlines())
        if not lines:
            dst_label.write_text("", encoding="utf-8")
        else:
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        shutil.copy2(src_label, dst_label)


def _load_meta_for_image(img_path: Path) -> dict:
    json_path = img_path.parent / img_path.name.replace("_有瑕疵干涉图.png", "_参数.json")
    if not json_path.exists():
        json_path = img_path.parent / f"{img_path.stem}_参数.json"
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def copy_and_label(
    samples: list[tuple[Path, Path, int, str]],
    output_dir: Path,
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
    preprocess_real: bool = False,
) -> None:
    for idx, (img_path, ideal_path, cls_id, sample_sim_type) in enumerate(samples):
        if idx in train_ids:
            split = "train"
        elif idx in val_ids:
            split = "val"
        else:
            split = "test"

        out_img = output_dir / "images" / split / img_path.name
        out_lbl = output_dir / "labels" / split / f"{img_path.stem}.txt"

        copy_image_maybe_preprocess(
            img_path,
            out_img,
            preprocess=bool(preprocess_real and sample_sim_type == "real"),
        )

        meta = _load_meta_for_image(img_path)
        sim_type = meta.get("sim_type", sample_sim_type)

        if sim_type == "real":
            pre_label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if pre_label.exists():
                _copy_real_label_with_cleanup(pre_label, out_lbl)
            else:
                write_weak_label(out_lbl, cls_id)
        elif cls_id in (0, 3):
            generate_yolo_labels_diff(img_path, ideal_path, cls_id, out_lbl, meta)
        else:
            generate_analytic_labels(img_path, ideal_path, meta, cls_id, out_lbl)


def write_small_target_manifest(
    samples: list[tuple[Path, Path, int, str]],
    output_dir: Path,
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
) -> None:
    records: list[dict] = []
    for idx, (img_path, _ideal_path, cls_id, sample_sim_type) in enumerate(samples):
        if cls_id not in (1, 2):
            continue
        meta = _load_meta_for_image(img_path)
        bump_height = float(meta.get("bump_height", 999))
        bump_radius_mm = float(meta.get("bump_radius_mm", 999))
        if bump_height > 0.08 and bump_radius_mm > 3.0:
            continue
        split = "train" if idx in train_ids else ("val" if idx in val_ids else "test")
        records.append(
            {
                "image": f"images/{split}/{img_path.name}",
                "label": f"labels/{split}/{img_path.stem}.txt",
                "split": split,
                "class_id": cls_id,
                "class_name": "bump" if cls_id == 1 else "dent",
                "sim_type": meta.get("sim_type", sample_sim_type),
                "bump_height": bump_height,
                "bump_radius_mm": bump_radius_mm,
                "expected_count": int(meta.get("bump_count", 1)),
            }
        )
    (output_dir / "metadata" / "small_target_eval.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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

    train_ids, val_ids, test_ids = split_indices(
        samples,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.max_real_train_ratio,
    )
    copy_and_label(
        samples,
        output_dir,
        train_ids,
        val_ids,
        test_ids,
        preprocess_real=args.preprocess_real,
    )
    write_small_target_manifest(samples, output_dir, train_ids, val_ids, test_ids)
    write_yaml(output_dir)

    print("Split finished")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Total: {len(samples)}")
    print(f"Train: {len(train_ids)}")
    print(f"Val: {len(val_ids)}")
    print(f"Test: {len(test_ids)}")
    print(f"YAML: {output_dir / 'dataset.yaml'}")
    print(f"Small-target manifest: {output_dir / 'metadata' / 'small_target_eval.json'}")


if __name__ == "__main__":
    main()
