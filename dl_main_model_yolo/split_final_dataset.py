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
    parser.add_argument(
        "--sim-types",
        nargs="+",
        choices=("wedge", "newton", "real"),
        default=None,
        help="Optional experiment-domain filter, for example: --sim-types wedge.",
    )
    parser.add_argument("--min-wavelength-nm", type=float, default=None)
    parser.add_argument("--max-wavelength-nm", type=float, default=None)
    parser.add_argument(
        "--balance-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample only valid, underrepresented training classes after labels are built.",
    )
    parser.add_argument(
        "--max-boxes-per-image",
        type=int,
        default=12,
        help="Maximum retained boxes of a single class in one image; 0 disables the cap.",
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


def collect_samples(
    source_dir: Path,
    allowed_sim_types: set[str] | None = None,
    min_wavelength_nm: float | None = None,
    max_wavelength_nm: float | None = None,
) -> list[tuple[Path, Path, int, str]]:
    samples: list[tuple[Path, Path, int, str]] = []
    skipped_unknown = 0
    for json_path in sorted(source_dir.glob("*_参数.json")):
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        sim_type = meta.get("sim_type", "unknown")
        if sim_type == "unknown":
            skipped_unknown += 1
            continue
        if allowed_sim_types and sim_type not in allowed_sim_types:
            continue
        if min_wavelength_nm is not None or max_wavelength_nm is not None:
            try:
                wavelength_nm = float(meta["lambda_nm"])
            except (KeyError, TypeError, ValueError):
                continue
            if min_wavelength_nm is not None and wavelength_nm < min_wavelength_nm:
                continue
            if max_wavelength_nm is not None and wavelength_nm > max_wavelength_nm:
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
    """Record that local labeling failed without fabricating a full-image defect box.

    Full-image fallback boxes caused the detector to learn that broad fringe changes
    are hair/scratch/dust.  The caller removes the corresponding image from the
    split, so an empty file is deliberately different from a negative sample.
    """
    label_path.write_text("", encoding="utf-8")


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


def _limit_boxes_per_class(lines: list[str], max_boxes_per_class: int) -> list[str]:
    """Keep spatially largest boxes when one dense class overwhelms an image."""
    if max_boxes_per_class <= 0:
        return lines

    grouped: dict[int, list[tuple[float, str]]] = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            area = float(parts[3]) * float(parts[4])
        except ValueError:
            continue
        grouped.setdefault(cls_id, []).append((area, line))

    limited: list[str] = []
    for cls_id in sorted(grouped):
        entries = sorted(grouped[cls_id], reverse=True)
        limited.extend(line for _area, line in entries[:max_boxes_per_class])
    return limited


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


def _diff_gray_with_bias_correction(
    defect_img: np.ndarray,
    ideal_img: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    """Plot-aligned absdiff with global-median bias removal.

    The generator now applies identical augmentations to both images, so the
    diff should be defect-only.  The median subtraction is a defence in depth
    against any residual linear (brightness/contrast) bias that could push
    whole fringe bands above the labelling threshold.

    Returns (diff_gray, (dx1, dy1)) where diff_gray is the blurred difference
    of the aligned plot regions and (dx1, dy1) is the plot offset inside
    defect_img; (None, None) when the plots cannot be aligned.
    """
    dx1, dy1, dx2, dy2 = _find_plot_bbox(defect_img)
    ix1, iy1, ix2, iy2 = _find_plot_bbox(ideal_img)
    defect_plot = defect_img[dy1:dy2, dx1:dx2]
    ideal_plot = ideal_img[iy1:iy2, ix1:ix2]
    if defect_plot.size == 0 or ideal_plot.size == 0:
        return None, None

    ideal_plot = cv2.resize(ideal_plot, (defect_plot.shape[1], defect_plot.shape[0]))
    diff = cv2.cvtColor(cv2.absdiff(defect_plot, ideal_plot), cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    bias = float(np.median(diff))
    if bias > 0.5:
        diff = np.clip(diff.astype(np.float32) - bias, 0, 255).astype(np.uint8)

    return diff, (dx1, dy1)


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


def _parse_local_target_geometry(meta: dict) -> tuple[int, list[list[float]]]:
    """Read strict generator-owned geometry for bump/dent labels.

    A fallback radius, a fixed default position, or a diff-only guess would
    silently create wrong supervision.  New datasets must therefore carry the
    versioned target geometry written by auto_dataset_generator.py.
    """
    if meta.get("target_geometry_version") != 3:
        raise ValueError("missing target_geometry_version=3; regenerate the source dataset")

    try:
        target_count = int(meta["target_count"])
        bump_count = int(meta["bump_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("target_count and bump_count are required integers") from exc
    if target_count <= 0 or target_count != bump_count:
        raise ValueError(
            f"target_count ({target_count}) does not match bump_count ({bump_count})"
        )

    boxes = meta.get("target_bboxes_xyxy_normalized")
    positions = meta.get("target_positions_mm")
    if not isinstance(boxes, list) or len(boxes) != target_count:
        raise ValueError(
            "target_bboxes_xyxy_normalized count does not match target_count"
        )
    if not isinstance(positions, list) or len(positions) != target_count:
        raise ValueError("target_positions_mm count does not match target_count")

    parsed_boxes: list[list[float]] = []
    for index, raw_box in enumerate(boxes):
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise ValueError(f"target box {index} must contain four normalized coordinates")
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_box)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target box {index} contains a non-numeric coordinate") from exc
        if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f"target box {index} contains a non-finite coordinate")
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"target box {index} is outside [0,1] or has non-positive area")
        parsed_boxes.append([x1, y1, x2, y2])

    try:
        target_radius_mm = float(meta["target_radius_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("target_radius_mm is required") from exc
    if not np.isfinite(target_radius_mm) or target_radius_mm <= 0:
        raise ValueError("target_radius_mm must be a positive finite number")
    try:
        target_box_radius_scale = float(meta["target_box_radius_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("target_box_radius_scale is required") from exc
    if not np.isfinite(target_box_radius_scale) or not 0.0 < target_box_radius_scale <= 1.0:
        raise ValueError("target_box_radius_scale must be in (0,1]")

    return target_count, parsed_boxes


def _yolo_line_from_normalized_box(cls_id: int, box: list[float]) -> str:
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return f"{cls_id} {xc:.6f} {yc:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def generate_analytic_labels(
    defect_img_path: Path,
    ideal_img_path: Path | None,
    meta: dict,
    cls_id: int,
    label_path: Path,
) -> None:
    if cls_id in (1, 2):
        target_count, target_boxes = _parse_local_target_geometry(meta)
    else:
        target_count, target_boxes = 0, []

    img = _load_image_unicode(defect_img_path)
    if img is None:
        write_weak_label(label_path, cls_id)
        return

    img_h, img_w = img.shape[:2]
    labels: list[str] = []

    if cls_id in (1, 2):
        labels = [_yolo_line_from_normalized_box(cls_id, box) for box in target_boxes]
        if len(labels) != target_count:
            raise ValueError(
                f"generated {len(labels)} local labels, expected {target_count}"
            )

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

        diff_gray, plot_offset = _diff_gray_with_bias_correction(defect_img, ideal_img)
        if diff_gray is None:
            write_weak_label(label_path, cls_id)
            return
        dx1, dy1 = plot_offset

        # Hair is a thin dark occlusion.  Use a stricter difference threshold than
        # dust, while preserving its disconnected curve segments for local boxes.
        # 阈值下限统一提升到 8/12：低于此值的差分多数为残留噪声而非可见缺陷。
        thresh_val = 8 if cls_id == 3 else 12
        _, thresh = cv2.threshold(diff_gray, thresh_val, 255, cv2.THRESH_BINARY)

        img_short = min(defect_img.shape[:2])
        if cls_id == 4:
            close_k = max(3, int(img_short * 0.006))
            dilate_k = max(3, int(img_short * 0.008))
        else:
            close_k = max(5, int(img_short * 0.015))
            dilate_k = max(6, int(img_short * 0.02))
        if close_k % 2 == 0:
            close_k += 1
        if dilate_k % 2 == 0:
            dilate_k += 1
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        thresh = cv2.dilate(thresh, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_d, w_d = defect_img.shape[:2]
        labels: list[str] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            min_area = 24 if cls_id == 4 else 100
            if w <= 3 or h <= 3 or w * h <= min_area:
                continue
            x_full = x + dx1
            y_full = y + dy1
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
    max_boxes_per_image: int = 12,
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
        elif cls_id in (0, 3, 4):
            generate_yolo_labels_diff(img_path, ideal_path, cls_id, out_lbl, meta)
        else:
            try:
                generate_analytic_labels(img_path, ideal_path, meta, cls_id, out_lbl)
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                raise RuntimeError(
                    f"Invalid local-defect metadata for {img_path.name}: {exc}"
                ) from exc

        # Never turn an unlocalised defect into a negative or a full-image box.
        # Such samples are not usable for object detection and must not reach YOLO.
        lines = _deduplicate_yolo_lines(out_lbl.read_text(encoding="utf-8").splitlines())
        if cls_id in (1, 2):
            expected_count, _ = _parse_local_target_geometry(meta)
            if len(lines) != expected_count:
                raise RuntimeError(
                    f"Local label count mismatch for {img_path.name}: "
                    f"labels={len(lines)}, expected={expected_count}"
                )
            if max_boxes_per_image > 0 and expected_count > max_boxes_per_image:
                raise ValueError(
                    f"max-boxes-per-image={max_boxes_per_image} is below local target count "
                    f"{expected_count} for {img_path.name}"
                )
        lines = _limit_boxes_per_class(lines, max_boxes_per_image)
        if not lines:
            out_lbl.unlink(missing_ok=True)
            out_img.unlink(missing_ok=True)
            continue
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebalance_training_split(output_dir: Path, seed: int) -> dict[int, int]:
    """Duplicate valid minority-class training samples without changing evaluation data."""
    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    by_class: dict[int, list[tuple[Path, Path]]] = {idx: [] for idx in range(5)}

    for label_path in sorted(label_dir.glob("*.txt")):
        lines = _deduplicate_yolo_lines(label_path.read_text(encoding="utf-8").splitlines())
        classes = {int(line.split()[0]) for line in lines if line.split()}
        image_path = image_dir / f"{label_path.stem}.png"
        if not image_path.exists():
            matches = list(image_dir.glob(f"{label_path.stem}.*"))
            image_path = matches[0] if matches else image_path
        if not image_path.exists():
            continue
        for cls_id in classes:
            by_class.setdefault(cls_id, []).append((image_path, label_path))

    # Dust is naturally multi-instance.  Balance against the largest non-dust
    # class so it cannot dominate classification gradients by object count.
    minority_sizes = [len(by_class[idx]) for idx in (0, 1, 2, 4) if by_class[idx]]
    if not minority_sizes:
        return {idx: len(items) for idx, items in by_class.items()}
    target = min(700, max(minority_sizes))
    rng = random.Random(seed)

    for cls_id in (0, 1, 2, 4):
        samples = by_class[cls_id]
        if not samples or len(samples) >= target:
            continue
        needed = target - len(samples)
        for repeat_idx in range(needed):
            src_image, src_label = samples[repeat_idx % len(samples)]
            suffix = f"__balanced_{cls_id}_{repeat_idx:04d}"
            dst_image = image_dir / f"{src_image.stem}{suffix}{src_image.suffix}"
            dst_label = label_dir / f"{src_label.stem}{suffix}.txt"
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_label, dst_label)
        rng.shuffle(samples)

    return {idx: len(items) for idx, items in by_class.items()}


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
        expected_count, target_boxes = _parse_local_target_geometry(meta)
        try:
            bump_height = float(meta["bump_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bump_height is required for {img_path.name}") from exc
        bump_radius_mm = float(meta["target_radius_mm"])
        if bump_height > 0.08 and bump_radius_mm > 3.0:
            continue
        split = "train" if idx in train_ids else ("val" if idx in val_ids else "test")
        output_label = output_dir / "labels" / split / f"{img_path.stem}.txt"
        output_image = output_dir / "images" / split / img_path.name
        if not output_label.exists() or not output_image.exists():
            continue
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
                "expected_count": expected_count,
                "target_positions_mm": meta["target_positions_mm"],
                "target_bboxes_xyxy_normalized": target_boxes,
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

    allowed_sim_types = set(args.sim_types) if args.sim_types else None
    samples = collect_samples(
        source_dir,
        allowed_sim_types,
        args.min_wavelength_nm,
        args.max_wavelength_nm,
    )
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
        max_boxes_per_image=args.max_boxes_per_image,
    )
    class_counts = {}
    if args.balance_train:
        class_counts = rebalance_training_split(output_dir, args.seed)
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
    if class_counts:
        print(f"Valid training images before balancing: {class_counts}")


if __name__ == "__main__":
    main()
