from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .types import DefectDetectionResult, DefectPrediction


# 尝试加载系统中文字体，找不到时回退到 PIL 默认字体
def _load_cn_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def get_default_model_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "dl_main_model_yolo"
        / "runs"
        / "train"
        / "yolo11s_final_split92"
        / "weights"
        / "best.pt"
    )


class DefectAIService:
    """UI-independent inference service for desktop and future web adapters."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path or os.getenv("DEFECT_MODEL_PATH", str(get_default_model_path())))
        self._model: YOLO | None = None

    def set_model_path(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self._model = None

    def _load_model(self) -> YOLO:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    @staticmethod
    def _find_plot_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
        """
        从带坐标轴的干涉图中提取纯干涉图区域的边界框。

        策略：寻找图像中连续的深色边框线（坐标轴边框），
        找到后返回绘图区内边界。若找不到明显边框则返回全图。

        Returns:
            (x1, y1, x2, y2) 绘图区在原图中的像素坐标
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 深色像素掩码（坐标轴线、边框通常是黑色或深灰）
        dark_mask = (gray < 80).astype(np.uint8)

        row_sums = np.sum(dark_mask, axis=1)  # 每行深色像素数
        col_sums = np.sum(dark_mask, axis=0)  # 每列深色像素数

        # 边框线：深色像素连续占该行/列宽度的 35% 以上
        row_threshold = w * 0.35
        col_threshold = h * 0.35

        row_border = np.where(row_sums > row_threshold)[0]
        col_border = np.where(col_sums > col_threshold)[0]

        if len(row_border) >= 2 and len(col_border) >= 2:
            # 取最外侧边框线，向内缩 1px 避免把边框本身送入模型
            x1 = int(col_border[0]) + 1
            y1 = int(row_border[0]) + 1
            x2 = int(col_border[-1]) - 1
            y2 = int(row_border[-1]) - 1

            # 裁剪区域至少要有原图 20% 的面积，否则认为没找到有效边框
            if (x2 - x1) > w * 0.2 and (y2 - y1) > h * 0.2:
                return x1, y1, x2, y2

        # 找不到明显边框：尝试用边缘检测找最大矩形轮廓
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest)
            if rw > w * 0.3 and rh > h * 0.3:
                return rx, ry, rx + rw, ry + rh

        # 兜底：返回全图
        return 0, 0, w, h

    @staticmethod
    def _remap_bbox(
        bbox_xyxy: list[float],
        crop_x1: int,
        crop_y1: int,
        orig_w: int,
        orig_h: int,
    ) -> list[float]:
        """将裁剪图上的检测框坐标映射回原图坐标。"""
        x1, y1, x2, y2 = bbox_xyxy
        return [
            x1 + crop_x1,
            y1 + crop_y1,
            x2 + crop_x1,
            y2 + crop_y1,
        ]

    @staticmethod
    def _label_to_cn(label: str) -> str:
        # 直接映射：训练数据中 bump(cls=1)=小凸起，dent(cls=2)=小凹陷
        mapping = {
            "scratch": "表面划痕",
            "bump":    "小凸起",
            "dent":    "小凹陷",
            "dust":    "灰尘污染",
            "hair":    "头发丝遮挡",
        }
        return mapping.get(label, label)

    @staticmethod
    def _bbox_iou(box1: list[float] | None, box2: list[float] | None) -> float:
        if not box1 or not box2:
            return 0.0
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _accept_prediction(label: str, conf_val: float, bbox_xyxy: list[float] | None, image_shape: tuple[int, int]) -> bool:
        h, w = image_shape
        per_class_min_conf = {
            "scratch": 0.30,
            "bump": 0.18,
            "dent": 0.18,
            "dust": 0.28,
            "hair": 0.40,
        }
        if conf_val < per_class_min_conf.get(label, 0.30):
            return False

        if label == "hair" and bbox_xyxy:
            x1, y1, x2, y2 = bbox_xyxy
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            area_ratio = (bw * bh) / float(max(1, w * h))
            if aspect < 2.2 and area_ratio < 0.35:
                return False
        return True

    def _postprocess_predictions(self, predictions: list[DefectPrediction], image_shape: tuple[int, int]) -> list[DefectPrediction]:
        accepted: list[DefectPrediction] = []
        for pred in predictions:
            if self._accept_prediction(pred.label, pred.confidence, pred.bbox_xyxy, image_shape):
                accepted.append(pred)

        # 类别优先：若局部 bump/dent 与 hair 重叠，优先保留 bump/dent，避免小凸起被误判成头发丝
        refined: list[DefectPrediction] = []
        local_targets = [p for p in accepted if p.label in {"bump", "dent"}]
        for pred in accepted:
            if pred.label == "hair":
                if any(self._bbox_iou(pred.bbox_xyxy, other.bbox_xyxy) > 0.2 for other in local_targets):
                    continue
            refined.append(pred)
        refined.sort(key=lambda x: x.confidence, reverse=True)
        return refined

    def predict_image(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        save_visualization: bool = True,
        output_image_path: str | None = None,
        crop_plot_area: bool = True,
    ) -> DefectDetectionResult:
        """
        对单张图片进行瑕疵检测。

        Args:
            crop_plot_area: 是否在推理前自动裁剪掉坐标轴/标题等装饰元素，
                            只将纯干涉图区域送入模型。默认开启，可避免把
                            坐标轴线条误识别为 dust/scratch。
        """
        model = self._load_model()
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # ── 读取原图 ──────────────────────────────────────────────
        orig_data = np.fromfile(str(image_path_obj), dtype=np.uint8)
        orig_img = cv2.imdecode(orig_data, cv2.IMREAD_COLOR)
        if orig_img is None:
            raise ValueError(f"Cannot decode image: {image_path}")

        orig_h, orig_w = orig_img.shape[:2]

        # ── 裁剪纯干涉图区域 ──────────────────────────────────────
        crop_x1, crop_y1 = 0, 0
        infer_img = orig_img  # 默认使用原图

        if crop_plot_area:
            cx1, cy1, cx2, cy2 = self._find_plot_bbox(orig_img)
            # 只有当裁剪区域明显小于原图时才真正裁剪（说明找到了坐标轴）
            area_ratio = ((cx2 - cx1) * (cy2 - cy1)) / (orig_w * orig_h)
            if area_ratio < 0.92:
                crop_x1, crop_y1 = cx1, cy1
                infer_img = orig_img[cy1:cy2, cx1:cx2]

        # ── 将裁剪图写入临时文件供 YOLO 推理 ─────────────────────
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            cv2.imwrite(tmp_path, infer_img)

            def _run_predict(dev: str):
                return model.predict(
                    source=tmp_path,
                    conf=conf,
                    iou=iou,
                    imgsz=960,
                    device=dev,
                    verbose=False,
                    save=False,
                )

            try:
                results = _run_predict(device)
            except Exception:
                # GPU 失败时回退 CPU
                results = _run_predict("cpu")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        result = results[0]
        names = result.names if isinstance(result.names, dict) else {}
        boxes = result.boxes

        raw_predictions: list[DefectPrediction] = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, f"class_{cls_id}"))
                conf_val = float(box.conf.item())
                # 将检测框坐标从裁剪图映射回原图
                coords_on_crop = box.xyxy[0].tolist() if hasattr(box, "xyxy") else None
                if coords_on_crop is not None and (crop_x1 > 0 or crop_y1 > 0):
                    coords = self._remap_bbox(coords_on_crop, crop_x1, crop_y1, orig_w, orig_h)
                else:
                    coords = coords_on_crop

                raw_predictions.append(
                    DefectPrediction(
                        cls_id=cls_id,
                        label=label,
                        label_cn=self._label_to_cn(label),
                        confidence=conf_val,
                        bbox_xyxy=coords,
                    )
                )
        predictions = self._postprocess_predictions(raw_predictions, (orig_h, orig_w))

        # 弱目标兜底：首轮没有明显目标时，再用更低阈值做一次 rescue pass，优先找 bump/dent
        if not predictions:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()
            try:
                cv2.imwrite(tmp_path, infer_img)

                def _run_rescue(dev: str):
                    return model.predict(
                        source=tmp_path,
                        conf=min(conf, 0.10),
                        iou=iou,
                        imgsz=1280,
                        device=dev,
                        verbose=False,
                        save=False,
                    )

                try:
                    rescue_results = _run_rescue(device)
                except Exception:
                    rescue_results = _run_rescue("cpu")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            rescue = rescue_results[0]
            rescue_boxes = rescue.boxes
            rescue_raw: list[DefectPrediction] = []
            if rescue_boxes is not None and len(rescue_boxes) > 0:
                for box in rescue_boxes:
                    cls_id = int(box.cls.item())
                    label = str(names.get(cls_id, f"class_{cls_id}"))
                    if label not in {"bump", "dent"}:
                        continue
                    conf_val = float(box.conf.item())
                    coords_on_crop = box.xyxy[0].tolist() if hasattr(box, "xyxy") else None
                    if coords_on_crop is not None and (crop_x1 > 0 or crop_y1 > 0):
                        coords = self._remap_bbox(coords_on_crop, crop_x1, crop_y1, orig_w, orig_h)
                    else:
                        coords = coords_on_crop
                    rescue_raw.append(
                        DefectPrediction(
                            cls_id=cls_id,
                            label=label,
                            label_cn=self._label_to_cn(label),
                            confidence=conf_val,
                            bbox_xyxy=coords,
                        )
                    )
            predictions = self._postprocess_predictions(rescue_raw, (orig_h, orig_w))

        # ── 在原图上绘制检测框并保存可视化结果 ───────────────────
        rendered_path = ""
        if save_visualization:
            vis_image = orig_img.copy()

            # 框颜色：按类别区分（BGR）
            color_map = {
                "scratch": (0, 165, 255),   # 橙色
                "bump":    (0, 255, 0),      # 绿色
                "dent":    (255, 0, 0),      # 蓝色
                "dust":    (0, 255, 255),    # 黄色
                "hair":    (255, 0, 255),    # 紫色
            }

            # 先用 cv2 画矩形框
            for pred in predictions:
                if pred.bbox_xyxy:
                    x1, y1, x2, y2 = [int(v) for v in pred.bbox_xyxy]
                    color = color_map.get(pred.label, (0, 200, 200))
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 转 PIL 绘制中文标签（cv2.putText 不支持中文）
            pil_img = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = _load_cn_font(18)

            for pred in predictions:
                if pred.bbox_xyxy:
                    x1, y1, x2, y2 = [int(v) for v in pred.bbox_xyxy]
                    color_bgr = color_map.get(pred.label, (0, 200, 200))
                    # PIL 用 RGB
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                    label_text = f"{pred.label_cn} {pred.confidence:.2f}"

                    # 计算文字尺寸
                    bbox_text = draw.textbbox((0, 0), label_text, font=font)
                    tw = bbox_text[2] - bbox_text[0]
                    th = bbox_text[3] - bbox_text[1]

                    # 标签背景矩形
                    bg_y1 = max(0, y1 - th - 6)
                    draw.rectangle(
                        [(x1, bg_y1), (x1 + tw + 6, y1)],
                        fill=color_rgb,
                    )
                    # 文字（黑色）
                    draw.text((x1 + 3, bg_y1 + 1), label_text, font=font, fill=(0, 0, 0))

            # 转回 BGR numpy
            vis_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if output_image_path:
                out_path = Path(output_image_path)
            else:
                out_path = image_path_obj.with_name(f"{image_path_obj.stem}_检测结果.png")

            # imencode + tofile 支持中文路径
            _, buf = cv2.imencode(".png", vis_image)
            buf.tofile(str(out_path))
            rendered_path = str(out_path)

        return DefectDetectionResult(predictions=predictions, output_image=rendered_path)
