# -*- coding: utf-8 -*-
"""
defect_platform.ai_service — AI 推理服务（UI 无关）
====================================================
对 defect_ai_module.DefectAIService 的薄封装，
统一对外命名（小凸起/小凹陷），并提供网页友好的接口。
"""

from __future__ import annotations

import os
from pathlib import Path

from .types import DetectionResult

# 延迟导入，避免在没有 ultralytics 的环境中报错
_service_instance = None


def _get_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_model_path() -> Path:
    return (
        _get_root()
        / "dl_main_model_yolo"
        / "runs"
        / "train"
        / "yolo11s_final_split2"
        / "weights"
        / "best.pt"
    )


class DefectAIService:
    """
    AI 瑕疵检测服务（UI 无关）

    可用于桌面端、命令行，以及未来的网页后端（Flask/FastAPI）。

    示例（网页后端）
    ----------------
    >>> ai = DefectAIService()
    >>> result = ai.predict_image("path/to/defect.png")
    >>> print(result.label_cn, result.confidence)

    示例（Flask 路由）
    ------------------
    >>> @app.route("/detect", methods=["POST"])
    >>> def detect():
    >>>     img_path = save_uploaded_file(request.files["image"])
    >>>     result = ai.predict_image(img_path)
    >>>     return jsonify(result.to_dict())
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = Path(model_path or os.getenv("DEFECT_MODEL_PATH", str(get_default_model_path())))
        self._inner: object | None = None

    def _get_inner(self):
        """懒加载底层服务"""
        if self._inner is None:
            import sys
            root = str(_get_root())
            if root not in sys.path:
                sys.path.insert(0, root)
            from defect_ai_module import DefectAIService as _Inner
            self._inner = _Inner(str(self._model_path))
        return self._inner

    def set_model_path(self, model_path: str) -> None:
        """切换模型权重"""
        self._model_path = Path(model_path)
        self._inner = None  # 重置，下次调用时重新加载

    @property
    def model_path(self) -> Path:
        return self._model_path

    def predict_image(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        save_visualization: bool = True,
        output_image_path: str | None = None,
    ) -> DetectionResult | None:
        """
        对单张图片执行瑕疵检测。

        参数
        ----
        image_path : str
            待检测图片路径。
        conf : float
            置信度阈值（默认 0.25）。
        iou : float
            NMS IoU 阈值（默认 0.7）。
        device : str
            推理设备（"0" 表示 GPU 0，"cpu" 表示 CPU）。
        save_visualization : bool
            是否保存带检测框的结果图（默认 True）。
        output_image_path : str, optional
            结果图保存路径，不指定则自动生成。

        返回
        ----
        DetectionResult | None
            置信度最高的检测结果，无检测目标时返回 None。
        """
        inner = self._get_inner()
        raw = inner.predict_image(
            image_path=image_path,
            conf=conf,
            iou=iou,
            device=device,
            save_visualization=save_visualization,
            output_image_path=output_image_path,
        )

        top = raw.top_prediction
        if top is None:
            return None

        return DetectionResult(
            label=top.label,
            label_cn=top.label_cn,
            confidence=top.confidence,
            cls_id=top.cls_id,
            bbox_xyxy=top.bbox_xyxy,
            output_image_path=raw.output_image,
        )

    def predict_image_all(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        save_visualization: bool = True,
        output_image_path: str | None = None,
    ) -> list[DetectionResult]:
        """
        返回所有检测结果（按置信度降序），而非仅最高置信度的结果。
        适合需要展示多个检测框的场景。
        """
        inner = self._get_inner()
        raw = inner.predict_image(
            image_path=image_path,
            conf=conf,
            iou=iou,
            device=device,
            save_visualization=save_visualization,
            output_image_path=output_image_path,
        )

        results = []
        for pred in raw.predictions:
            results.append(
                DetectionResult(
                    label=pred.label,
                    label_cn=pred.label_cn,
                    confidence=pred.confidence,
                    cls_id=pred.cls_id,
                    bbox_xyxy=pred.bbox_xyxy,
                    output_image_path=raw.output_image,
                )
            )
        return results
