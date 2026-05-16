from __future__ import annotations

import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from .types import DefectDetectionResult, DefectPrediction


def get_default_model_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "dl_main_model_yolo"
        / "runs"
        / "train"
        / "yolo11s_final_split3"
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
    def _label_to_cn(label: str) -> str:
        # 注意：模型训练时 bump(cls=1) 对应小凹陷视觉特征，dent(cls=2) 对应小凸起视觉特征
        # 这是因为物理仿真中"小突起"使间隙减小(depth_sign=-1)，干涉条纹向外弯曲，
        # 与"小凹陷"使间隙增大(depth_sign=+1)，条纹向内弯曲，两者视觉特征与直觉相反。
        # 经验证，交换 bump/dent 的中文标签可修正检测结果。
        mapping = {
            "scratch": "表面划痕",
            "bump": "小凹陷",   # 修正：模型学到的 bump 特征实为凹陷
            "dent": "小凸起",   # 修正：模型学到的 dent 特征实为凸起
            "dust": "灰尘污染",
            "hair": "头发丝遮挡",
        }
        return mapping.get(label, label)

    def predict_image(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        save_visualization: bool = True,
        output_image_path: str | None = None,
    ) -> DefectDetectionResult:
        model = self._load_model()
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        try:
            results = model.predict(
                source=str(image_path_obj),
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
                save=False,
            )
        except Exception:
            # GPU runtime can fail on some machines; fallback to CPU keeps service usable.
            results = model.predict(
                source=str(image_path_obj),
                conf=conf,
                iou=iou,
                device="cpu",
                verbose=False,
                save=False,
            )

        result = results[0]
        names = result.names if isinstance(result.names, dict) else {}
        boxes = result.boxes

        predictions: list[DefectPrediction] = []
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, f"class_{cls_id}"))
                coords = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
                conf_val = float(box.conf.item())
                # TODO 4.1：过滤低置信度结果，避免把不确定的猜测当作检测结果输出
                # 35% 以下视为未检测到，防止出现 29% 置信度的误判
                if conf_val < 0.35:
                    continue
                predictions.append(
                    DefectPrediction(
                        cls_id=cls_id,
                        label=label,
                        label_cn=self._label_to_cn(label),
                        confidence=conf_val,
                        bbox_xyxy=coords,
                    )
                )
            predictions.sort(key=lambda x: x.confidence, reverse=True)

        rendered_path = ""
        if save_visualization:
            vis_image = result.plot()
            if output_image_path:
                out_path = Path(output_image_path)
            else:
                out_path = image_path_obj.with_name(f"{image_path_obj.stem}_检测结果.png")
            cv2.imwrite(str(out_path), vis_image)
            rendered_path = str(out_path)

        return DefectDetectionResult(predictions=predictions, output_image=rendered_path)
