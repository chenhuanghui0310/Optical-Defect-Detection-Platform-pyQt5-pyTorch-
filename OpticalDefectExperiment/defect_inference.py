from __future__ import annotations

from pathlib import Path
from typing import Any

from defect_ai_module import DefectAIService


class DefectInferenceEngine:
    """Desktop wrapper around shared AI service for backward compatibility."""

    def __init__(self, model_path: str | None = None) -> None:
        self._service = DefectAIService(model_path)

    @property
    def model_path(self) -> Path:
        return self._service.model_path

    def set_model_path(self, model_path: str) -> None:
        self._service.set_model_path(model_path)

    @staticmethod
    def _label_to_cn(label: str) -> str:
        mapping = {
            "scratch": "表面划痕",
            "bump": "小突起",
            "dent": "小凹陷",
            "dust": "灰尘污染",
        }
        return mapping.get(label, label)

    def predict_image(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
    ) -> dict[str, Any]:
        result = self._service.predict_image(
            image_path=image_path,
            conf=conf,
            iou=iou,
            device=device,
            save_visualization=True,
        )
        return result.to_legacy_dict()
