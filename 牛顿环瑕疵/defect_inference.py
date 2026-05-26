from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure sibling module defect_ai_module is importable when running this folder directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from defect_ai_module import DefectAIService  # noqa: E402


class DefectInferenceEngine:
    """Compatibility wrapper for desktop window integration."""

    def __init__(self, model_path: str | None = None) -> None:
        self._service = DefectAIService(model_path)

    @property
    def model_path(self) -> Path:
        return self._service.model_path

    def set_model_path(self, model_path: str) -> None:
        self._service.set_model_path(model_path)

    def predict_image(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        **kwargs,
    ) -> dict[str, Any]:
        result = self._service.predict_image(
            image_path=image_path,
            conf=conf,
            iou=iou,
            device=device,
            save_visualization=True,
            **kwargs,
        )
        return result.to_legacy_dict()
