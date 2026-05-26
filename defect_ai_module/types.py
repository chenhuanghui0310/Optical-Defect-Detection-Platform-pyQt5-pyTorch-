from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DefectPrediction:
    cls_id: int
    label: str
    label_cn: str
    confidence: float
    bbox_xyxy: list[float] | None = None


@dataclass
class DefectDetectionResult:
    predictions: list[DefectPrediction]
    output_image: str
    summary: dict | None = None

    @property
    def top_prediction(self) -> DefectPrediction | None:
        return self.predictions[0] if self.predictions else None

    def to_legacy_dict(self) -> dict:
        top = self.top_prediction
        return {
            "predictions": [
                {
                    "cls_id": p.cls_id,
                    "label": p.label,
                    "label_cn": p.label_cn,
                    "confidence": p.confidence,
                    "bbox_xyxy": p.bbox_xyxy,
                }
                for p in self.predictions
            ],
            "top_prediction": {
                "cls_id": top.cls_id,
                "label": top.label,
                "label_cn": top.label_cn,
                "confidence": top.confidence,
                "bbox_xyxy": top.bbox_xyxy,
            }
            if top
            else None,
            "output_image": self.output_image,
            "summary": self.summary or {},
        }
