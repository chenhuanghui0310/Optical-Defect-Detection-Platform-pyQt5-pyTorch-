# -*- coding: utf-8 -*-
"""
defect_platform.types — 平台公共数据类型定义
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimulationResult:
    """仿真结果数据类"""
    sim_type: str                    # "wedge" 或 "newton"
    defect_type: str                 # 瑕疵类型（中文）
    ideal_pattern: np.ndarray        # 理想干涉图样（2D numpy数组）
    defect_pattern: np.ndarray       # 有瑕疵干涉图样（2D numpy数组）
    ideal_thickness: np.ndarray      # 理想厚度分布（米）
    defect_thickness: np.ndarray     # 有瑕疵厚度分布（米）
    params_dict: dict = field(default_factory=dict)  # 仿真参数字典

    @property
    def thickness_diff_um(self) -> np.ndarray:
        """厚度差（微米）"""
        return (self.defect_thickness - self.ideal_thickness) * 1e6


@dataclass
class DetectionResult:
    """AI 检测结果数据类"""
    label: str           # 英文标签（scratch/bump/dent/dust）
    label_cn: str        # 中文标签
    confidence: float    # 置信度 [0, 1]
    cls_id: int          # 类别 ID
    bbox_xyxy: list[float] | None = None  # 检测框坐标 [x1, y1, x2, y2]
    output_image_path: str = ""           # 带检测框的结果图路径

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_cn": self.label_cn,
            "confidence": self.confidence,
            "cls_id": self.cls_id,
            "bbox_xyxy": self.bbox_xyxy,
            "output_image_path": self.output_image_path,
        }
