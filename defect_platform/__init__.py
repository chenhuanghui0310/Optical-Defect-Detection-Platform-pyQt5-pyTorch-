# -*- coding: utf-8 -*-
"""
defect_platform — 光学干涉瑕疵检测平台核心模块
================================================
提供与 UI 无关的核心 API，方便后续迁移到网页版（Flask/FastAPI）。

主要子模块：
  - simulation   : 物理仿真（劈尖 / 牛顿环）
  - ai_service   : AI 推理服务（YOLO）
  - dataset      : 数据集生成与划分
  - report       : 评估报告生成

快速使用示例：
    from defect_platform import DefectSimulator, DefectAIService

    # 生成仿真图像
    sim = DefectSimulator(sim_type="wedge")
    ideal_img, defect_img, topo_img = sim.generate_images(
        defect_type="小凸起", bump_height=0.05, bump_count=3
    )

    # AI 检测
    ai = DefectAIService()
    result = ai.predict_image("path/to/defect.png")
    print(result.top_prediction.label_cn)  # "小凸起"
"""

from .simulation import DefectSimulator
from .ai_service import DefectAIService
from .types import SimulationResult, DetectionResult

__all__ = [
    "DefectSimulator",
    "DefectAIService",
    "SimulationResult",
    "DetectionResult",
]

__version__ = "1.0.0"
