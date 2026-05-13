# 深度学习模块快速接入说明

本文用于你后续网页化和同事代码更新时，快速接入深度学习检测能力。

## 模块边界

### 1) 独立模块（建议长期稳定）
- `defect_ai_module/`
  - `service.py`: 纯推理服务（无 PyQt 依赖）
  - `types.py`: 结果数据结构
  - `__init__.py`: 对外导出

### 2) 桌面平台适配层（可变）
- `OpticalDefectExperiment/defect_inference.py`
  - 兼容旧接口，内部委托到 `DefectAIService`

### 3) 训练与权重（模型资产）
- `dl_main_model_yolo/`
- 默认权重路径：
  - `dl_main_model_yolo/runs/train/yolo11s_final_split/weights/best.pt`

## 同事更新平台代码后的最小接入步骤

1. 在新主窗口中初始化推理引擎
```python
from OpticalDefectExperiment.defect_inference import DefectInferenceEngine
self.inference_engine = DefectInferenceEngine()
```

2. 在保存图片/导入图片后调用检测
```python
result = self.inference_engine.predict_image(image_path)
```

3. 展示关键结果
- `result["top_prediction"]`
- `result["output_image"]`

## 网页端迁移建议

1. Web API 直接调用 `defect_ai_module.DefectAIService`
2. 上传图片后执行 `predict_image`
3. 返回 JSON：类别、置信度、可视化图路径

## 不容易受同事改动影响的做法

1. 同事只改 UI/绘图逻辑，不改 `defect_ai_module`
2. 平台适配代码只保留 2 个调用点：初始化 + 推理调用
3. 模型路径统一通过 `DEFECT_MODEL_PATH` 环境变量或设置页面配置
