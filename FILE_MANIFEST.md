# 瑕疵检测平台 — 文件清单

> 本文档列出平台运行所需的**必要文件**，以及可以安全删除的**冗余文件**。

---

## 一、必要文件（不可删除）

### 1. 平台入口
| 文件 | 说明 |
|------|------|
| `main_platform.py` | **主入口**，统一劈尖+牛顿环平台 |
| `main.py` | 备用单独入口（可选保留） |

### 2. 劈尖干涉实验模块（`劈尖瑕疵0329/`）
| 文件 | 说明 |
|------|------|
| `parameters.py` | 物理仿真核心（厚度计算、干涉图样） |
| `main_window.py` | 劈尖主窗口 UI |
| `control_panel.py` | 左侧参数控制面板 |
| `simulation_widgets.py` | 图像显示容器 |
| `drawing_canvas.py` | 手动绘制划痕画板 |
| `defect_inference.py` | AI 推理包装器 |
| `intensity_plot.py` | 光强曲线图 |
| `spectrum_plot.py` | 光谱图 |
| `__init__.py` | 模块初始化 |

### 3. 牛顿环干涉实验模块（`牛顿环瑕疵/`）
| 文件 | 说明 |
|------|------|
| `parameters.py` | 牛顿环物理仿真核心 |
| `main_window.py` | 牛顿环主窗口 UI |
| `control_panel.py` | 参数控制面板 |
| `simulation_widgets.py` | 图像显示容器 |
| `drawing_canvas.py` | 手动绘制画板 |
| `defect_inference.py` | AI 推理包装器 |
| `intensity_plot.py` | 光强曲线图 |
| `spectrum_plot.py` | 光谱图 |
| `__init__.py` | 模块初始化 |

### 4. AI 推理服务（`defect_ai_module/`）
| 文件 | 说明 |
|------|------|
| `service.py` | **核心**：YOLO 推理服务（UI 无关） |
| `types.py` | 数据类型定义 |
| `__init__.py` | 模块初始化 |

### 5. 模块化平台 API（`defect_platform/`）⭐ 新增
| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块初始化，导出主要 API |
| `simulation.py` | 仿真 API（UI 无关，可用于网页后端） |
| `ai_service.py` | AI 检测 API（UI 无关，可用于网页后端） |
| `types.py` | 平台公共数据类型 |

### 6. 深度学习模块（`dl_main_model_yolo/`）
| 文件 | 说明 |
|------|------|
| `split_final_dataset.py` | 数据集划分（8:1:1）+ 自动标注 |
| `train_final_split.py` | YOLO 训练脚本 |
| `validate_main_model.py` | 模型验证脚本 |
| `predict_main_model.py` | 批量预测脚本 |
| `generate_report.py` | **评估报告生成**（含指标说明）⭐ 新增 |
| `runs/train/yolo11s_final_split2/weights/best.pt` | **训练好的模型权重**（最重要） |
| `yolo11s.pt` | YOLO11s 预训练基础权重 |

### 7. 数据集（`dataset/final_split_yolo/`）
| 文件/目录 | 说明 |
|-----------|------|
| `dataset.yaml` | 数据集配置文件 |
| `images/train/` | 训练集图像（约 800 张） |
| `images/val/` | 验证集图像（约 100 张） |
| `images/test/` | 测试集图像（约 100 张） |
| `labels/train/` | 训练集 YOLO 标注 |
| `labels/val/` | 验证集 YOLO 标注 |
| `labels/test/` | 测试集 YOLO 标注 |

### 8. 数据集生成
| 文件 | 说明 |
|------|------|
| `auto_dataset_generator.py` | 自动批量生成干涉图数据集 |

### 9. 依赖环境
| 文件/目录 | 说明 |
|-----------|------|
| `.venv/` 或 `.venv_report/` | Python 虚拟环境（保留其中一个即可） |
| `dl_main_model_yolo/requirements.txt` | Python 依赖列表 |

---

## 二、可以删除的冗余文件

> ⚠️ 删除前请确认不再需要这些文件。

### 可删除的脚本
| 文件 | 原因 |
|------|------|
| `quick_start_generation.py` | 功能已被 `auto_dataset_generator.py` 覆盖 |
| `dataset_config.py` | 配置已内嵌到生成脚本中 |
| `patch_hair.py` | 临时补丁脚本，功能已集成到 parameters.py |
| `simulation_main.py` | 旧版仿真入口，已被 main_platform.py 替代 |
| `virtual_main.py` | 旧版虚拟实验入口 |
| `opticaldefectt_main.py` | 旧版入口 |
| `dl_main_model_yolo/train_main_model.py` | 旧版训练脚本，已被 train_final_split.py 替代 |
| `dl_main_model_yolo/train_smallscale_split.py` | 小规模测试脚本 |

### 可删除的目录
| 目录 | 原因 |
|------|------|
| `build/` | PyInstaller 构建缓存，可重新生成 |
| `dist/` | PyInstaller 打包输出，可重新生成 |
| `发布包_轻量汇报版/` | 旧版发布包 |
| `老师演示轻量包_v2/` | 旧版演示包 |
| `DeepSeekAi/` | 旧版 AI 模块，已被 defect_ai_module 替代 |
| `SimulationExperiment/` | 旧版仿真实验模块 |
| `VirtualExperiment/` | 旧版虚拟实验模块 |
| `OpticalDefectExperiment/` | 旧版光学实验模块（如果劈尖/牛顿环模块已独立） |
| `dataset/images/Auto_Generated/` | 生成的原始图像，已划分到 final_split_yolo |
| `dataset/images/Final_Dataset/` | 同上，已划分完成可删除原始集 |
| `.venv_report/` | 如果 `.venv/` 已包含所有依赖 |
| `__pycache__/` | Python 字节码缓存，可自动重新生成 |
| `dl_main_model_yolo/runs/train/` 中除 `yolo11s_final_split2` 外的目录 | 旧版训练结果 |

### 可删除的文档（如不需要）
| 文件 | 原因 |
|------|------|
| `DATASET_GENERATION_GUIDE.md` | 已有 README.md 覆盖 |
| `DEEP_LEARNING_MODULE_INTEGRATION.md` | 集成说明，开发完成后可删 |
| `新增深度学习后项目整理文档.md` | 开发记录 |
| `QUICK_REFERENCE.md` | 快速参考，可选保留 |
| `dl_main_model_yolo/NEW_OLD_COMPARISON_smallscale_v2.md` | 对比记录 |
| `dl_main_model_yolo/RESULT_INTERPRETATION_TEMPLATE.md` | 已被 generate_report.py 替代 |

---

## 三、网页版迁移所需文件清单

如果未来要将瑕疵平台迁移到网页版（Flask/FastAPI），只需以下文件：

```
defect_platform/          ← 核心 API 模块（新增）
  __init__.py
  simulation.py
  ai_service.py
  types.py

defect_ai_module/         ← AI 推理服务
  service.py
  types.py
  __init__.py

劈尖瑕疵0329/
  parameters.py           ← 物理仿真核心

牛顿环瑕疵/
  parameters.py           ← 物理仿真核心

dl_main_model_yolo/runs/train/yolo11s_final_split2/weights/best.pt
                          ← 模型权重
```

网页后端示例（Flask）：
```python
from defect_platform import DefectSimulator, DefectAIService

sim = DefectSimulator("wedge")
ai = DefectAIService()

@app.route("/simulate")
def simulate():
    ideal_b, defect_b, topo_b = sim.generate_images("小凸起", bump_height=0.05)
    return send_file(io.BytesIO(defect_b), mimetype="image/png")

@app.route("/detect", methods=["POST"])
def detect():
    img_path = save_upload(request.files["image"])
    result = ai.predict_image(img_path)
    return jsonify(result.to_dict() if result else {"error": "未检测到目标"})
```

---

*最后更新：由 Kiro 自动生成*
