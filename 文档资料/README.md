# 项目文档资料

本目录集中存放项目报告、项目说明、修改记录和文档检查输出。

- `.docx`：竞赛报告、模型数据集与训练评估报告
- `.md`：项目上下文、开发记录、快速参考和文件清单
- `.txt`：文档检查输出

程序运行所需的实验资料仍保留在 `docts/` 和 `resources/docts/`，这些路径由打包配置作为运行时资源使用。

## 项目结构

- `main.py`：物理实验综合平台启动器
- `main_platform.py`：光学瑕疵检测综合平台入口
- `劈尖瑕疵0329/`、`牛顿环瑕疵/`：两类干涉实验模块
- `defect_ai_module/`、`defect_platform/`：AI 推理和平台 API
- `dl_main_model_yolo/`：数据划分、训练、验证和评估脚本
- `dataset/`：数据集
- `runs/`、`dl_main_model_yolo/runs/`：训练和评估产物
- `resources/`、`docts/`：程序运行资源和实验资料
- `build/`、`dist/`、`dist_v3_cpu/`：构建及发布产物
