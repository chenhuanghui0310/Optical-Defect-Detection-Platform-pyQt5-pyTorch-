# 劈尖瑕疵0329 - AI接入说明

当前版本已接入以下能力：
- 保存图片后自动检测
- 检测任意外部图片
- 选择并切换模型权重

## 关键文件
- main_window.py：UI与检测调用点
- defect_inference.py：平台适配层
- ../defect_ai_module/：独立推理核心（建议长期稳定）

## 使用方式
1. 启动 main.py
2. 菜单栏点击“AI检测”
   - 选择模型权重...
   - 检测外部图片...
3. 点击“保存图片”后会自动检测“有瑕疵干涉图”并显示结果

## 默认模型路径
- ../dl_main_model_yolo/runs/train/yolo11s_final_split/weights/best.pt
- 也可以在菜单中手动切换 .pt 文件

## 后续开发建议
- UI改动尽量只在 main_window.py
- 推理逻辑统一走 defect_ai_module/service.py
- Web迁移时可直接复用 defect_ai_module，不依赖 PyQt
