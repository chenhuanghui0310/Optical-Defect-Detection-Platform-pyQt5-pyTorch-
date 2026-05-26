# 2026-05-22 修改整理

本文档整理了本轮针对劈尖实验、牛顿环实验、数据集构建、YOLO 训练与推理稳定性做出的代码修改，以及过程中出现的回归问题与修复情况。

## 1. 修改目标

本轮修改主要围绕以下问题展开：

1. 劈尖小凸起瑕疵容易被误识别成头发丝遮挡。
2. 加入真实图片后，瑕疵预测不稳定，仿真图精度受到干扰。
3. 牛顿环凸起瑕疵有时检测不到，提示“未检测到明显目标”。
4. 当参数变化导致图像肉眼难分辨时，希望优先保证瑕疵类别判断正确。

## 2. 已修改文件

本轮修改涉及以下文件：

- [牛顿环瑕疵/parameters.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/parameters.py)
- [劈尖瑕疵0329/parameters.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/parameters.py)
- [牛顿环瑕疵/control_panel.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/control_panel.py)
- [劈尖瑕疵0329/control_panel.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/control_panel.py)
- [牛顿环瑕疵/main_window.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/main_window.py)
- [劈尖瑕疵0329/main_window.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/main_window.py)
- [dl_main_model_yolo/split_final_dataset.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/split_final_dataset.py)
- [dl_main_model_yolo/train_final_split.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/train_final_split.py)
- [defect_ai_module/service.py](D:/2026物理竞赛/Physics/defect_ai_module/service.py)

## 3. 仿真模块修改

### 3.1 为小凸起/小凹陷记录真实生成位置

修改文件：

- [牛顿环瑕疵/parameters.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/parameters.py)
- [劈尖瑕疵0329/parameters.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/parameters.py)

新增内容：

- 新增 `generated_bump_positions` 属性。
- 在 `add_bump_defect()` 中，把实际生成出的凸起/凹陷位置保存到 `generated_bump_positions`。
- 在 `export_parameters()` 中导出该字段。
- 在 `import_parameters()` 中支持读取该字段。

目的：

- 解决“随机分布”模式下，数据集构建阶段无法还原真实凸起位置的问题。
- 为后续精确生成 YOLO 局部框提供依据，避免退化成整图弱标签。

### 3.2 修复“小凸起”UI 名称与内部逻辑名不一致的问题

修改文件：

- [牛顿环瑕疵/control_panel.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/control_panel.py)
- [劈尖瑕疵0329/control_panel.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/control_panel.py)
- [牛顿环瑕疵/main_window.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/main_window.py)
- [劈尖瑕疵0329/main_window.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/main_window.py)

修复内容：

- UI 下拉框中仍使用 `"小凸起"`。
- 内部仿真逻辑统一使用 `"小突起"`。
- UI 下拉框中仍使用 `"凸起划痕"`。
- 内部逻辑统一使用 `"突起划痕"`。

具体做法：

- `update_parameters_from_ui()` 中加入映射：
  - `"小凸起"` -> `"小突起"`
  - `"凸起划痕"` -> `"突起划痕"`
- `defect_type_changed()` 中加入同样映射。
- `scratch_type_changed()` 中加入同样映射。
- `update_ui_from_parameters()` 中把内部名映射回 UI 名：
  - `"小突起"` -> `"小凸起"`
  - `"突起划痕"` -> `"凸起划痕"`

目的：

- 修复“小凸起图样和三维形貌不随参数变化”的回归问题。
- 修复“小凸起参数区被错误禁用”的回归问题。

### 3.3 修复小凸起参数区不可编辑的问题

修改文件：

- [牛顿环瑕疵/control_panel.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/control_panel.py)
- [劈尖瑕疵0329/control_panel.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/control_panel.py)

修复内容：

- `update_defect_controls_visibility()` 原先只识别 `"小凸起"`，而内部状态已经被改为 `"小突起"`。
- 现已兼容两种名称：
  - `"小突起"`
  - `"小凸起"`
  - `"小凹陷"`

目的：

- 让小凸起切换后，数量、高度/深度、横向半径、位置模式等参数重新可编辑。

## 4. 数据集构建修改

修改文件：

- [dl_main_model_yolo/split_final_dataset.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/split_final_dataset.py)

### 4.1 修复随机分布小凸起/小凹陷标注错误

原问题：

- 旧逻辑下，如果 `bump_position_mode == "随机分布"`，构建数据集时无法从 JSON 重建位置。
- 最终会退回成整图弱标签。
- 这会让模型把局部弱目标学坏，直接影响类别判断与定位。

修改后：

- 若 JSON 中存在 `generated_bump_positions`，则优先使用真实生成位置来生成解析标注。
- 不再因为“随机分布”就自动退回整图弱标签。

目的：

- 强化小凸起/小凹陷的定位和类别训练信号。
- 减少误判成 `hair` 或漏检。

### 4.2 新增真实图预处理能力

新增参数：

- `--preprocess-real`

新增逻辑：

- 对真实图先做中值滤波去噪。
- 再做 CLAHE 局部对比度增强。

目的：

- 降低真实图片中的随机噪点被模型学成“灰尘污染”的风险。
- 缓解真实图分布对仿真图主训练目标的扰动。

### 4.3 新增训练集真实图占比限制

新增参数：

- `--max-real-train-ratio`

作用：

- 限制训练集中 `real` 样本的占比。
- 若真实图过多，则自动把多余的真实图移到 `val/test`。

目的：

- 在保留真实图泛化价值的同时，优先保证仿真图上的检测精度。

## 5. 推理服务修改

修改文件：

- [defect_ai_module/service.py](D:/2026物理竞赛/Physics/defect_ai_module/service.py)

### 5.1 调整各类别置信度保留策略

新增逻辑：

- 不再统一使用 `0.35` 作为所有类别的过滤阈值。
- 改为按类别分别设置最小置信度：
  - `scratch`: 0.30
  - `bump`: 0.18
  - `dent`: 0.18
  - `dust`: 0.28
  - `hair`: 0.40

目的：

- 减少小凸起/小凹陷这种弱目标被早早过滤掉的情况。
- 对 `hair` 继续保持更谨慎的保留标准。

### 5.2 增加 hair 形状过滤

新增逻辑：

- 对 `hair` 检测框增加长宽比与面积判断。
- 对“不像头发丝”的框直接过滤。

目的：

- 降低小凸起被误判成头发丝遮挡的概率。

### 5.3 增加类别优先规则

新增逻辑：

- 当 `hair` 与 `bump/dent` 检测框重叠时，优先保留 `bump/dent`。

目的：

- 在类别冲突时优先保护“小凸起/小凹陷”的类别判断正确性。

### 5.4 增加弱目标兜底二次推理

新增逻辑：

- 第一轮没检出目标时，再进行一次更低阈值、更高分辨率的补扫。
- 第二轮只重点保留 `bump/dent`。

目的：

- 缓解牛顿环凸起瑕疵“未检测到明显目标”的问题。

### 5.5 提高默认推理分辨率

修改内容：

- 推理时 `imgsz` 提高到 `960`
- 弱目标 rescue pass 使用 `1280`

目的：

- 提升局部弱变形瑕疵的可检出性。

## 6. 训练脚本修改

修改文件：

- [dl_main_model_yolo/train_final_split.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/train_final_split.py)

### 6.1 训练参数调整

曾做过的调整：

- 把默认 `imgsz` 提高到 `960`
- 降低 mosaic 到 `0.2`
- 设置 `close_mosaic=15`

目的：

- 更有利于小弱目标的训练。

### 6.2 因显存不足再次调稳默认值

背景：

- 在 `RTX 4060 Laptop 8GB` 上，`YOLO11s + imgsz=960 + batch=16` 会触发 `CUDA out of memory`。

后续调整：

- 默认 `imgsz` 改为 `768`
- 默认 `batch` 改为 `8`
- 新增参数：
  - `--workers`
  - `--cache`
  - `--amp`

目的：

- 让训练脚本在 8GB 显存环境下更稳地启动。

## 7. 本轮出现的回归问题

### 7.1 回归一：小凸起图样和三维形貌不变化

现象：

- 选择小凸起后，调整高度/深度、横向半径等参数，图样与三维形貌几乎不变化。

原因：

- UI 使用 `"小凸起"`，内部逻辑使用 `"小突起"`。
- 某些更新链路中直接把 `"小凸起"` 传给了内部仿真分支，导致仿真分支未命中。

处理：

- 已修复，统一加入名称映射。

### 7.2 回归二：切换小凸起后参数区被禁用

现象：

- 小凸起模式下，瑕疵类型设置区的参数无法编辑。

原因：

- 控件启用逻辑只识别 `"小凸起"`，而参数对象中已经保存为 `"小突起"`。

处理：

- 已修复，控件逻辑现兼容 `"小突起"` 和 `"小凸起"`。

## 8. 当前建议的训练命令

如果已经重建了数据集，建议优先使用以下训练命令：

```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\train_final_split.py --model yolo11s.pt --epochs 120 --imgsz 768 --batch 8 --workers 4 --cache disk --device 0 --name yolo11s_final_split9
```

如果仍然显存不足，可以进一步降低为：

```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\train_final_split.py --model yolo11s.pt --epochs 120 --imgsz 640 --batch 8 --workers 2 --cache disk --device 0 --name yolo11s_final_split9
```

## 9. 当前建议的数据集重建命令

```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\split_final_dataset.py --clean --preprocess-real --max-real-train-ratio 0.15
```

含义：

- `--preprocess-real`：对真实图做轻量去噪和局部对比度归一。
- `--max-real-train-ratio 0.15`：限制训练集中真实图比例为 15% 左右，优先保障仿真图精度。

## 10. 后续可继续做的优化

如果后续继续迭代，建议优先考虑以下方向：

1. 把 `小凸起/小突起`、`凸起划痕/突起划痕` 的 UI 名称与内部枚举彻底统一成一套常量，避免再出现名称映射回归。
2. 把 `hair` 从“全图弱标签”升级为更贴近真实路径的局部框标注。
3. 进一步清理 `split_final_dataset.py` 中的重复标签来源，减少训练日志里的 `duplicate labels removed`。
4. 增加一套“小弱目标优先”的验证集专项评测样本，用于专门检查凸起/凹陷类别准确率。

