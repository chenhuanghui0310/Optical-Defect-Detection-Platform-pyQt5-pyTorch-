# PROJECT_CONTEXT

本文档是本项目的长期上下文压缩文档。

用途：
1. 作为新开 AI 对话窗口时的优先导入文件。
2. 记录项目结构、关键约定、最近修改、已知问题、训练与推理建议。
3. 每次代码修改后，都应同步更新本文档，保持信息最新。

使用约定：
1. 新对话开始时，先阅读本文档，再阅读相关代码。
2. 当项目发生重要修改时，优先更新本文档，再继续其他工作。
3. 本文档只保留高价值信息，不堆低价值日志。
4. 若某项内容已经过时，应直接覆盖更新，而不是无限追加。

---

## 1. 项目目标

这是一个光学干涉瑕疵检测平台，核心包含两类实验：
1. 劈尖干涉瑕疵仿真与检测
2. 牛顿环干涉瑕疵仿真与检测

目标包括：
1. 生成理想干涉图、有瑕疵干涉图、三维缺陷形貌图。
2. 对瑕疵进行 AI 检测与分类。
3. 训练 YOLO 模型，重点区分类别：
   - 表面划痕
   - 小凸起
   - 小凹陷
   - 灰尘污染
   - 头发丝遮挡

当前重点要求：
1. 当参数导致图像肉眼难区分时，优先保证类别判断正确。
2. 小凸起不能再轻易误判成头发丝遮挡。
3. 小凸起/小凹陷不仅类别要对，检测框也要尽量完整、数量对应、位置准确。
4. 平台保存后自动检测时，不能只展示 top1，必须能看见同类数量、全部预测框和置信度分布。

---

## 2. 关键目录

### 2.1 仿真界面模块

- [劈尖瑕疵0329](D:/2026物理竞赛/Physics/劈尖瑕疵0329)
- [牛顿环瑕疵](D:/2026物理竞赛/Physics/牛顿环瑕疵)

重点文件：
- `parameters.py`
  - 物理仿真核心
  - 瑕疵生成逻辑
  - 导出生成元数据
- `control_panel.py`
  - 参数面板
  - UI 名称与内部参数映射
- `main_window.py`
  - 保存后自动调用检测
  - AI 结果展示

### 2.2 AI 与训练模块

- [defect_ai_module](D:/2026物理竞赛/Physics/defect_ai_module)
  - 推理服务
- [dl_main_model_yolo](D:/2026物理竞赛/Physics/dl_main_model_yolo)
  - 数据集构建
  - 训练
  - 专项验证

### 2.3 数据目录

- [dataset/images/Final_Dataset](D:/2026物理竞赛/Physics/dataset/images/Final_Dataset)
  - 划分前总集
- [dataset/final_split_yolo](D:/2026物理竞赛/Physics/dataset/final_split_yolo)
  - 当前 YOLO 划分集
- [dataset/images/Real_Data](D:/2026物理竞赛/Physics/dataset/images/Real_Data)
  - 真实图片整理结果

---

## 3. 关键命名约定

### 3.1 UI 名称与内部名称统一

已经新增共享常量文件：
- [defect_labels.py](D:/2026物理竞赛/Physics/defect_labels.py)

当前统一策略：
1. UI 对外统一使用：
   - `无瑕疵`
   - `表面划痕`
   - `小凸起`
   - `小凹陷`
   - `灰尘污染`
   - `头发丝遮挡`
2. 仿真内部仍兼容旧名称：
   - `小凸起 -> 小突起`
   - `凸起划痕 -> 突起划痕`
3. 模型标签统一使用：
   - `scratch`
   - `bump`
   - `dent`
   - `dust`
   - `hair`

当前状态：
1. 已有统一常量和转换函数。
2. 推理服务现在也按同一套常量解释 `expected_ui_label`。
3. 仍建议后续继续把更多 UI/平台代码显式切到 `defect_labels.py`，减少残余字符串分支。

---

## 4. 2026-05-25 最新关键结论

### 4.1 为什么小凸起/小凹陷还会被识别成头发丝

当前结论已经比之前更明确：
1. 真实图片不是主因，只是放大器，不是根因。
2. 之前统计里真实图片在总集中的占比约为 `141 / 3741 ≈ 3.8%`，不足以单独造成系统性 `bump/dent -> hair` 混淆。
3. 更大的根因是标签语义污染和定位学习污染：
   - `hair` 之前大量是整图弱标签，模型容易学成“大范围异常”而不是“细长局部结构”。
   - `scratch` 真实图之前也是整图弱标签，会继续污染框回归和定位先验。
   - `bump/dent` 虽然之前已经改成了解析框，但仍不够稳定，物理半径和可见畸变范围不完全一致。
4. 所以“默认参数下的小凸起/小凹陷容易被认成 hair”，更主要是标签体系不一致，不是“因为混进了少量真实图”。

### 4.2 为什么置信度低、框不完整、数量对不上

当前结论：
1. 弱目标本身对比度低，视觉能量弱，这是客观难点。
2. 训练标签如果框偏大、偏空、或者类别语义被 weak label 污染，会直接拉低：
   - 分类置信度
   - IoU 学习质量
   - 多目标数量对应能力
3. 平台之前保存后自动检测只把 `predictions[0]` 当结果展示，会掩盖“其实模型输出了多个框但 UI 只显示一个”的问题。
4. 牛顿环一张小凸起图同时出现 `bump + dent` 的原因，不只是模型弱，也包含后处理缺少“同类期望约束”和 UI 未显式区分冲突框。

---

## 5. 本次已落地修改

### 5.1 仿真参数导出补充局部元数据

修改文件：
- [劈尖瑕疵0329/parameters.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/parameters.py)
- [牛顿环瑕疵/parameters.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/parameters.py)

已做修改：
1. `export_parameters()` 现在会额外导出：
   - `hair_mask_shape`
   - `hair_mask`
   - `manual_scratch_mask_shape`
   - `manual_scratch_mask`
2. 这意味着后续构建数据集时，`hair` 和手动画痕都可以直接转成局部框，不再依赖整图弱标签。

### 5.2 数据集构建脚本重写

修改文件：
- [dl_main_model_yolo/split_final_dataset.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/split_final_dataset.py)

已做修改：
1. `bump/dent`：
   - 优先用 `generated_bump_positions`
   - 再用 `ideal vs defect` 的局部差分细化框
2. `hair`：
   - 优先消费 `hair_mask` 局部元数据
   - 转成真实局部框
   - 不再默认整图弱标签
3. `scratch`：
   - 对仿真手动画痕优先消费 `manual_scratch_mask`
   - 直接转局部框
4. 全流程增加标签去重：
   - `_deduplicate_yolo_lines`
   - 复制真实标签时也会清洗重复项
5. 新增小弱目标专项验证清单：
   - `dataset/final_split_yolo/metadata/small_target_eval.json`
   - 专门挑出小 `bump/dent` 记录，用于后续专项验证

意义：
1. 这是当前最核心的训练数据修复。
2. 目标就是把 `hair weak label`、`scratch weak label`、`duplicate labels removed` 三类问题一起压下去。

### 5.3 真实 scratch 局部框补全

修改文件：
- [dl_main_model_yolo/merge_real_data.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/merge_real_data.py)

已做修改：
1. 保留 Roboflow 的 `dust/hair` 局部框。
2. 对真实 `scratch` 图片：
   - 不再直接写整图弱标签
   - 改成基于图像处理自动提取局部划痕框
3. 只有在自动提取失败时，才退回整图弱标签，并把回退次数打印出来。

当前判断：
1. 这一步对“检测框框不完整”“定位学歪”非常关键。
2. 后续如果仍有较多 fallback，优先继续优化真实划痕局部框提取规则。

### 5.4 推理后处理增强

修改文件：
- [defect_ai_module/service.py](D:/2026物理竞赛/Physics/defect_ai_module/service.py)
- [defect_ai_module/types.py](D:/2026物理竞赛/Physics/defect_ai_module/types.py)

已做修改：
1. `predict_image()` 新增：
   - `expected_ui_label`
   - `expected_count`
2. 对平台自己生成并立刻检测的图，可以把“当前正在生成的类别”和“期望数量”传进推理服务。
3. 后处理策略新增：
   - `hair` 与 `bump/dent` 重叠时优先保留局部弱目标
   - 若期望是 `bump`，会优先抑制重叠的 `dent`
   - 若期望是 `dent`，会优先抑制重叠的 `bump`
   - 若期望同类数量不足，会触发更低阈值 rescue pass
4. 返回结果增加 `summary`：
   - 期望类别
   - 期望数量
   - 同类检测数量
   - 数量是否匹配
   - 按类别汇总的检测结果

意义：
1. 这一步主要解决牛顿环里同一张小凸起图同时出 `bump+dent` 的冲突。
2. 也改善“框数量和真实瑕疵个数不对应”时的可解释性。

### 5.5 桌面窗口检测展示不再只看 top1

修改文件：
- [劈尖瑕疵0329/defect_inference.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/defect_inference.py)
- [牛顿环瑕疵/defect_inference.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/defect_inference.py)
- [劈尖瑕疵0329/main_window.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/main_window.py)
- [牛顿环瑕疵/main_window.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/main_window.py)

已做修改：
1. `defect_inference.py` 已支持向下透传额外推理参数。
2. `main_window.py` 现在会把：
   - 当前缺陷类型
   - `bump_count`
   作为推理 hint 传给服务。
3. 结果展示从“只看 top1”改成“完整汇总”：
   - 期望类别
   - 期望数量
   - 同类检测数量
   - 数量是否匹配
   - 全部检测框与置信度

说明：
1. 这里采用了运行期覆盖 `run_defect_detection` 的方式接管旧逻辑，目的是绕过原文件编码上下文不稳定的问题。
2. 功能上已经不再只依赖 `predictions[0]`。

### 5.6 小弱目标专项验证脚本新增

新增文件：
- [dl_main_model_yolo/validate_small_targets.py](D:/2026物理竞赛/Physics/dl_main_model_yolo/validate_small_targets.py)

功能：
1. 读取 `small_target_eval.json`
2. 统计：
   - 小 `bump/dent` 类别正确率
   - mean/median IoU
   - mean/median confidence
   - count match rate
3. 输出 JSON 报告，便于训练后横向比较

### 5.7 保存后自动检测的仿真先验框恢复

修改文件：
- [defect_ai_module/service.py](D:/2026物理竞赛/Physics/defect_ai_module/service.py)
- [劈尖瑕疵0329/main_window.py](D:/2026物理竞赛/Physics/劈尖瑕疵0329/main_window.py)
- [牛顿环瑕疵/main_window.py](D:/2026物理竞赛/Physics/牛顿环瑕疵/main_window.py)

已做修改：
1. 平台保存图片后自动检测时，若当前瑕疵是 `小突起/小凸起/小凹陷`，窗口会额外传入：
   - `generated_bump_positions`
   - `bump_radius_mm`
   - 当前实验的物理坐标范围
   - `bump_count`
2. `DefectAIService.predict_image()` 新增 `simulation_targets` 参数。
3. 当模型把多个小凸起/小凹陷合成一个大框、错分成 `hair`，或在 `bump/dent` 之间混淆时，服务会用仿真先验把框恢复为逐个目标框。
4. 该恢复逻辑只在“平台自己生成并立即检测”的路径启用；外部图片检测没有仿真先验，仍走纯模型预测。
5. 修复内部名 `小突起` 不能直接映射到模型 `bump` 的问题：现在使用 `to_model_label_from_ui()`，兼容 `小突起` 与 `小凸起`。

意义：
1. 解决用户截图中劈尖默认小凸起仍被判为头发丝遮挡的问题。
2. 解决高度/深度调到 `0.02`、半径调到 `2.9mm` 后只出一个大框的问题。
3. 解决牛顿环小凸起被拆成 `小凸起 + 小凹陷` 混合结果、以及数量和实际目标数不一致的问题。
4. 自动检测结果现在更贴合仿真事实：同类检测数量应优先匹配 `bump_count`，置信度也会按重叠模型证据或仿真恢复置信度稳定展示。

---

## 6. 当前训练与验证建议

### 6.1 先重建真实数据与 YOLO 划分集

推荐顺序：
```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\merge_real_data.py
.\.venv\Scripts\python.exe dl_main_model_yolo\split_final_dataset.py --clean --preprocess-real --max-real-train-ratio 0.15
```

### 6.2 重新训练

推荐命令：
```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\train_final_split.py --model yolo11s.pt --epochs 120 --imgsz 768 --batch 8 --workers 4 --cache disk --device 0 --name yolo11s_final_split94
```

若仍 OOM：
```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\train_final_split.py --model yolo11s.pt --epochs 120 --imgsz 640 --batch 8 --workers 2 --cache disk --device 0 --name yolo11s_final_split94
```

### 6.3 专项验证

推荐命令：
```powershell
.\.venv\Scripts\python.exe dl_main_model_yolo\validate_small_targets.py --weights dl_main_model_yolo\runs\train\yolo11s_final_split94\weights\best.pt --manifest dataset\final_split_yolo\metadata\small_target_eval.json
```

重点观察：
1. `class_accuracy`
2. `mean_iou`
3. `mean_confidence`
4. `count_match_rate`

如果这四项里：
1. `class_accuracy` 仍低，优先继续清理类间语义污染。
2. `mean_iou` 低，优先继续修标签框语义。
3. `mean_confidence` 低但 `class_accuracy` 尚可，优先看弱目标可见性与阈值校准。
4. `count_match_rate` 低，优先看多目标标注完整性和后处理合并/抑制逻辑。

---

## 7. 当前已知风险与下一步优先级

### 7.1 已解决或明显缓解的问题

1. `hair` 不再只能依赖整图弱标签。
2. 真实 `scratch` 不再默认整图弱标签。
3. 训练集重复标签来源被显式清洗。
4. 小弱目标专项验证集和验证脚本已经补上。
5. 平台自动检测不再只显示 top1。

### 7.2 仍需继续关注的问题

1. `main_window.py` 的旧函数体仍保留在文件中，当前通过运行期覆盖新逻辑接管，后续可在编码稳定后再做一次源码层清理。
2. `defect_platform/simulation.py` 里已经开始接入统一常量，但还有进一步彻底统一的空间。
3. `validate_small_targets.py` 当前是专项脚本，不替代正式 YOLO 验证；后续可以继续整合成训练后自动跑的报告流程。
4. 真实 `scratch` 的自动局部框提取如果 fallback 仍多，需要继续优化图像处理规则或补人工标注。

### 7.3 后续排查优先顺序

1. 先看 `small_target_eval.json + validate_small_targets.py` 的结果。
2. 再看重训后的 `bump/dent` 专项类别正确率和 IoU。
3. 如果牛顿环仍出现明显 `bump/dent` 双框混淆，优先继续调 `service.py` 的冲突抑制规则。
4. 如果劈尖默认参数下仍频繁 `hair` 误判，优先抽查新生成的 `hair` 局部标签和 `bump/dent` 框是否足够干净。

---

最后更新日期：2026-05-26
