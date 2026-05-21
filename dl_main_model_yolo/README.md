# YOLO 瑕疵检测模型工作区

本目录是光学干涉图瑕疵检测的深度学习训练与推理工作区。

## 当前状态

- **最新模型**: `runs/train/yolo11s_final_split8/weights/best.pt`
- **基础架构**: YOLOv11s（Ultralytics 8.4.23）
- **训练数据**: `dataset/final_split_yolo/`（含仿真 + 真实数据，共约 3700 张）
- **最终 mAP50**: 0.820 | **mAP50-95**: 0.776
- **训练轮次**: 120 epochs，640×640，batch=16

## 检测类别

| ID | 类别 | 英文名 | 标注方式 |
|----|------|--------|----------|
| 0 | 表面划痕 | scratch | 差分图自动标注 |
| 1 | 小凸起 | bump | 解析标注（从 JSON 参数计算） |
| 2 | 小凹陷 | dent | 解析标注（从 JSON 参数计算） |
| 3 | 灰尘污染 | dust | 差分图自动标注（阈值=3） |
| 4 | 头发丝遮挡 | hair | 全图弱标签 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `split_final_dataset.py` | 数据集划分（8:1:1）+ 自动标注生成 |
| `train_final_split.py` | YOLO 训练脚本 |
| `validate_main_model.py` | 模型验证脚本 |
| `predict_main_model.py` | 批量预测脚本 |
| `generate_report.py` | 评估报告生成（含指标说明） |
| `merge_real_data.py` | 真实图片合并到 Final_Dataset |
| `yolo11s.pt` | YOLO11s 预训练基础权重（训练起点） |
| `requirements.txt` | Python 依赖列表 |

## 训练结果目录

```
runs/train/
├── yolo11s_final_split8/    ← ✅ 最新（差分法灰尘标注）
│   ├── weights/
│   │   ├── best.pt          ← 最终模型权重
│   │   └── last.pt          ← 最后 epoch 权重
│   ├── results.csv          ← 训练指标日志
│   ├── results.png          ← 训练曲线图
│   ├── confusion_matrix.png ← 混淆矩阵
│   ├── BoxPR_curve.png      ← PR 曲线
│   └── val_batch*_pred.jpg  ← 验证集预测可视化
├── yolo11s_final_split7/    ← 旧版（全图弱标签灰尘，可删）
├── yolo11s_final_split6/    ← 旧版（可删）
└── yolo11s_final_split/     ← 最早版本（可删）
```

## 快速使用

### 1. 重建数据集（修改标注逻辑后必须执行）

```bash
python dl_main_model_yolo/split_final_dataset.py --clean
```

输出到 `dataset/final_split_yolo/`，自动生成 `dataset.yaml`。

### 2. 训练模型

```bash
python dl_main_model_yolo/train_final_split.py --model yolo11s.pt --epochs 120 --batch 16 --device 0
```

### 3. 验证模型

```bash
python dl_main_model_yolo/validate_main_model.py --weights dl_main_model_yolo/runs/train/yolo11s_final_split8/weights/best.pt --device 0
```

### 4. 批量预测

```bash
python dl_main_model_yolo/predict_main_model.py --weights dl_main_model_yolo/runs/train/yolo11s_final_split8/weights/best.pt --source dataset/final_split_yolo/images/test --device 0
```

## 标注逻辑说明

`split_final_dataset.py` 中的标注策略：

1. **表面划痕 (cls=0)** + **灰尘污染 (cls=3)**：差分图标注
   - 用有瑕疵图减去理想图，二值化后提取轮廓生成精确局部框
   - 灰尘阈值降低到 3（划痕为 8），因为灰尘亮度差更弱

2. **小凸起 (cls=1)** + **小凹陷 (cls=2)**：解析标注
   - 直接从 JSON 参数计算每个 bump 的物理坐标，转换为 YOLO 格式
   - 随机分布模式退回差分图标注

3. **头发丝遮挡 (cls=4)**：全图弱标签
   - 路径随机无法解析，使用 `0.5 0.5 1.0 1.0`

4. **真实数据 (sim_type=real)**：使用预生成标注文件或弱标签

## 版本历史

| 版本 | 说明 |
|------|------|
| split8 | ✅ 灰尘改用差分图标注，删除350个旧版样本后重建 |
| split7 | 灰尘仍为全图弱标签（数据集未重建） |
| split6 | 加入真实图片数据 |
| split | 最初版本，纯仿真数据 |

## 调参建议

1. `--epochs` 和 `--batch`（当前 120/16 已较优）
2. `--imgsz`（当前 640，可尝试 800）
3. 模型规模：`yolo11n.pt → yolo11s.pt → yolo11m.pt`
4. 推理置信度阈值 `--conf`

---

*最后更新：2026-05-21*
