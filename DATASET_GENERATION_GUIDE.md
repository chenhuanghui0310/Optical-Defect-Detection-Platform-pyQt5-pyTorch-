# 🚀 光学瑕疵检测 - 自动化数据生成完整指南

## 📋 目录
1. [快速开始](#快速开始)
2. [详细说明](#详细说明)
3. [参赛方案建议](#参赛方案建议)
4. [常见问题](#常见问题)

---

## 快速开始

### 方式1：一键生成（推荐新手）

```bash
# 进入项目目录
cd d:\2026物理竞赛\Physics

# 运行快速生成脚本（生成300张图）
python quick_start_generation.py
```

**需要3-5分钟**，会在 `dataset/images/Auto_Generated/` 生成包含：
- 300张基础数据集
- 每种缺陷类型 ~75 张
- 理想图 + 缺陷图 + 3D形貌 三种输出

### 方式2：自定义生成（高级用户）

```bash
# 生成500张图到指定目录
python auto_dataset_generator.py --count 500 --output dataset/custom_dataset

# 或简写
python auto_dataset_generator.py -c 500 -o dataset/custom_dataset
```

---

## 详细说明

### 📁 生成的文件结构

```
dataset/images/Auto_Generated/
├── 0001_表面划痕_理想干涉图.png          # 理想干涉条纹
├── 0001_表面划痕_有瑕疵干涉图.png        # 有缺陷的干涉条纹
├── 0001_表面划痕_三维缺陷形貌.png        # 3D形貌可视化
├── 0001_表面划痕_参数.json              # 参数信息
├── 0002_小突起_理想干涉图.png
├── 0002_小突起_有瑕疵干涉图.png
├── 0002_小突起_三维缺陷形貌.png
├── 0002_小突起_参数.json
└── ... (以此类推)
```

### 📊 缺陷类型分布

每次生成会均匀分布四种缺陷类型：

| 缺陷类型 | 样本比例 | 特点 |
|---------|--------|------|
| 🔲 表面划痕 | 25% | 凹陷或突起的线性划痕 |
| 📍 小突起 | 25% | 表面微小的隆起 |
| 🕳️  小凹陷 | 25% | 表面微小的凹陷 |
| 💧 灰尘颗粒 | 25% | 附着的颗粒性缺陷 |

### 🎛️ 可配置参数

#### 方式1：命令行参数

```bash
# 参数说明
--count N          # 总样本数
--output PATH      # 输出目录
--continue         # 接续之前的生成（可选）
```

#### 方式2：编程配置

```python
from auto_dataset_generator import DatasetGenerator

# 自定义生成
generator = DatasetGenerator(
    output_dir="dataset/my_dataset",
    num_samples=1000  # 生成1000张
)

stats = generator.generate_dataset()
print(f"生成成功: {stats['success']} 张")
```

#### 方式3：配置文件（高级）

创建 `config.json`：

```json
{
  "dataset": {
    "name": "MyDataset",
    "description": "自定义数据集"
  },
  "generation": {
    "total_samples": 1000,
    "output_dir": "dataset/my_dataset"
  },
  "defects": {
    "表面划痕": {
      "enabled": true,
      "samples_ratio": 0.3,
      "scratch_length_range": [50, 250]
    },
    "小突起": {
      "enabled": true,
      "samples_ratio": 0.25
    }
  }
}
```

然后使用：

```python
from dataset_config import DatasetConfig
config = DatasetConfig("config.json")
```

---

## 参赛方案建议

### 🏆 推荐方案：Pix2Pix 图像到图像转换

**为什么选择Pix2Pix？**
- ✓ 你已有配对训练数据（理想图↔缺陷图）
- ✓ 创新性强（物理仿真 + 深度学习结合）
- ✓ 视觉效果优秀（能生成连贯的像素级输出）
- ✓ 易于演示（输入缺陷图→输出恢复后的理想图）

### 📈 数据集规模建议

| 阶段 | 样本数 | 用途 |
|------|-------|------|
| 原始数据 | 40 | 参考基准 |
| 第一轮 | 200-300 | 算法验证 |
| **完整训练** | **1000-2000** | **比赛提交** |
| 增强版 | 5000+ | 实际应用 |

### 🛠️ 完整工作流

**第1阶段：数据准备**

```bash
# 生成基础数据集
python quick_start_generation.py

# 或生成更大规模
python auto_dataset_generator.py -c 1000 -o dataset/v1_1000samples
python auto_dataset_generator.py -c 1000 -o dataset/v2_1000samples
# 合并累积到2000张

# 数据检查（预留功能）
python prepare_dataset.py
```

**第2阶段：数据预处理**

```python
import torch
from torch.utils.data import DataLoader

# 加载数据
train_dataset = OpticalDefectDataset("dataset/v1_1000samples")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 检查数据形状和范围
for ideal, defect in train_loader:
    print(f"Ideal shape: {ideal.shape}")  # 应该是 [32, 1, H, W]
    print(f"Defect shape: {defect.shape}")
    break
```

**第3阶段：模型训练**

```python
# 训练Pix2Pix模型
# 输入：有瑕疵干涉图 (1, 256, 256)
# 输出：理想干涉图 (1, 256, 256)
```

---

## 常见问题

### Q1：生成速度太慢？

**A：** 正常速度为 2-5 秒/张。如需加快：

1. **减少分辨率**
   ```python
   # 修改auto_dataset_generator.py中的DPI
   fig, ax = plt.subplots(figsize=(8, 5), dpi=50)  # 改为50
   ```

2. **减少生成图片类型**
   ```python
   # 只生成其中两种缺陷类型
   defect_types = ["表面划痕", "小突起"]
   ```

3. **使用并行化** (需额外编码)

### Q2：生成的图片质量不好？

**A：** 检查以下项：

- 确保使用最新的 OpticalDefectExperiment 代码
- 参数范围是否合理（见config.json说明）
- DPI设置是否过低

### Q3：能否只生成某一种缺陷？

**A：** 两种方式：

方式1（快速）：
```bash
# 修改auto_dataset_generator.py第40行
self.defect_types = ["表面划痕"]  # 只保留这一种
```

方式2（配置）：
编辑 config.json，将其他缺陷 `enabled` 设置为 `false`

### Q4：生成后的图片可以直接用于训练吗？

**A：** 建议预处理步骤：

1. **规范化大小** → 统一 (256, 256) 或 (512, 512)
2. **标准化像素值** → [0, 1] 或 [-1, 1]
3. **数据划分** → 训练/验证/测试 (70/15/15)
4. **数据增强** → 旋转、翻转、添加噪声

### Q5：如何查看生成参数？

**A：** 查看 JSON 文件：

```json
// 0001_表面划痕_参数.json
{
  "defect_type": "表面划痕",
  "scratch_length": 120,
  "scratch_angle": 45.3,
  "wavelength": 650,
  "oblique_angle": 2.5,
  ...
}
```

---

## 📚 相关文件

- `auto_dataset_generator.py` - 核心生成脚本
- `dataset_config.py` - 配置管理模块
- `quick_start_generation.py` - 快速开始脚本
- `defect.yaml` - 平台配置文件
- `OpticalDefectExperiment/` - 仿真核心库

---

## 🎯 下一步

1. ✅ 生成 1000+ 样本数据集
2. ⏳ 准备数据预处理脚本
3. ⏳ 实现 Pix2Pix 训练框架
4. ⏳ 模型评估和对比

---

## 💡 优化建议

| 优化方向 | 预期收益 | 难度 |
|---------|--------|------|
| 增加数据量到 2000-3000 | +5-10% | ⭐ |
| 数据增强 (旋转/噪声) | +3-7% | ⭐⭐ |
| 使用 GPU 加速生成 | 快 10-20倍 | ⭐⭐ |
| Pix2Pix 模型微调 | +10-20% | ⭐⭐⭐ |
| 多任务学习（分类+恢复） | +5-15% | ⭐⭐⭐⭐ |

---

**需要帮助？** 查看代码注释或运行 `python auto_dataset_generator.py --help`

