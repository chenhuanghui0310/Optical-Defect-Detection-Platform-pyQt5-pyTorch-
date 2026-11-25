# 🚀 快速参考卡 - 数据自动生成

## 一张图理解全流程

```
您的需求
    ↓
你有: 40张 defect platform 输出
你想: 用有瑕疵干涉图训练深度学习
    ↓
我提供的解决方案
    ├─ auto_dataset_generator.py    [核心生成脚本]
    ├─ dataset_config.py            [配置管理]
    ├─ quick_start_generation.py    [一键启动]
    └─ DATASET_GENERATION_GUIDE.md  [完整文档]
    ↓
你得到
    ├─ 300+ 张理想干涉图
    ├─ 300+ 张有瑕疵干涉图  ← 用来训练深度学习
    ├─ 300+ 张三维形貌图
    └─ 参数json记录
    ↓
用于模型训练（推荐Pix2Pix）
```

---

## 5分钟快速开始

### 第1步：打开命令行 (5秒)
```bash
# Windows: Win+R 输入 cmd
cd d:\2026物理竞赛\Physics
```

### 第2步：激活虚拟环境 (1秒)
```bash
.venv\Scripts\activate
```

### 第3步：运行生成 (5-30分钟，取决于生成数量)
```bash
# 快速测试 (生成100张，~10分钟)
python auto_dataset_generator.py -c 100

# 或使用快速启动脚本 (生成300张)
python quick_start_generation.py
```

### 完成！✓
生成的图片位置: `dataset/images/Auto_Generated/`

---

## 文件位置参考

```
Physics/
│
├─ auto_dataset_generator.py           # 核心脚本 ⭐
│  └─ 类: DatasetGenerator
│     ├─ def __init__()               # 初始化
│     ├─ def generate_sample()         # 生成单个样本
│     ├─ def generate_dataset()        # 批量生成
│     └─ def help()                    # --help
│
├─ dataset_config.py                   # 配置管理
│  └─ 类: DatasetConfig
│     ├─ def load()                    # 加载配置
│     ├─ def save()                    # 保存配置
│     └─ def get/set()                 # 参数访问
│
├─ quick_start_generation.py           # 快速入门
│
├─ test_generation.py                  # 测试脚本
│
├─ DATASET_GENERATION_GUIDE.md         # 完整文档 📖
│
└─ dataset/images/
   ├─ Defect platform results/         # 你原始的 40 张
   └─ Auto_Generated/                  # 自动生成的新数据 (待生成)
```

---

## 命令行速查表

```bash
# 基础用法
python auto_dataset_generator.py

# 指定样本数
python auto_dataset_generator.py -c 500

# 指定输出目录
python auto_dataset_generator.py -o dataset/my_v1

# 两者结合
python auto_dataset_generator.py -c 1000 -o dataset/large_dataset

# 查看帮助
python auto_dataset_generator.py --help
```

---

## 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|-------|------|
| `-c, --count` | 生成样本总数 | 100 | `-c 500` |
| `-o, --output` | 输出目录 | `dataset/images/Auto_Generated` | `-o data/v1` |
| `--help` | 显示帮助 | - | `--help` |

---

## 输出文件格式

每个样本生成 4 个文件：

```
0001_表面划痕_理想干涉图.png          # 理想干涉条纹 (用作标签)
0001_表面划痕_有瑕疵干涉图.png        # 有缺陷的干涉 (用来训练)  ⭐
0001_表面划痕_三维缺陷形貌.png        # 3D形貌可视化 (辅用)
0001_表面划痕_参数.json              # 参数信息 (记录)
```

**关键：** 你需要的是 `_有瑕疵干涉图.png` 用于模型训练

---

## 生成进度估计

```
样本数    | 预计时间     | 文件数   | 磁盘大小
---------|-------------|--------|----------
100      | ~5分钟      | 400    | ~50 MB
300      | ~15分钟     | 1200   | ~150 MB
500      | ~25分钟     | 2000   | ~250 MB
1000     | ~50分钟     | 4000   | ~500 MB
```

---

## 故障排查

### 问题1：命令行找不到 python
**解决1：** 激活虚拟环境
```bash
.venv\Scripts\activate
python auto_dataset_generator.py
```

**解决2：** 使用完整路径
```bash
.venv\Scripts\python.exe auto_dataset_generator.py
```

### 问题2：找不到 OpticalDefectExperiment 模块
**原因：** 没有在正确目录运行
**解决：** 确保在 `Physics/` 目录下
```bash
cd d:\2026物理竞赛\Physics  # 重要！
python auto_dataset_generator.py
```

### 问题3：生成速度太慢
**原因：** 正常，每张需要2-5秒
**建议：**
- 减少生成数量 (用 `-c 100` 测试)
- 或者在后台运行，同时做其他事情

---

## 下一步建议

✅ **现在就做：**
```bash
python auto_dataset_generator.py -c 100  # 小规模测试
```

✅ **然后做：**
```bash
python auto_dataset_generator.py -c 1000 -o dataset/v1_1000  # 正式数据
```

✅ **接着做：** 编写数据预处理脚本（待提供）

✅ **最后做：** 训练 Pix2Pix 模型（待提供）

---

## 💡 Pro Tips

1. **并行运行多次**
   ```bash
   # 终端1: 生成500张到v1
   python auto_dataset_generator.py -c 500 -o dataset/v1

   # 终端2: 生成500张到v2 (同时)
   python auto_dataset_generator.py -c 500 -o dataset/v2

   # 最后手动合并所有文件
   ```

2. **后台运行** (Windows)
   ```bash
   start "数据生成" .venv\Scripts\python.exe auto_dataset_generator.py -c 1000
   ```

3. **定时生成** (Windows 任务计划)
   可以设置每天凌晨自动运行

---

## 📞 需要帮助？

查看完整文档:
```bash
cat DATASET_GENERATION_GUIDE.md
```

或直接问我！

