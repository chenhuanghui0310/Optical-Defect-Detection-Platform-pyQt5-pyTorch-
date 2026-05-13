#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速开始脚本 - 数据集自动化生成
无需任何配置，1条命令开始生成！
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_dataset_generator import DatasetGenerator


def quick_start():
    """快速开始（使用默认参数）"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║          光学瑕疵检测 - 自动化数据集生成器                      ║
║          Optical Defect Detection Auto Dataset Generator       ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # 生成500张图片（作为入门示例）
    # 建议从小规模开始，逐步扩大
    generator = DatasetGenerator(
        output_dir="dataset/images/Auto_Generated",
        num_samples=300  # 第一次生成300张，每种缺陷约75张
    )

    stats = generator.generate_dataset()

    # 打印生成结果总结
    print("""
╔════════════════════════════════════════════════════════════════╗
║                       📊 生成结果总结                           ║
╚════════════════════════════════════════════════════════════════╝
    """)

    if stats['success'] > 0:
        print(f"""
✓ 成功生成 {stats['success']} 个样本

📍 文件位置:
   dataset/images/Auto_Generated/

📋 每个样本包含 3 张图片:
   • *_理想干涉图.png        (理想干涉条纹)
   • *_有瑕疵干涉图.png      (缺陷干涉条纹)
   • *_三维缺陷形貌.png      (3D形貌可视化)
   • *_参数.json            (参数记录)

💡 下一步建议:
   1. 检查生成的图片质量
   2. 运行 prepare_dataset.py 进行数据预处理
   3. 开始深度学习模型训练

📈 数据集规模:
   当前: {stats['success']} 张有瑕疵干涉图
   建议完整规模: 1000-2000 张
   可通过多次运行此脚本逐步扩充

⚡ 加快生成速度的建议:
   • 使用 GPU 计算（需修改OpticalDefectExperiment代码）
   • 并行化处理（需使用multiprocessing）
   • 减少图片分辨率（修改dpi参数）
    """)

        print("\n" + "="*60)
        print("✓ 数据集生成成功！")
        print("="*60)
        return 0
    else:
        print("✗ 生成失败，请检查错误信息")
        return 1


def advanced_generation():
    """高级生成（自定义参数）"""
    print("\n调用方式:")
    print("  python auto_dataset_generator.py --count 500 --output dataset/new_dataset")
    print("\n参数说明:")
    print("  --count, -c        : 生成的总样本数 (默认: 100)")
    print("  --output, -o       : 输出目录 (默认: dataset/images/Auto_Generated)")
    print("  --continue, -cc    : 接续上次的生成")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        advanced_generation()
        sys.exit(0)

    # 默认：快速开始
    exit_code = quick_start()
    sys.exit(exit_code)
