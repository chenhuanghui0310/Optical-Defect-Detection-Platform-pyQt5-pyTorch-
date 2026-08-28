# -*- coding: utf-8 -*-
"""临时诊断：复现 0009 样本的增强链路，检查噪声/模糊为何未在差分中抵消。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_dataset_generator import WedgeParameters

SMOKE = r"d:\physics_compisition_2026\Physics\dataset\images\_smoke_v2"

prefix = "0009_小凹陷"
with open(os.path.join(SMOKE, prefix + "_参数.json"), encoding="utf-8") as f:
    meta = json.load(f)

params = WedgeParameters()
for key in (
    "lambda_nm", "d0_um", "theta_rad", "n1", "n2", "n3",
    "defect_type", "bump_count", "bump_height", "bump_radius_mm",
    "bump_position_mode",
):
    if key in meta and hasattr(params, key):
        setattr(params, key, meta[key])
params.defect_type = "小凹陷"

_, ideal = params.calculate_ideal_interference()
_, defect = params.calculate_defect_interference()

print("ideal shape/dtype:", ideal.shape, ideal.dtype, "range", ideal.min(), ideal.max())
print("defect shape/dtype:", defect.shape, defect.dtype, "range", defect.min(), defect.max())

raw_diff = np.abs(ideal.astype(np.float32) - defect.astype(np.float32))
print("raw diff mean=%.4f max=%.3f" % (raw_diff.mean(), raw_diff.max()))

# 关键检查：两个 pattern 的数值是否完全一致地进入增强
aug_params = meta.get("aug_params", {})
print("aug_params:", aug_params)

from auto_dataset_generator import DatasetGenerator

a1 = DatasetGenerator._apply_augmentation_params(ideal.astype(np.float32), aug_params)
a2 = DatasetGenerator._apply_augmentation_params(defect.astype(np.float32), aug_params)
aug_diff = np.abs(a1 - a2)
print("aug diff mean=%.4f max=%.3f >0.05=%.1f%%" % (aug_diff.mean(), aug_diff.max(), (aug_diff > 0.05).mean() * 100))

# 逐项排查
if "noise" in aug_params:
    rng1 = np.random.RandomState(aug_params["noise"]["seed"])
    n1 = rng1.normal(0.0, aug_params["noise"]["sigma"], ideal.shape)
    rng2 = np.random.RandomState(aug_params["noise"]["seed"])
    n2 = rng2.normal(0.0, aug_params["noise"]["sigma"], defect.shape)
    print("noise identical:", np.array_equal(n1, n2), "shapes:", n1.shape, n2.shape)

# 检查 clip 引起的非抵消（饱和像素处）
if "brightness" in aug_params:
    x1 = np.clip(ideal * aug_params["contrast"] + aug_params["brightness"], 0, 1)
    x2 = np.clip(defect * aug_params["contrast"] + aug_params["brightness"], 0, 1)
    print("brightness-only diff mean=%.4f" % np.abs(x1 - x2).mean())

# 渲染路径：imshow 会按数据 min/max 自动归一化 —— 检查两图 min/max 是否不同！
print("ideal min/max:", float(ideal.min()), float(ideal.max()))
print("defect min/max:", float(defect.min()), float(defect.max()))
