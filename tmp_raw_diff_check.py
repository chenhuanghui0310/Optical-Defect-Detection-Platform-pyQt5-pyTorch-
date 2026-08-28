# -*- coding: utf-8 -*-
"""临时诊断：重算指定样本的原始（未增强）理想/瑕疵图样差分，定位全局差异来源。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_dataset_generator import WedgeParameters, NewtonParameters

SMOKE = r"d:\physics_compisition_2026\Physics\dataset\images\_smoke_v2"


def load_meta(prefix):
    with open(os.path.join(SMOKE, prefix + "_参数.json"), encoding="utf-8") as f:
        return json.load(f)


def run(prefix):
    meta = load_meta(prefix)
    sim_type = meta.get("sim_type", "wedge")
    params = NewtonParameters() if sim_type == "newton" else WedgeParameters()
    # 只回填与瑕疵/光学相关的字段（import_parameters 依赖完整字段，手动逐个设置更稳）
    for key in (
        "lambda_nm", "d0_um", "theta_rad", "n1", "n2", "n3",
        "defect_type", "scratch_type", "bump_count", "bump_height",
        "bump_radius_mm", "bump_type", "bump_position_mode",
        "fixed_bump_positions", "dust_density", "hair_count",
        "use_manual_scratches", "manual_scratch_type",
    ):
        if key in meta and hasattr(params, key):
            setattr(params, key, meta[key])

    _, ideal = params.calculate_ideal_interference()
    defect_result = params.calculate_defect_interference()
    defect = defect_result[1]

    diff = np.abs(ideal.astype(np.float32) - defect.astype(np.float32))
    print(
        "%-18s sim=%-6s type=%-4s h=%-8.4f n=%-4.2f | raw diff mean=%.4f max=%.3f >0.05占比=%.1f%%"
        % (
            prefix, sim_type, meta.get("defect_type"),
            float(meta.get("bump_height", -1)),
            float(meta.get("n", meta.get("n2", -1))),
            diff.mean(), diff.max(), (diff > 0.05).mean() * 100,
        )
    )


if __name__ == "__main__":
    for prefix in sys.argv[1:]:
        run(prefix)
