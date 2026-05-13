# -*- coding: utf-8 -*-
"""
defect_platform.simulation — 物理仿真核心 API（UI 无关）
=========================================================
封装劈尖和牛顿环两种干涉实验的仿真逻辑，提供统一接口。
可直接用于桌面端、命令行脚本，以及未来的网页后端。
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np

# 确保父目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .types import SimulationResult

# 支持的瑕疵类型
DEFECT_TYPES = ["无瑕疵", "表面划痕", "小凸起", "小凹陷", "灰尘污染", "头发丝遮挡"]

# 内部映射：统一"凸"字命名 → 物理仿真内部名称
_DEFECT_NAME_MAP = {
    "小凸起": "小突起",
    "凸起划痕": "突起划痕",
}


def _normalize_defect_name(name: str) -> str:
    """将对外统一命名转换为物理仿真内部名称"""
    return _DEFECT_NAME_MAP.get(name, name)


class DefectSimulator:
    """
    光学干涉瑕疵仿真器（UI 无关）

    支持劈尖（wedge）和牛顿环（newton）两种实验类型。

    参数
    ----
    sim_type : "wedge" | "newton"
        仿真类型，默认劈尖。

    示例
    ----
    >>> sim = DefectSimulator(sim_type="wedge")
    >>> result = sim.simulate(defect_type="小凸起", bump_height=0.05)
    >>> print(result.defect_pattern.shape)
    """

    def __init__(self, sim_type: Literal["wedge", "newton"] = "wedge") -> None:
        self.sim_type = sim_type
        self._params = self._load_params()

    def _load_params(self):
        """懒加载对应实验的参数类"""
        if self.sim_type == "wedge":
            wedge_dir = str(_ROOT / "劈尖瑕疵0329")
            if wedge_dir not in sys.path:
                sys.path.insert(0, wedge_dir)
            # 避免模块缓存污染
            import importlib
            spec = importlib.util.spec_from_file_location(
                "wedge_parameters",
                str(_ROOT / "劈尖瑕疵0329" / "parameters.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.SimulationParameters()
        else:
            newton_dir = str(_ROOT / "牛顿环瑕疵")
            if newton_dir not in sys.path:
                sys.path.insert(0, newton_dir)
            import importlib
            spec = importlib.util.spec_from_file_location(
                "newton_parameters",
                str(_ROOT / "牛顿环瑕疵" / "parameters.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.SimulationParameters()

    @property
    def params(self):
        """访问底层参数对象（高级用法）"""
        return self._params

    def set_optical_params(
        self,
        lambda_nm: int | None = None,
        d0_um: float | None = None,
        theta_rad: float | None = None,
        R_mm: float | None = None,
    ) -> None:
        """设置光学参数"""
        if lambda_nm is not None:
            self._params.lambda_nm = lambda_nm
        if d0_um is not None:
            self._params.d0_um = d0_um
        if self.sim_type == "wedge" and theta_rad is not None:
            self._params.theta_rad = theta_rad
        if self.sim_type == "newton" and R_mm is not None:
            self._params.R_mm = R_mm

    def simulate(
        self,
        defect_type: str = "无瑕疵",
        bump_height: float | None = None,
        bump_count: int | None = None,
        bump_radius_mm: float | None = None,
        bump_position_mode: str | None = None,
        dust_density: int | None = None,
        scratch_type: str | None = None,
        scratch_mask: np.ndarray | None = None,
    ) -> SimulationResult:
        """
        执行仿真并返回结果。

        参数
        ----
        defect_type : str
            瑕疵类型，可选值见 DEFECT_TYPES。
        bump_height : float, optional
            突起/凹陷高度（μm）。
        bump_count : int, optional
            突起/凹陷数量。
        bump_radius_mm : float, optional
            突起/凹陷横向半径（mm）。
        bump_position_mode : str, optional
            位置模式："网格排列" | "随机分布" | "固定位置"。
        dust_density : int, optional
            灰尘密度（%）。
        scratch_type : str, optional
            划痕类型："凹陷划痕" | "凸起划痕"。
        scratch_mask : np.ndarray, optional
            手动绘制的划痕掩码（0~1 归一化）。

        返回
        ----
        SimulationResult
        """
        p = self._params

        # 统一命名转换
        internal_defect = _normalize_defect_name(defect_type)
        p.defect_type = internal_defect

        # 设置瑕疵参数
        if bump_height is not None:
            p.bump_height = bump_height
        if bump_count is not None:
            p.bump_count = bump_count
        if bump_radius_mm is not None:
            p.bump_radius_mm = bump_radius_mm
        if bump_position_mode is not None:
            p.bump_position_mode = bump_position_mode
        if dust_density is not None:
            p.dust_density = dust_density
        if scratch_type is not None:
            p.scratch_type = scratch_type
            p.manual_scratch_type = scratch_type
        if scratch_mask is not None:
            p.manual_scratch_mask = scratch_mask
            p.use_manual_scratches = True
        else:
            if defect_type == "表面划痕":
                p.use_manual_scratches = False

        # 计算干涉图样
        _, ideal_pattern = p.calculate_ideal_interference()
        defect_result = p.calculate_defect_interference()
        _, defect_pattern = defect_result[0], defect_result[1]

        # 获取厚度数据
        ideal_thickness, defect_thickness, _, _ = p.get_thickness_data()

        # 导出参数
        params_dict = p.export_parameters() if hasattr(p, "export_parameters") else {}
        params_dict["sim_type"] = self.sim_type
        # 统一对外命名
        if params_dict.get("defect_type") == "小突起":
            params_dict["defect_type"] = "小凸起"

        return SimulationResult(
            sim_type=self.sim_type,
            defect_type=defect_type,
            ideal_pattern=ideal_pattern,
            defect_pattern=defect_pattern,
            ideal_thickness=ideal_thickness,
            defect_thickness=defect_thickness,
            params_dict=params_dict,
        )

    def generate_images(
        self,
        defect_type: str = "无瑕疵",
        colormap_wavelength_nm: int | None = None,
        dpi: int = 150,
        **kwargs,
    ) -> tuple[bytes, bytes, bytes]:
        """
        生成三张图像的 PNG 字节流：理想干涉图、有瑕疵干涉图、三维形貌图。

        返回
        ----
        (ideal_png_bytes, defect_png_bytes, topo_png_bytes)
            每个元素为 PNG 格式的 bytes，可直接写入文件或通过 HTTP 返回。

        示例（网页后端）
        ----------------
        >>> sim = DefectSimulator("wedge")
        >>> ideal_b, defect_b, topo_b = sim.generate_images("小凸起", bump_height=0.05)
        >>> # Flask 示例：
        >>> return send_file(io.BytesIO(defect_b), mimetype="image/png")
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        result = self.simulate(defect_type=defect_type, **kwargs)

        wl = colormap_wavelength_nm or self._params.lambda_nm
        cmap = self._make_cmap(wl)

        # 坐标范围
        if self.sim_type == "newton":
            extent = [-3, 3, -3, 3]
            xlabel = ylabel = "mm"
        else:
            extent = [0, 30, 0, 15]
            xlabel, ylabel = "x (mm)", "y (mm)"

        figsize_2d = (7, 3.5) if self.sim_type == "wedge" else (5, 5)

        def _render_2d(pattern, title) -> bytes:
            fig, ax = plt.subplots(figsize=figsize_2d, dpi=dpi)
            ax.imshow(pattern, extent=extent, cmap=cmap, origin="lower", aspect="equal")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        def _render_3d(thickness) -> bytes:
            fig = plt.figure(figsize=(9, 4.5) if self.sim_type == "wedge" else (6, 5), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            if self.sim_type == "newton":
                x = np.linspace(-3, 3, thickness.shape[1])
                y = np.linspace(-3, 3, thickness.shape[0])
            else:
                x = np.linspace(0, 30, thickness.shape[1])
                y = np.linspace(0, 15, thickness.shape[0])
            X, Y = np.meshgrid(x, y)
            surf = ax.plot_surface(X, Y, thickness * 1e6, cmap="viridis", alpha=0.9)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("Height (μm)")
            ax.set_title(f"三维缺陷形貌（{defect_type}）", fontsize=12, fontweight="bold")
            ax.view_init(elev=30, azim=45)
            fig.colorbar(surf, ax=ax, shrink=0.6, label="Height (μm)")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        exp_name = "劈尖" if self.sim_type == "wedge" else "牛顿环"
        ideal_bytes = _render_2d(result.ideal_pattern, f"理想{exp_name}干涉图样")
        defect_bytes = _render_2d(
            result.defect_pattern, f"有瑕疵{exp_name}干涉图样（{defect_type}）"
        )
        topo_bytes = _render_3d(result.defect_thickness)

        return ideal_bytes, defect_bytes, topo_bytes

    @staticmethod
    def _make_cmap(wavelength_nm: int):
        """根据波长生成颜色映射"""
        from matplotlib.colors import LinearSegmentedColormap

        wl = wavelength_nm * 1e-9
        if 380e-9 <= wl < 435e-9:
            color = "purple"
        elif 435e-9 <= wl < 450e-9:
            color = "blue"
        elif 450e-9 <= wl < 492e-9:
            color = "cyan"
        elif 492e-9 <= wl < 577e-9:
            color = "green"
        elif 577e-9 <= wl < 597e-9:
            color = "goldenrod"
        elif 597e-9 <= wl < 622e-9:
            color = "darkorange"
        elif 622e-9 <= wl < 760e-9:
            color = "red"
        else:
            color = "gray"
        return LinearSegmentedColormap.from_list("defect_cmap", [(0, "black"), (1, color)])

    def get_supported_defect_types(self) -> list[str]:
        """返回支持的瑕疵类型列表"""
        return DEFECT_TYPES.copy()
