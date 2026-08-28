# -*- coding: utf-8 -*-
"""
自动化数据集生成脚本
无需GUI即可批量生成光学瑕疵干涉图数据集
"""

import os
import sys
import traceback
import importlib.util

# 强制使用非GUI后端，避免Qt平台插件问题
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import random

# 配置中文字体显示
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from scipy.ndimage import gaussian_filter
from pathlib import Path
from datetime import datetime
import json

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _load_parameter_class(is_newton: bool):
    """Load the active simulation parameter modules without package-name clashes."""
    root = Path(__file__).resolve().parent
    candidates = list(root.glob("*/parameters.py"))
    for candidate in candidates:
        source = candidate.read_bytes()
        candidate_is_newton = b"self.R_mm" in source
        if candidate_is_newton != is_newton:
            continue
        module_name = "dataset_newton_parameters" if is_newton else "dataset_wedge_parameters"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SimulationParameters
    kind = "Newton" if is_newton else "wedge"
    raise ImportError(f"Could not locate the {kind} simulation parameters module")


WedgeParameters = _load_parameter_class(is_newton=False)
NewtonParameters = _load_parameter_class(is_newton=True)

class DatasetGenerator:
    """自动化数据集生成器"""

    TARGET_GEOMETRY_VERSION = 3
    TARGET_BOX_RADIUS_SCALE = 0.78

    def __init__(
        self,
        output_dir="dataset/images/Auto_Generated",
        num_samples=100,
        weak_shape_focus=False,
        focused_defects=False,
        focus_count_choices=None,
    ):
        """
        初始化数据集生成器

        Args:
            output_dir: 输出目录
            num_samples: 生成的样本数量
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.weak_shape_focus = weak_shape_focus
        self.focus_count_choices = tuple(focus_count_choices or ())
        self.generated_count = 0
        self.defect_types = ["表面划痕", "小凸起", "小凹陷", "灰尘污染", "头发丝遮挡"]
        # 统一命名：凸字统一用"凸"（小凸起、凸起划痕）
        self.sim_types = ["wedge", "newton"] # 新增：仿真类型

        # 统计信息
        self.small_defect_types = self.defect_types[1:3]
        if focused_defects:
            self.defect_types = self.small_defect_types + [self.defect_types[-1]]

        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "by_defect_type": {}
        }

        print(f"[OK] Dataset Generator Initialized")
        print(f"  Output dir: {self.output_dir.absolute()}")
        print(f"  Target samples: {num_samples}")

    def _apply_weak_shape_focus(self, params, sim_type="wedge"):
        """Bias bump/dent samples toward subtle, wide local deformations.

        The two experiments have different physical fields of view.  A
        3-mm radius is a small local target in the wedge field, but is already
        about half of the Newton-ring field, so the focus radius must be
        sampled in experiment-specific units.
        """
        max_count = 4 if sim_type == "newton" else 6
        params.bump_count = min(max_count, (
            int(np.random.choice(self.focus_count_choices))
            if self.focus_count_choices
            else np.random.randint(1, 7)
        ))
        if sim_type == "newton":
            focus_radius_choices = (0.9, 1.0)
            radius_low, radius_high = 0.8, 1.2
        else:
            focus_radius_choices = (2.9, 3.0)
            radius_low, radius_high = 2.7, 3.2
        if np.random.random() < 0.45:
            params.bump_height = 0.02
            params.bump_radius_mm = float(np.random.choice(focus_radius_choices))
        else:
            params.bump_height = float(np.random.uniform(0.014, 0.030))
            params.bump_radius_mm = float(np.random.uniform(radius_low, radius_high))

    @staticmethod
    def _field_limits(sim_type, params):
        """Return the physical image field as (xmin, xmax, ymin, ymax), in m."""
        if sim_type == "newton":
            field_x = float(getattr(params, "screen_length_x", 0.006))
            field_y = float(getattr(params, "screen_length_y", 0.006))
            return -field_x / 2.0, field_x / 2.0, -field_y / 2.0, field_y / 2.0
        return 0.0, 0.030, 0.0, 0.015

    def _ensure_fixed_bump_positions(self, params, sim_type):
        """Make the fixed-position mode contain at least bump_count valid points."""
        if str(getattr(params, "bump_position_mode", "")) != "固定位置":
            return

        xmin, xmax, ymin, ymax = self._field_limits(sim_type, params)
        count = int(getattr(params, "bump_count", 0))
        positions = []
        for raw_position in getattr(params, "fixed_bump_positions", []):
            try:
                x, y = float(raw_position[0]), float(raw_position[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(x) and np.isfinite(y) and xmin <= x <= xmax and ymin <= y <= ymax:
                positions.append((x, y))

        # The UI defaults contain only three fixed points.  Complete them
        # from the same grid generator used by the simulator instead of
        # silently producing fewer targets than bump_count.
        grid_generator = getattr(params, "_generate_grid_positions", None)
        if len(positions) < count and callable(grid_generator):
            for raw_position in grid_generator(xmax - xmin, ymax - ymin):
                candidate = (float(raw_position[0]), float(raw_position[1]))
                if not any(np.hypot(candidate[0] - x, candidate[1] - y) < 1e-12 for x, y in positions):
                    positions.append(candidate)
                if len(positions) >= count:
                    break

        if len(positions) < count:
            raise RuntimeError(
                f"fixed bump positions are insufficient: count={count}, positions={len(positions)}"
            )
        params.fixed_bump_positions = positions[:count]

    @staticmethod
    def _effective_bump_radius_m(params, sim_type):
        """Return the radius actually used by the active simulator."""
        xmin, xmax, ymin, ymax = DatasetGenerator._field_limits(sim_type, params)
        raw_radius_mm = float(getattr(params, "bump_radius_mm", 0.0))
        max_radius_m = 0.48 * min(xmax - xmin, ymax - ymin)
        return float(np.clip(raw_radius_mm * 1e-3, 0.05e-3, max_radius_m))

    def _sample_separated_bump_positions(self, params, sim_type, count):
        """Sample centers with a safety margin so neighboring targets remain separable."""
        xmin, xmax, ymin, ymax = self._field_limits(sim_type, params)
        radius_m = self._effective_bump_radius_m(params, sim_type)
        requested_gap_m = max(0.15e-3, min(0.40e-3, radius_m * 0.15))
        # Preserve a visible border even for the large-radius tail in the
        # wedge field.  Without this cap, r≈7 mm plus a fixed gap would not
        # fit inside the 15-mm field and valid one-target samples would fail.
        border_m = 0.05e-3
        max_gap_m = max(0.0, 0.5 * min(xmax - xmin, ymax - ymin) - radius_m - border_m)
        gap_m = min(requested_gap_m, max_gap_m)
        x_low = xmin + radius_m + gap_m
        x_high = xmax - radius_m - gap_m
        y_low = ymin + radius_m + gap_m
        y_high = ymax - radius_m - gap_m
        min_center_distance = 2.0 * radius_m + gap_m

        if x_low > x_high or y_low > y_high:
            return []

        def greedy_random():
            positions = []
            for _ in range(2500):
                candidate = (
                    float(np.random.uniform(x_low, x_high)),
                    float(np.random.uniform(y_low, y_high)),
                )
                if all(
                    np.hypot(candidate[0] - x, candidate[1] - y) >= min_center_distance
                    for x, y in positions
                ):
                    positions.append(candidate)
                    if len(positions) >= count:
                        return positions
            return []

        # Random candidates preserve diversity.  The grid fallback prevents
        # rare failures when a valid arrangement exists but random rejection
        # sampling does not find it quickly.
        for _ in range(12):
            positions = greedy_random()
            if len(positions) >= count:
                return positions

        corners = [
            (float(x_low), float(y_low)),
            (float(x_high), float(y_low)),
            (float(x_low), float(y_high)),
            (float(x_high), float(y_high)),
        ]
        grid_size = max(2, int(np.ceil(np.sqrt(count))))
        x_candidates = np.linspace(x_low, x_high, max(grid_size, count))
        y_candidates = np.linspace(y_low, y_high, max(grid_size, count))
        grid_candidates = [(float(x), float(y)) for y in y_candidates for x in x_candidates]

        # Try corners first.  This is important for a 6×6-mm Newton field:
        # four separated targets fit at the corners, but a shuffled dense grid
        # can greedily choose interior points and incorrectly conclude that
        # only three targets fit.
        for candidates in (corners + grid_candidates, grid_candidates + corners):
            positions = []
            for candidate in candidates:
                if all(
                    np.hypot(candidate[0] - x, candidate[1] - y) >= min_center_distance
                    for x, y in positions
                ):
                    positions.append(candidate)
                    if len(positions) >= count:
                        return positions
        return []

    def _prepare_local_geometry(self, params, sim_type):
        """Constrain local-target count and center spacing before simulation."""
        requested_count = int(getattr(params, "bump_count", 1))
        max_count = 4 if sim_type == "newton" else 6
        requested_count = max(1, min(requested_count, max_count))

        # A 6×6-mm Newton field cannot support the old 2-mm upper tail as a
        # set of separable local targets.  Keep the experiment local while
        # retaining enough radius variation for training.
        if sim_type == "newton":
            params.bump_radius_mm = float(np.clip(params.bump_radius_mm, 0.2, 1.2))

        for count in range(requested_count, 0, -1):
            positions = self._sample_separated_bump_positions(params, sim_type, count)
            if len(positions) == count:
                params.bump_count = count
                params.bump_position_mode = "固定位置"
                params.fixed_bump_positions = positions
                return

        raise RuntimeError(
            f"could not place a visible local geometry: sim_type={sim_type}, "
            f"count={requested_count}, radius_mm={params.bump_radius_mm}"
        )

    @staticmethod
    def _reuse_local_geometry_for_thickness(params, generated_positions):
        """Compute topography with the exact positions already used by the image."""
        original_mode = getattr(params, "bump_position_mode", None)
        original_fixed_positions = list(getattr(params, "fixed_bump_positions", []))
        params.bump_position_mode = "固定位置"
        params.fixed_bump_positions = list(generated_positions)
        try:
            return params.get_thickness_data()
        finally:
            if original_mode is not None:
                params.bump_position_mode = original_mode
            params.fixed_bump_positions = original_fixed_positions

    def _build_local_target_metadata(self, params, sim_type):
        """Build versioned, image-space ground truth for bump/dent instances."""
        target_count = int(getattr(params, "bump_count", 0))
        raw_positions = list(getattr(params, "generated_bump_positions", []))
        if target_count <= 0 or len(raw_positions) != target_count:
            raise RuntimeError(
                "local-defect geometry mismatch: "
                f"bump_count={target_count}, generated_positions={len(raw_positions)}"
            )

        xmin, xmax, ymin, ymax = self._field_limits(sim_type, params)
        field_x = xmax - xmin
        field_y = ymax - ymin
        radius_mm = float(getattr(params, "bump_radius_mm", 0.0))
        if hasattr(params, "get_bump_radius"):
            radius_m = float(params.get_bump_radius())
        else:
            radius_m = float(np.clip(radius_mm * 1e-3, 0.05e-3, 0.48 * min(field_x, field_y)))
        box_radius_m = radius_m * self.TARGET_BOX_RADIUS_SCALE
        if not np.isfinite(radius_m) or radius_m <= 0:
            raise RuntimeError(f"invalid effective bump radius: {radius_m!r}")

        positions_mm = []
        boxes = []
        for raw_position in raw_positions:
            try:
                cx, cy = float(raw_position[0]), float(raw_position[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise RuntimeError(f"invalid generated bump position: {raw_position!r}") from exc
            if not np.isfinite(cx) or not np.isfinite(cy):
                raise RuntimeError(f"invalid generated bump position: {raw_position!r}")

            x1 = float(np.clip((cx - box_radius_m - xmin) / field_x, 0.0, 1.0))
            x2 = float(np.clip((cx + box_radius_m - xmin) / field_x, 0.0, 1.0))
            # Physical y increases upward; image y increases downward.
            y1 = float(np.clip(1.0 - (cy + box_radius_m - ymin) / field_y, 0.0, 1.0))
            y2 = float(np.clip(1.0 - (cy - box_radius_m - ymin) / field_y, 0.0, 1.0))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"generated bump position has no visible box: {raw_position!r}")
            positions_mm.append([cx * 1000.0, cy * 1000.0])
            boxes.append([x1, y1, x2, y2])

        return {
            "target_geometry_version": self.TARGET_GEOMETRY_VERSION,
            "target_count": target_count,
            "target_radius_mm": radius_m * 1000.0,
            "target_box_radius_scale": self.TARGET_BOX_RADIUS_SCALE,
            "target_positions_mm": positions_mm,
            "target_bboxes_xyxy_normalized": boxes,
        }

    def _get_color_mapping(self, wavelength_nm):
        """根据波长获取颜色映射（与平台一致）"""
        from matplotlib.colors import LinearSegmentedColormap

        wavelength = wavelength_nm * 1e-9  # 转换为米

        # 根据波长获取对应的颜色
        if 380e-9 <= wavelength < 435e-9:  # 紫光
            color = 'purple'
        elif 435e-9 <= wavelength < 450e-9:  # 蓝光
            color = 'blue'
        elif 450e-9 <= wavelength < 492e-9:  # 青光
            color = 'cyan'
        elif 492e-9 <= wavelength < 577e-9:  # 绿光
            color = 'green'
        elif 577e-9 <= wavelength < 597e-9:  # 黄光
            color = 'goldenrod'
        elif 597e-9 <= wavelength < 622e-9:  # 橙光
            color = 'darkorange'
        elif 622e-9 <= wavelength < 760e-9:  # 红光
            color = 'red'
        else:
            color = 'gray'

        # 创建从黑色到波长颜色的渐变
        colors = [(0, 'black'), (1, color)]
        return LinearSegmentedColormap.from_list('custom_cmap', colors)

    def _create_brush_profile(self, brush_size, brush_depth):
        """
        创建画笔剖面 - 严格匹配平台的 drawing_canvas.py 实现
        使用余弦函数 + 幂函数 + 混合函数，并应用中心增强
        """
        diameter = brush_size * 2
        if diameter <= 0:
            return np.zeros((1, 1))

        profile = np.zeros((diameter, diameter))
        center = brush_size

        # 根据画笔大小选择过渡函数（与平台完全一致）
        if brush_size > 15:
            transition_type = 'cosine'
            power_factor = 1.5
        elif brush_size > 8:
            transition_type = 'blend'
            power_factor = 1.8
        else:
            transition_type = 'power'
            power_factor = 2.0

        for i in range(diameter):
            for j in range(diameter):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= brush_size:
                    normalized_dist = distance / brush_size

                    if transition_type == 'cosine':
                        profile_value = 0.5 * (1 + np.cos(np.pi * normalized_dist))
                    elif transition_type == 'power':
                        profile_value = (1 - normalized_dist) ** power_factor
                    elif transition_type == 'blend':
                        cosine_value = 0.5 * (1 + np.cos(np.pi * normalized_dist))
                        power_value = (1 - normalized_dist) ** power_factor
                        blend_weight = normalized_dist
                        profile_value = (1 - blend_weight) * power_value + blend_weight * cosine_value

                    # 应用深度参数
                    profile[i, j] = profile_value * brush_depth

                    # 中心增强（与平台一致：距离<0.3时增强）
                    if normalized_dist < 0.3:
                        center_boost = 1 + 0.5 * (1 - normalized_dist / 0.3)
                        profile[i, j] *= center_boost

        # 归一化到0-1范围
        if profile.max() > 0:
            profile = profile / profile.max()

        # 应用高斯平滑（与平台一致 sigma=1.2）
        profile = gaussian_filter(profile, sigma=1.2)
        profile = np.clip(profile, 0, 1)

        return profile

    def _generate_random_scratch_mask(self, size, sim_type="wedge"):
        """
        生成随机划痕mask - 完全模拟手动在画板上绘制

        关键：生成连续流畅的手绘曲线（而不是短线段聚在一起）
        """
        import random

        # 1. 在画布坐标系上绘制
        if sim_type == "newton":
            canvas_width = 500
            canvas_height = 500
        else:
            canvas_width = 710
            canvas_height = 350
            
        canvas = np.ones((canvas_height, canvas_width), dtype=np.float32) * 255  # 白色

        # 画笔参数（与平台默认一致）
        brush_size = random.randint(12, 18)  # 平台默认15
        # 提高小尺寸(0.01um级)样本比例：让更多笔画使用较小深度。
        if random.random() < 0.45:
            brush_depth = random.uniform(0.01, 0.08)
        else:
            brush_depth = random.uniform(0.08, 0.22)

        # 2. 生成1-3条手绘曲线
        num_strokes = random.randint(1, 3)

        for _ in range(num_strokes):
            # 生成一条流畅的手绘曲线
            stroke_points = self._generate_handdrawn_stroke(canvas_width, canvas_height)

            if len(stroke_points) < 2:
                continue

            # 沿曲线绘制（模拟鼠标拖动）
            step = max(1.0, brush_size / 4.0)

            for i in range(len(stroke_points) - 1):
                x1, y1 = stroke_points[i]
                x2, y2 = stroke_points[i + 1]

                dx = x2 - x1
                dy = y2 - y1
                distance = max(1, np.sqrt(dx*dx + dy*dy))
                steps = max(1, int(distance / step))

                for j in range(steps + 1):
                    t = j / steps if steps > 0 else 0
                    cx = int(x1 + dx * t)
                    cy = int(y1 + dy * t)

                    # 应用画笔到画布
                    self._apply_brush_to_canvas(canvas, cx, cy, brush_size, brush_depth)

        # 3. 将画布灰度转换为mask（白色=0, 黑色=1）
        canvas_mask = (255 - canvas) / 255.0

        # 4. 缩放到目标尺寸
        from scipy.ndimage import zoom
        scale_y = size / canvas_height
        scale_x = size / canvas_width
        mask = zoom(canvas_mask, (scale_y, scale_x), order=1)
        mask = np.clip(mask, 0, 1).astype(np.float32)

        return mask

    def _generate_handdrawn_stroke(self, canvas_width, canvas_height):
        """
        生成一条模拟手绘的流畅曲线（参考你图片中的曲线形状）
        """
        import random
        points = []

        # 边距
        margin_x = 50
        margin_y = 40
        x_min, x_max = margin_x, canvas_width - margin_x
        y_min, y_max = margin_y, canvas_height - margin_y

        # 随机选择笔画类型
        stroke_type = random.choice(['zigzag', 'curve', 'diagonal', 'wave'])

        if stroke_type == 'zigzag':
            # Z字形笔画（参考你图1、图3的效果）
            num_turns = random.randint(2, 5)
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            points.append((x, y))

            direction = random.choice([-1, 1])
            for i in range(num_turns):
                # 水平方向移动
                dx = random.randint(100, 200) * direction
                x_new = max(x_min, min(x_max, x + dx))
                # 沿途添加轻微抖动点
                for step in range(5):
                    t = (step + 1) / 5
                    mid_x = int(x + (x_new - x) * t)
                    mid_y = y + random.randint(-8, 8)
                    points.append((mid_x, mid_y))
                x = x_new

                # 垂直方向移动
                dy = random.randint(50, 120) * (1 if i % 2 == 0 else -1)
                y_new = max(y_min, min(y_max, y + dy))
                for step in range(3):
                    t = (step + 1) / 3
                    mid_x = x + random.randint(-8, 8)
                    mid_y = int(y + (y_new - y) * t)
                    points.append((mid_x, mid_y))
                y = y_new

                direction *= -1

        elif stroke_type == 'curve':
            # 平滑弧线（贝塞尔曲线）
            start_x = random.randint(x_min, x_max // 2)
            start_y = random.randint(y_min, y_max)

            ctrl1_x = start_x + random.randint(80, 180)
            ctrl1_y = start_y + random.randint(-120, 120)
            ctrl2_x = ctrl1_x + random.randint(80, 180)
            ctrl2_y = ctrl1_y + random.randint(-120, 120)
            end_x = max(x_min, min(x_max, ctrl2_x + random.randint(50, 120)))
            end_y = max(y_min, min(y_max, ctrl2_y + random.randint(-60, 60)))

            # 采样贝塞尔曲线点
            for t in np.linspace(0, 1, 60):
                bx = int((1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x +
                        3*(1-t)*t**2 * ctrl2_x + t**3 * end_x)
                by = int((1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y +
                        3*(1-t)*t**2 * ctrl2_y + t**3 * end_y)
                bx = max(x_min, min(x_max, bx))
                by = max(y_min, min(y_max, by))
                points.append((bx, by))

        elif stroke_type == 'diagonal':
            # 斜线（参考你图3右侧）
            start_x = random.randint(x_min, canvas_width // 2)
            start_y = random.randint(y_min, y_max)
            end_x = random.randint(canvas_width // 2, x_max)
            end_y = random.randint(y_min, y_max)

            num_points = 40
            for i in range(num_points):
                t = i / (num_points - 1)
                px = int(start_x + (end_x - start_x) * t + random.randint(-4, 4))
                py = int(start_y + (end_y - start_y) * t + random.randint(-4, 4))
                px = max(x_min, min(x_max, px))
                py = max(y_min, min(y_max, py))
                points.append((px, py))

        else:  # wave
            # 波浪形曲线
            start_x = random.randint(x_min, x_min + 100)
            base_y = random.randint(y_min + 50, y_max - 50)
            amplitude = random.randint(40, 100)
            frequency = random.uniform(0.015, 0.03)

            num_points = 80
            for i in range(num_points):
                px = int(start_x + (x_max - start_x) * i / num_points)
                py = int(base_y + amplitude * np.sin(frequency * px * 2 * np.pi / 10))
                py = max(y_min, min(y_max, py))
                points.append((px, py))

        return points

    def _apply_brush_to_canvas(self, canvas, center_x, center_y, brush_size, brush_depth):
        """
        在画布上应用画笔 - 与平台 drawing_canvas.py 完全一致
        """
        profile = self._create_brush_profile(brush_size, brush_depth)
        half_size = profile.shape[0] // 2

        height, width = canvas.shape

        for i in range(profile.shape[0]):
            for j in range(profile.shape[1]):
                py = center_y - half_size + i  # 行
                px = center_x - half_size + j  # 列

                if 0 <= py < height and 0 <= px < width:
                    weight = profile[i, j]

                    if weight > 0.01:
                        # 计算新灰度值（与平台完全一致）
                        depth_value = min(brush_depth * weight, 1.0)
                        new_gray = 255 * (1 - depth_value)

                        # 取较深的值（模拟颜色累积）
                        current_gray = canvas[py, px]
                        canvas[py, px] = min(current_gray, new_gray)

    def _sample_bump_geometry(self, sim_type: str) -> tuple:
        """凸起/凹陷的高度与横向半径采样。

        启用弱形变聚焦时约 55% 样本落在弱形变区（h∈[0.014,0.030]μm，
        用户指定的 h≈0.02 重点区间），其余样本保留更强、可见性更好的分层样本；
        半径按实验类型分别聚焦：劈尖 r≈3mm（30mm 视场，占 ~6% 面积），
        牛顿环按视场等比例缩放为 r≈1mm（6mm 视场）——若牛顿环直接用 r=3mm
        会覆盖约 3/4 视场，不再是"小"缺陷且标签框几乎全图。
        其余样本分层覆盖可训练的难度范围，保证难易样本兼顾。
        """
        focus_probability = 0.55 if self.weak_shape_focus else 0.35
        if np.random.random() < focus_probability:
            if sim_type == "newton":
                return float(np.random.uniform(0.014, 0.030)), float(np.random.uniform(0.8, 1.2))
            return float(np.random.uniform(0.014, 0.030)), float(np.random.uniform(2.7, 3.2))

        r_h = np.random.random()
        if r_h < 0.45:
            height = np.random.uniform(0.03, 0.10)
        elif r_h < 0.75:
            height = np.random.uniform(0.10, 1.0)
        else:
            height = np.random.uniform(1.0, 5.0)

        r_r = np.random.random()
        if sim_type == "newton":
            # 牛顿环视场仅 6×6mm，大半径会覆盖大半视场，上限限在 1.2mm
            if r_r < 0.40:
                radius = np.random.uniform(0.2, 0.7)
            elif r_r < 0.75:
                radius = np.random.uniform(0.7, 1.4)
            else:
                radius = np.random.uniform(1.4, 1.5)
        else:
            if r_r < 0.40:
                radius = np.random.uniform(0.2, 1.5)
            elif r_r < 0.75:
                radius = np.random.uniform(1.5, 4.0)
            else:
                radius = np.random.uniform(4.0, 7.0)
        return float(height), float(radius)

    def _get_random_parameters(self, defect_type, sim_type="wedge"):
        """生成随机参数（使用平台已支持的参数）"""
        if sim_type == "newton":
            params = NewtonParameters()
        else:
            params = WedgeParameters()
            
        sim_defect_type = "小突起" if defect_type == "小凸起" else defect_type

        # 设置缺陷类型
        params.defect_type = sim_defect_type

        # 根据缺陷类型设置随机参数
        if defect_type == "表面划痕":
            # 表面划痕需要手动绘制mask
            params.use_manual_scratches = True
            # 统一命名为“凸起划痕/凹陷划痕”，并保持两类都有。
            params.scratch_type = np.random.choice(["凹陷划痕", "凸起划痕"])
            params.manual_scratch_type = params.scratch_type
            # 生成随机划痕mask
            params.manual_scratch_mask = self._generate_random_scratch_mask(params.simulation_size, sim_type)

        elif defect_type in ("小突起", "小凸起"):
            params.use_manual_scratches = False
            # 上限 6：避免超出标注端 max_boxes_per_image 截断造成"同缺陷时标时不标"
            params.bump_count = (
                int(np.random.choice(self.focus_count_choices))
                if self.weak_shape_focus and self.focus_count_choices
                else np.random.randint(1, 7)
            )
            params.bump_height, params.bump_radius_mm = self._sample_bump_geometry(sim_type)
            self._prepare_local_geometry(params, sim_type)

        elif defect_type == "小凹陷":
            params.use_manual_scratches = False
            # 上限 6：避免超出标注端 max_boxes_per_image 截断造成"同缺陷时标时不标"
            params.bump_count = (
                int(np.random.choice(self.focus_count_choices))
                if self.weak_shape_focus and self.focus_count_choices
                else np.random.randint(1, 7)
            )
            params.bump_height, params.bump_radius_mm = self._sample_bump_geometry(sim_type)
            self._prepare_local_geometry(params, sim_type)

        elif defect_type == "灰尘污染":
            params.use_manual_scratches = False
            if sim_type == "newton":
                params.defect_type = "灰尘污染"
                params.dust_density = np.random.randint(0, 30)   # 覆盖控制面板 0~30
            else:
                params.dust_density = np.random.randint(0, 30)

        elif defect_type == "头发丝遮挡":
            params.use_manual_scratches = False
            params.defect_type = "头发丝遮挡"
            params.hair_count = np.random.randint(1, 20)   # 覆盖控制面板 1~20

        # 光学参数：覆盖完整可见光范围 380~760nm，确保各颜色均匀采样
        # 分段采样：紫/蓝(380-492) 20%，绿/黄(492-597) 40%，橙/红(597-760) 40%
        if (
            self.weak_shape_focus
            and defect_type not in self.small_defect_types
            and self.focus_count_choices
            and hasattr(params, "hair_count")
        ):
            params.hair_count = int(np.random.choice(self.focus_count_choices))

        color_roll = np.random.random()
        if color_roll < 0.20:
            params.lambda_nm = np.random.randint(380, 492)   # 紫/蓝/青
        elif color_roll < 0.60:
            params.lambda_nm = np.random.randint(492, 597)   # 绿/黄
        else:
            params.lambda_nm = np.random.randint(597, 760)   # 橙/红

        if sim_type == "newton":
            # 曲率半径：覆盖控制面板完整范围 1000~5000mm
            params.R_mm = np.random.uniform(1000.0, 5000.0)
            # 中心间隙：覆盖控制面板完整范围 0~15μm（避免 0 导致除零，从 0.1 开始）
            params.d0_um = np.random.uniform(0.1, 15.0)
            # 折射率：覆盖控制面板范围 1.0~5.0
            n_val = np.random.uniform(1.0, 5.0)
            params.n = n_val
            params.n1 = n_val
            params.n2 = n_val
            params.n3 = n_val
        else:
            # 劈尖角：覆盖控制面板范围 0.1~20（×1e-5 rad）
            params.theta_rad = np.random.uniform(0.1e-5, 20e-5)
            # 劈尖厚度偏移：覆盖控制面板范围 0~50μm
            params.d0_um = np.random.uniform(0.0, 50.0)
            # 劈尖三层折射率各自独立随机，范围 1.0~3.0
            params.n1 = np.random.uniform(1.0, 3.0)
            params.n2 = np.random.uniform(1.0, 3.0)
            params.n3 = np.random.uniform(1.0, 3.0)

        self._ensure_fixed_bump_positions(params, sim_type)

        return params

    def generate_sample(self, sample_id, defect_type, sim_type="wedge"):
        """生成单个样本的三张图"""
        try:
            # 生成随机参数
            params = self._get_random_parameters(defect_type, sim_type)

            # 关闭matplotlib交互模式
            plt.ioff()

            # 计算理想干涉图 - 返回 (intensity_curve, pattern)
            ideal_curve, ideal_pattern = params.calculate_ideal_interference()

            # 计算缺陷干涉图 - 适配不同类型的返回值数量
            defect_result = params.calculate_defect_interference()
            defect_curve = defect_result[0]
            defect_pattern = defect_result[1]

            local_target_metadata = None
            generated_positions = None
            if defect_type in ("小凸起", "小凹陷"):
                generated_positions = list(getattr(params, "generated_bump_positions", []))
                local_target_metadata = self._build_local_target_metadata(params, sim_type)

            # 获取厚度数据 - 返回 (ideal_thickness, defect_thickness, X, Y)
            if generated_positions is not None:
                ideal_thickness, defect_thickness, X, Y = self._reuse_local_geometry_for_thickness(
                    params, generated_positions
                )
            else:
                ideal_thickness, defect_thickness, X, Y = params.get_thickness_data()

            # 获取颜色映射（与平台一致）
            cmap = self._get_color_mapping(params.lambda_nm)

            # 统一归一化到 [0,1] 显示范围，再施加同参数增强：
            # - wedge pattern = 4·I0·cos² ∈ [0, 4·I0]
            # - newton pattern = 0.05 + 0.95·cos² ∈ [0.05, 1]
            # 归一化必须在增强前完成且两图使用同一确定变换；否则 imshow 的逐图
            # auto vmin/vmax 归一化（两图极值受噪声/模糊影响而不同）会产生全局
            # 灰度映射差异，污染后续差分标注。
            if sim_type == "wedge":
                display_max = 4.0 * float(getattr(params, "I0", 1.0))
                ideal_pattern = np.clip(ideal_pattern / display_max, 0.0, 1.0)
                defect_pattern = np.clip(defect_pattern / display_max, 0.0, 1.0)
            else:
                ideal_pattern = np.clip((ideal_pattern - 0.05) / 0.95, 0.0, 1.0)
                defect_pattern = np.clip((defect_pattern - 0.05) / 0.95, 0.0, 1.0)

            # 同参数增强（核心修复）：理想图与瑕疵图施加完全相同的变换与同一噪声场，
            # 保证后续差分标注中全局增强分量与噪声精确抵消。
            aug_params = self._sample_augmentation_params()
            ideal_pattern = self._apply_augmentation_params(ideal_pattern.astype(np.float32), aug_params)
            defect_pattern = self._apply_augmentation_params(defect_pattern.astype(np.float32), aug_params)

            # 三张图的文件名
            base_filename = f"{sample_id:04d}_{defect_type}"
            ideal_path = self.output_dir / f"{base_filename}_理想干涉图.png"
            defect_path = self.output_dir / f"{base_filename}_有瑕疵干涉图.png"
            topography_path = self.output_dir / f"{base_filename}_三维缺陷形貌.png"

            # 保存干涉图（使用正确的colormap和中文标题）
            ideal_title = "理想劈尖干涉图样" if sim_type == "wedge" else "理想牛顿环干涉图样"
            defect_title_prefix = "有瑕疵劈尖干涉图样" if sim_type == "wedge" else "有瑕疵牛顿环干涉图样"
            
            self._save_interference_image(ideal_pattern, ideal_path, ideal_title, cmap, sim_type)
            self._save_interference_image(defect_pattern, defect_path, f"{defect_title_prefix}（{defect_type}）", cmap, sim_type)

            # 保存3D形貌图
            self._save_topography_image(defect_thickness, topography_path, defect_type, sim_type)

            # 保存参数信息到JSON（含增强参数，便于审计与复现）
            params_path = self.output_dir / f"{base_filename}_参数.json"
            self._save_parameters_json(
                params,
                params_path,
                sim_type,
                aug_params=aug_params,
                local_target_metadata=local_target_metadata,
            )

            self.generated_count += 1
            self.stats["success"] += 1
            self.stats["by_defect_type"][defect_type] = \
                self.stats["by_defect_type"].get(defect_type, 0) + 1

            return True

        except Exception as e:
            print(f"  [FAILED] Sample {sample_id} generation failed: {str(e)}")
            traceback.print_exc()
            self.stats["failed"] += 1
            return False
        finally:
            plt.close('all')

    # ── 数据增强：让仿真图更像真实图 ──────────────────────────────
    # 关键约束：增强参数对"理想图"与"瑕疵图"必须完全一致（含同一噪声场），
    # 否则后续 split_final_dataset.py 的差分标注会被全局差异污染
    # （旧版只增强瑕疵图，导致约一半样本的标签框落在条纹带上）。
    @staticmethod
    def _sample_augmentation_params() -> dict:
        """采样一组增强参数；noise 使用独立种子，保证两图噪声场一致。"""
        aug_params: dict = {}
        # 1. 高斯噪声（50% 概率，强度 0.5~2%）
        if np.random.random() < 0.5:
            aug_params["noise"] = {
                "seed": int(np.random.randint(0, 2**31 - 1)),
                "sigma": float(np.random.uniform(0.005, 0.02)),
            }
        # 2. 高斯模糊（40% 概率，sigma 0.3~1.2px）
        if np.random.random() < 0.4:
            aug_params["blur_sigma"] = float(np.random.uniform(0.3, 1.2))
        # 3. 亮度/对比度扰动（60% 概率）
        if np.random.random() < 0.6:
            aug_params["brightness"] = float(np.random.uniform(-0.08, 0.08))
            aug_params["contrast"] = float(np.random.uniform(0.85, 1.15))
        return aug_params

    @staticmethod
    def _apply_augmentation_params(img_array: np.ndarray, aug_params: dict) -> np.ndarray:
        """按给定参数确定性施加增强：同参数 + 同形状 + 同噪声种子 → 同噪声场。

        注意：不再对单图做 JPEG 压缩伪影——压缩残差与内容相关，
        两图无法对齐，会重新引入差分污染。
        """
        aug = img_array.astype(np.float32, copy=True)

        noise = aug_params.get("noise")
        if noise:
            rng = np.random.RandomState(noise["seed"])
            aug = np.clip(aug + rng.normal(0.0, noise["sigma"], aug.shape), 0.0, 1.0)

        blur_sigma = aug_params.get("blur_sigma")
        if blur_sigma:
            from scipy.ndimage import gaussian_filter as _gf
            aug = np.clip(_gf(aug, sigma=blur_sigma), 0.0, 1.0)

        if "brightness" in aug_params:
            # 亮度偏移 ±8%，对比度缩放 0.85~1.15（两图同一变换，差分中精确抵消）
            aug = np.clip(aug * aug_params["contrast"] + aug_params["brightness"], 0.0, 1.0)

        return aug.astype(np.float32)

    def _save_interference_image(self, pattern, filepath, title, cmap, sim_type="wedge"):
        """保存干涉图。增强已在数组层面由 _apply_augmentation_params 完成。"""
        fig, ax = plt.subplots(figsize=(7, 3.5) if sim_type == "wedge" else (5, 5), dpi=100)
        fig.set_facecolor('#FFFFFF')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if sim_type == "newton":
            extent = [-3, 3, -3, 3]
        else:
            extent = [0, 30, 0, 15]

        ax.imshow(pattern, extent=extent, cmap=cmap, origin='lower', aspect='equal',
                  vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        fig.savefig(filepath, dpi=150, facecolor='white', pad_inches=0)
        plt.close(fig)

    def _save_topography_image(self, thickness_data, filepath, defect_type, sim_type="wedge"):
        """保存3D形貌图（与平台风格一致）"""
        fig = plt.figure(figsize=(9, 4.5) if sim_type == "wedge" else (6, 5), dpi=100)
        fig.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111, projection='3d')

        # 创建网格
        if sim_type == "newton":
            x = np.linspace(-3, 3, thickness_data.shape[1])
            y = np.linspace(-3, 3, thickness_data.shape[0])
        else:
            x = np.linspace(0, 30, thickness_data.shape[1])
            y = np.linspace(0, 15, thickness_data.shape[0])
            
        X, Y = np.meshgrid(x, y)

        # 绘制表面
        surf = ax.plot_surface(X, Y, thickness_data * 1e6, cmap='viridis', alpha=0.9)

        ax.set_xlabel('x (mm)', fontsize=10)
        ax.set_ylabel('y (mm)', fontsize=10)
        ax.set_zlabel('Height (um)', fontsize=10)
        ax.set_title(f'三维缺陷形貌（{defect_type}）', fontsize=14, fontweight='bold')

        # 设置视角
        ax.view_init(elev=30, azim=45)

        fig.colorbar(surf, ax=ax, shrink=0.6, label='Height (um)')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _save_parameters_json(
        self,
        params,
        filepath,
        sim_type="wedge",
        aug_params=None,
        local_target_metadata=None,
    ):
        """Save parameters to JSON"""
        # 有的参数类可能没有 export_parameters 方法，这里做一个兼容处理
        if hasattr(params, 'export_parameters'):
            params_dict = params.export_parameters()
        else:
            params_dict = {
                "defect_type": params.defect_type,
                "lambda_nm": params.lambda_nm,
                "bump_count": getattr(params, 'bump_count', 0),
                "bump_height": getattr(params, 'bump_height', 0),
                "bump_radius_mm": getattr(params, 'bump_radius_mm', 2.4),
            }
            if sim_type == "newton":
                params_dict["R_mm"] = getattr(params, 'R_mm', 1000.0)
                params_dict["d0_um"] = getattr(params, 'd0_um', 5.0)

        params_dict["sim_type"] = sim_type # 记录仿真类型
        if aug_params is not None:
            params_dict["aug_params"] = aug_params # 记录两图共用的增强参数
        if local_target_metadata is not None:
            params_dict.update(local_target_metadata)

        # 对外统一命名：小凸起 / 凸起划痕
        if params_dict.get("defect_type") == "小突起":
            params_dict["defect_type"] = "小凸起"
        if params_dict.get("bump_type") == "小突起":
            params_dict["bump_type"] = "小凸起"
        if params_dict.get("scratch_type") == "突起划痕":
            params_dict["scratch_type"] = "凸起划痕"
        if params_dict.get("manual_scratch_type") == "突起划痕":
            params_dict["manual_scratch_type"] = "凸起划痕"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)

    def generate_dataset(self):
        """批量生成数据集"""
        print(f"\n{'='*60}")
        print(f"Generating dataset...")
        print(f"{'='*60}\n")

        start_time = datetime.now()

        # 初始化所有缺陷类型的计数
        for defect_type in self.defect_types:
            self.stats['by_defect_type'][defect_type] = 0

        # 类别配比向小凸起/小凹陷倾斜：凸起:凹陷:划痕:灰尘:头发 = 10:10:6:7:7
        # （旧版均匀分配，配合灰尘/头发每图多实例，造成严重的实例级失衡）
        class_weights = {"小凸起": 10, "小凹陷": 10, "表面划痕": 6, "灰尘污染": 7, "头发丝遮挡": 7}
        total_weight = sum(class_weights.get(t, 1) for t in self.defect_types)
        samples_by_type = {
            t: max(1, int(round(self.num_samples * class_weights.get(t, 1) / total_weight)))
            for t in self.defect_types
        }
        # 余数补到最后一类，保证总数等于 num_samples
        remainder = self.num_samples - sum(samples_by_type.values())
        last_type = self.defect_types[-1]
        samples_by_type[last_type] = max(1, samples_by_type[last_type] + remainder)

        sample_id = 1

        for defect_idx, defect_type in enumerate(self.defect_types):
            num_for_this_type = samples_by_type[defect_type]

            print(f"[GENERATING] Defect Type: {defect_type}")
            print(f"  Target count: {num_for_this_type}")

            for i in range(num_for_this_type):
                # 随机选择仿真类型 (1:1 混合劈尖和牛顿环)
                sim_type = np.random.choice(self.sim_types)
                
                self.generate_sample(sample_id, defect_type, sim_type)
                sample_id += 1

                # 进度显示
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{num_for_this_type} (Current: {sim_type})")

            print(f"  Completed {defect_type}: {self.stats['by_defect_type'].get(defect_type, 0)} samples\n")

        # 统计总结
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"{'='*60}")
        print(f"[SUCCESS] Dataset generation completed!")
        print(f"{'='*60}")
        print(f"Time elapsed: {duration:.1f} seconds")
        print(f"Success: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Total: {self.stats['success'] + self.stats['failed']}")
        print(f"\nDefect type distribution:")
        for defect_type, count in self.stats['by_defect_type'].items():
            print(f"  * {defect_type}: {count}")
        print(f"\nOutput dir: {self.output_dir.absolute()}")
        print(f"{'='*60}\n")

        return self.stats


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Auto-generate optical defect detection dataset')
    parser.add_argument('--count', '-c', type=int, default=100,
                        help='Total number of samples to generate (default: 100)')
    parser.add_argument('--output', '-o', type=str,
                        default='dataset/images/Auto_Generated',
                        help='Output directory (default: dataset/images/Auto_Generated)')
    parser.add_argument('--continue', '-cc', action='store_true',
                        help='Continue from previous generation')
    parser.add_argument(
        '--weak-shape-focus',
        action='store_true',
        help='Concentrate bump/dent samples around 0.02 um and 2.9/3.0 mm.',
    )
    parser.add_argument(
        '--focused-defects',
        action='store_true',
        help='Generate only bump, dent, and hair samples.',
    )
    parser.add_argument(
        '--focus-counts',
        nargs='+',
        type=int,
        default=None,
        help='Optional target-count choices for focus-mode samples.',
    )
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed.')

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # 创建生成器并执行
    generator = DatasetGenerator(
        output_dir=args.output,
        num_samples=args.count,
        weak_shape_focus=args.weak_shape_focus,
        focused_defects=args.focused_defects,
        focus_count_choices=args.focus_counts,
    )

    stats = generator.generate_dataset()

    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
