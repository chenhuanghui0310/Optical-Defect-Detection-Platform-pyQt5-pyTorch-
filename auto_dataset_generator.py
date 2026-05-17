# -*- coding: utf-8 -*-
"""
自动化数据集生成脚本
无需GUI即可批量生成光学瑕疵干涉图数据集
"""

import os
import sys
import traceback

# 强制使用非GUI后端，避免Qt平台插件问题
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

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

from OpticalDefectExperiment.parameters import SimulationParameters as WedgeParameters
from 牛顿环瑕疵.parameters import SimulationParameters as NewtonParameters

class DatasetGenerator:
    """自动化数据集生成器"""

    def __init__(self, output_dir="dataset/images/Auto_Generated", num_samples=100):
        """
        初始化数据集生成器

        Args:
            output_dir: 输出目录
            num_samples: 生成的样本数量
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.generated_count = 0
        self.defect_types = ["表面划痕", "小凸起", "小凹陷", "灰尘污染", "头发丝遮挡"]
        # 统一命名：凸字统一用"凸"（小凸起、凸起划痕）
        self.sim_types = ["wedge", "newton"] # 新增：仿真类型

        # 统计信息
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "by_defect_type": {}
        }

        print(f"[OK] Dataset Generator Initialized")
        print(f"  Output dir: {self.output_dir.absolute()}")
        print(f"  Target samples: {num_samples}")

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
            params.bump_count = np.random.randint(1, 11)   # 覆盖控制面板 1~10
            if sim_type == "newton":
                # 高度：60% 小尺寸(0.01~0.1μm)，40% 覆盖到控制面板上限 5.0μm
                r_h = np.random.random()
                if r_h < 0.60:
                    params.bump_height = np.random.uniform(0.01, 0.10)
                elif r_h < 0.80:
                    params.bump_height = np.random.uniform(0.10, 1.0)
                else:
                    params.bump_height = np.random.uniform(1.0, 5.0)
                # 半径：三段分层覆盖 0.2~7.0mm（对齐控制面板 0.2~7.0）
                r_r = np.random.random()
                if r_r < 0.40:
                    params.bump_radius_mm = np.random.uniform(0.2, 1.5)    # 小半径
                elif r_r < 0.75:
                    params.bump_radius_mm = np.random.uniform(1.5, 4.0)    # 中半径
                else:
                    params.bump_radius_mm = np.random.uniform(4.0, 7.0)    # 大半径
            else:
                # 劈尖小凸起
                r_h = np.random.random()
                if r_h < 0.60:
                    params.bump_height = np.random.uniform(0.01, 0.10)
                elif r_h < 0.80:
                    params.bump_height = np.random.uniform(0.10, 1.0)
                else:
                    params.bump_height = np.random.uniform(1.0, 5.0)
                # 半径：覆盖控制面板 0.2~7.0mm
                r_r = np.random.random()
                if r_r < 0.35:
                    params.bump_radius_mm = np.random.uniform(0.2, 2.0)
                elif r_r < 0.70:
                    params.bump_radius_mm = np.random.uniform(2.0, 4.5)
                else:
                    params.bump_radius_mm = np.random.uniform(4.5, 7.0)

            params.bump_position_mode = np.random.choice(["网格排列", "随机分布", "固定位置"])

        elif defect_type == "小凹陷":
            params.use_manual_scratches = False
            params.bump_count = np.random.randint(1, 11)   # 覆盖控制面板 1~10
            if sim_type == "newton":
                r_h = np.random.random()
                if r_h < 0.60:
                    params.bump_height = np.random.uniform(0.01, 0.10)
                elif r_h < 0.80:
                    params.bump_height = np.random.uniform(0.10, 1.0)
                else:
                    params.bump_height = np.random.uniform(1.0, 5.0)
                # 半径：覆盖控制面板 0.2~7.0mm
                r_r = np.random.random()
                if r_r < 0.40:
                    params.bump_radius_mm = np.random.uniform(0.2, 1.5)
                elif r_r < 0.75:
                    params.bump_radius_mm = np.random.uniform(1.5, 4.0)
                else:
                    params.bump_radius_mm = np.random.uniform(4.0, 7.0)
            else:
                # 劈尖小凹陷
                r_h = np.random.random()
                if r_h < 0.60:
                    params.bump_height = np.random.uniform(0.01, 0.10)
                elif r_h < 0.80:
                    params.bump_height = np.random.uniform(0.10, 1.0)
                else:
                    params.bump_height = np.random.uniform(1.0, 5.0)
                # 半径：覆盖控制面板 0.2~7.0mm
                r_r = np.random.random()
                if r_r < 0.35:
                    params.bump_radius_mm = np.random.uniform(0.2, 2.0)
                elif r_r < 0.70:
                    params.bump_radius_mm = np.random.uniform(2.0, 4.5)
                else:
                    params.bump_radius_mm = np.random.uniform(4.5, 7.0)
            params.bump_position_mode = np.random.choice(["网格排列", "随机分布", "固定位置"])

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

            # 获取厚度数据 - 返回 (ideal_thickness, defect_thickness, X, Y)
            ideal_thickness, defect_thickness, X, Y = params.get_thickness_data()

            # 获取颜色映射（与平台一致）
            cmap = self._get_color_mapping(params.lambda_nm)

            # 三张图的文件名
            base_filename = f"{sample_id:04d}_{defect_type}"
            ideal_path = self.output_dir / f"{base_filename}_理想干涉图.png"
            defect_path = self.output_dir / f"{base_filename}_有瑕疵干涉图.png"
            topography_path = self.output_dir / f"{base_filename}_三维缺陷形貌.png"

            # 保存干涉图（使用正确的colormap和中文标题）
            ideal_title = "理想劈尖干涉图样" if sim_type == "wedge" else "理想牛顿环干涉图样"
            defect_title_prefix = "有瑕疵劈尖干涉图样" if sim_type == "wedge" else "有瑕疵牛顿环干涉图样"
            
            self._save_interference_image_augmented(ideal_pattern, ideal_path, ideal_title, cmap, sim_type, augment=False)
            self._save_interference_image_augmented(defect_pattern, defect_path, f"{defect_title_prefix}（{defect_type}）", cmap, sim_type, augment=True)

            # 保存3D形貌图
            self._save_topography_image(defect_thickness, topography_path, defect_type, sim_type)

            # 保存参数信息到JSON
            params_path = self.output_dir / f"{base_filename}_参数.json"
            self._save_parameters_json(params, params_path, sim_type)

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
    @staticmethod
    def _augment_interference_image(img_array: np.ndarray) -> np.ndarray:
        """
        对干涉图 numpy 数组（H×W，float32，0~1）施加随机增强。
        增强在保存前应用，不影响理想图（理想图只用于差分标注，不送入训练）。

        增强项：
          1. 高斯噪声（模拟传感器噪声）
          2. 高斯模糊（模拟镜头/对焦模糊）
          3. 亮度/对比度随机扰动
          4. JPEG 压缩伪影（模拟图片传输/保存损失）
        """
        aug = img_array.copy()

        # 1. 高斯噪声（50% 概率，强度 0.5~2%）
        if np.random.random() < 0.5:
            sigma = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, sigma, aug.shape)
            aug = np.clip(aug + noise, 0.0, 1.0)

        # 2. 高斯模糊（40% 概率，sigma 0.3~1.2px）
        if np.random.random() < 0.4:
            from scipy.ndimage import gaussian_filter as _gf
            blur_sigma = np.random.uniform(0.3, 1.2)
            aug = _gf(aug, sigma=blur_sigma)
            aug = np.clip(aug, 0.0, 1.0)

        # 3. 亮度/对比度扰动（60% 概率）
        if np.random.random() < 0.6:
            # 亮度偏移 ±8%
            brightness = np.random.uniform(-0.08, 0.08)
            # 对比度缩放 0.85~1.15
            contrast = np.random.uniform(0.85, 1.15)
            aug = np.clip(aug * contrast + brightness, 0.0, 1.0)

        return aug.astype(np.float32)

    def _save_interference_image_augmented(
        self, pattern, filepath, title, cmap, sim_type="wedge", augment=False
    ):
        """保存干涉图，augment=True 时对图像数据施加随机增强后再保存。"""
        aug_pattern = pattern
        if augment:
            aug_pattern = self._augment_interference_image(pattern.astype(np.float32))

        fig, ax = plt.subplots(figsize=(7, 3.5) if sim_type == "wedge" else (5, 5), dpi=100)
        fig.set_facecolor('#FFFFFF')

        if sim_type == "newton":
            extent = [-3, 3, -3, 3]
        else:
            extent = [0, 30, 0, 15]

        ax.imshow(aug_pattern, extent=extent, cmap=cmap, origin='lower', aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # JPEG 压缩伪影（30% 概率）：先保存为 JPEG 再读回，再保存为 PNG
        if augment and np.random.random() < 0.3:
            import tempfile, cv2 as _cv2
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            plt.tight_layout()
            plt.savefig(tmp.name, dpi=150, bbox_inches='tight',
                        facecolor='white', format='jpeg',
                        pil_kwargs={"quality": np.random.randint(60, 85)})
            plt.close(fig)
            # 读回再存为 PNG
            img_bgr = _cv2.imread(tmp.name)
            os.unlink(tmp.name)
            if img_bgr is not None:
                _, buf = _cv2.imencode(".png", img_bgr)
                buf.tofile(str(filepath))
                return
        else:
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
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

    def _save_parameters_json(self, params, filepath, sim_type="wedge"):
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

        # 每种缺陷类型生成的样本数（确保每类都有）
        samples_per_type = max(1, self.num_samples // len(self.defect_types))
        remainder = self.num_samples - samples_per_type * len(self.defect_types)

        sample_id = 1

        for defect_idx, defect_type in enumerate(self.defect_types):
            # 最后一种类型处理余数
            num_for_this_type = samples_per_type + (remainder if defect_idx == len(self.defect_types) - 1 else 0)
            # 保证每类至少1个
            num_for_this_type = max(1, num_for_this_type)

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

    args = parser.parse_args()

    # 创建生成器并执行
    generator = DatasetGenerator(
        output_dir=args.output,
        num_samples=args.count
    )

    stats = generator.generate_dataset()

    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
