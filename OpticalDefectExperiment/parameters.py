import numpy as np
import random


class SimulationParameters:
    """仿真参数管理类"""

    def __init__(self):
        # 基本参数
        self.lambda_nm = 622
        self.d0_um = 10
        self.theta_rad = 10e-5
        self.n1 = 1.5
        self.n2 = 2.0
        self.n3 = 1.5
        self.I0 = 1.0

        # 瑕疵参数
        self.defect_type = "无瑕疵"
        self.dust_density = 5
        self.scratch_type = "凹陷划痕"

        # 小突起/小凹陷参数
        self.bump_count = 3
        self.bump_height = 0.1
        self.bump_type = "小突起"
        self.bump_position_mode = "网格排列"  # 先定义这个属性
        self.fixed_bump_positions = [  # 然后使用它
            (0.005, 0.005), (0.010, 0.005), (0.005, 0.010)
        ]

        # 手动画板参数
        self.use_manual_scratches = False
        self.manual_scratch_mask = None
        self.manual_scratch_type = "凹陷划痕"
        self.brush_size = 15
        self.brush_depth = 0.15
        self.eraser_mode = False

        self.simulation_size = 200
        self.canvas_state = None

    def reset(self):
        """重置所有参数"""
        self.lambda_nm = 622
        self.d0_um = 10
        self.theta_rad = 10e-5
        self.n1 = 1.5
        self.n2 = 2.0
        self.n3 = 1.5
        self.defect_type = "无瑕疵"
        self.dust_density = 5
        self.scratch_type = "凹陷划痕"
        self.bump_count = 3
        self.bump_height = 0.1
        self.bump_type = "小突起"
        self.bump_position_mode = "网格排列"
        self.manual_scratch_type = "凹陷划痕"
        self.brush_size = 15
        self.brush_depth = 0.15
        self.eraser_mode = False
        self.use_manual_scratches = False
        self.manual_scratch_mask = None
        self.canvas_state = None

    def get_wavelength_meters(self):
        return self.lambda_nm * 1e-9

    def get_d0_meters(self):
        return self.d0_um * 1e-6

    def get_theta_radians(self):
        return self.theta_rad

    def get_thickness_data(self):
        """获取理想和有缺陷的厚度数据"""
        lamda = self.get_wavelength_meters()
        d0 = self.get_d0_meters()
        theta = self.get_theta_radians()

        Screen_length_mm = 15  # 直接使用15mm
        Screen_length = Screen_length_mm * 0.001  # 转换为米用于内部计算
        x = np.linspace(0, Screen_length, self.simulation_size)  # 内部计算仍用米
        X, Y = np.meshgrid(x, x)

        # 理想厚度
        ideal_thickness = d0 + X * np.tan(theta)

        # 有缺陷厚度
        defect_thickness = ideal_thickness.copy()

        if self.defect_type == "灰尘污染":
            defect_thickness = self.add_dust_defect(ideal_thickness, X, Y)
        elif self.defect_type == "表面划痕":
            defect_thickness = self.add_scratch_defect(ideal_thickness, X, Y)
        elif self.defect_type == "小突起":
            defect_thickness = self.add_bump_defect(ideal_thickness, X, Y, bump_type="突起")
        elif self.defect_type == "小凹陷":
            defect_thickness = self.add_bump_defect(ideal_thickness, X, Y, bump_type="凹陷")

        return ideal_thickness, defect_thickness, X, Y

    def calculate_ideal_interference(self):
        """计算理想干涉图样"""
        lamda = self.get_wavelength_meters()
        d0 = self.get_d0_meters()
        theta = self.get_theta_radians()

        Screen_length = 0.015
        x = np.linspace(0, Screen_length, self.simulation_size)

        thickness = d0 + x * np.tan(theta)

        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            delta = 2 * self.n2 * thickness + lamda / 2
        else:
            delta = 2 * self.n2 * thickness

        phase_diff = (2 * np.pi * delta) / lamda
        intensity_curve = 4 * self.I0 * (np.cos(phase_diff / 2)) ** 2

        X, Y = np.meshgrid(x, x)
        thickness_2d = d0 + X * np.tan(theta)
        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            delta_2d = 2 * self.n2 * thickness_2d + lamda / 2
        else:
            delta_2d = 2 * self.n2 * thickness_2d
        phase_diff_2d = (2 * np.pi * delta_2d) / lamda
        pattern = 4 * self.I0 * (np.cos(phase_diff_2d / 2)) ** 2

        return intensity_curve, pattern

    def calculate_defect_interference(self):
        """计算有瑕疵干涉图样 - 增强小突起/小凹陷效应"""
        lamda = self.get_wavelength_meters()
        d0 = self.get_d0_meters()
        theta = self.get_theta_radians()

        Screen_length = 0.015
        x = np.linspace(0, Screen_length, self.simulation_size)
        X, Y = np.meshgrid(x, x)

        base_thickness = d0 + X * np.tan(theta)
        thickness = base_thickness.copy()

        if self.defect_type == "无瑕疵":
            pass
        elif self.defect_type == "灰尘污染":
            thickness = self.add_dust_defect(thickness, X, Y)
        elif self.defect_type == "表面划痕":
            thickness = self.add_scratch_defect(thickness, X, Y)
        elif self.defect_type == "小突起":
            thickness = self.add_bump_defect(thickness, X, Y, bump_type="突起")
        elif self.defect_type == "小凹陷":
            thickness = self.add_bump_defect(thickness, X, Y, bump_type="凹陷")

        # 计算光程差时，考虑缺陷对光程的放大效应
        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            delta = 2 * self.n2 * thickness + lamda / 2
        else:
            delta = 2 * self.n2 * thickness

        phase_diff = (2 * np.pi * delta) / lamda
        pattern = 4 * self.I0 * (np.cos(phase_diff / 2)) ** 2
        intensity_curve = pattern[self.simulation_size // 2, :]

        return intensity_curve, pattern

    def add_scratch_defect(self, thickness, X, Y):
        """添加表面划痕缺陷"""
        scratch_thickness = thickness.copy()
        height, width = thickness.shape

        if self.use_manual_scratches and self.manual_scratch_mask is not None:
            # 使用手动绘制的划痕
            mask_height, mask_width = self.manual_scratch_mask.shape

            # 步骤1：对掩码进行高斯平滑处理
            smoothed_mask = self.manual_scratch_mask.copy()
            try:
                from scipy.ndimage import gaussian_filter
                # 应用高斯滤波，sigma值控制平滑程度
                sigma = 0.8  # 可调整参数，越大越平滑
                smoothed_mask = gaussian_filter(smoothed_mask, sigma=sigma)
            except ImportError:
                # 如果scipy不可用，使用简单的均值滤波
                pass

            for i in range(height):
                for j in range(width):
                    mask_y = mask_height - 1 - i
                    if 0 <= mask_y < mask_height:
                        mask_value = smoothed_mask[mask_y, j]  # 使用平滑后的掩码
                    else:
                        mask_value = 0

                    if mask_value > 0.001:
                        # 直接使用归一化深度值（0-1范围）
                        normalized_depth = mask_value
                        depth_factor = normalized_depth  # 直接使用归一化深度作为系数

                        if self.manual_scratch_type == "凹陷划痕":
                            base_depth = -1.0e-6  # 基准深度：-1.0微米
                            depth = base_depth * depth_factor  # 直接乘以深度系数
                        else:
                            base_depth = 1.0e-6  # 基准深度：+1.0微米
                            depth = base_depth * depth_factor  # 直接乘以深度系数

                        scratch_thickness[i, j] += depth

        return scratch_thickness

    def add_dust_defect(self, thickness, X, Y):
        """添加灰尘污染缺陷"""
        dust_thickness = thickness.copy()
        height, width = thickness.shape
        num_dust = int(self.dust_density * 0.01 * height * width / 100)

        for _ in range(num_dust):
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            dust_size = random.uniform(2, 5)
            dust_height = random.uniform(-0.5e-6, 0.5e-6)

            for di in range(-int(dust_size), int(dust_size) + 1):
                for dj in range(-int(dust_size), int(dust_size) + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        dist = np.sqrt(di ** 2 + dj ** 2)
                        if dist <= dust_size:
                            attenuation = np.exp(-dist ** 2 / (dust_size ** 2))
                            dust_thickness[ni, nj] += dust_height * attenuation
        return dust_thickness

    def add_bump_defect(self, thickness, X, Y, bump_type="突起"):
        """增强版：使深度计算与手动划痕完全一致"""
        bump_thickness = thickness.copy()
        height, width = thickness.shape
        screen_length = 0.015

        # 设置与手动划痕一致的基准深度
        base_depth = self.bump_height * 1e-6  # 转换为米

        if bump_type == "突起":
            depth_sign = -1.0
        else:
            depth_sign = 1.0

        base_depth_m = base_depth * depth_sign

        positions = self._generate_bump_positions(screen_length)

        for center_x, center_y in positions[:self.bump_count]:
            base_radius = 0.15 * screen_length
            radius = base_radius

            for i in range(height):
                for j in range(width):
                    x = X[i, j]
                    y = Y[i, j]
                    stretch_factor = 0.5  # X轴是Y轴的2倍
                    stretched_x = (x - center_x) / stretch_factor
                    dist = np.sqrt(stretched_x ** 2 + (y - center_y) ** 2)

                    if dist <= radius:
                        # 关键修改：使用与手动划痕完全一致的计算方式
                        # 直接使用归一化距离计算深度系数
                        normalized_dist = dist / radius

                        # 计算深度系数：中心为1，边缘为0
                        depth_coefficient = 1.0 - normalized_dist

                        # 添加平滑过渡
                        edge_smooth_start = 0.7
                        if normalized_dist > edge_smooth_start:
                            t = (normalized_dist - edge_smooth_start) / (1.0 - edge_smooth_start)
                            depth_coefficient *= (1 - t ** 2)  # 平滑衰减到0

                        # 直接计算深度，与手动划痕一致
                        depth = base_depth_m * depth_coefficient

                        bump_thickness[i, j] += depth

        return bump_thickness

    def _generate_bump_positions(self, screen_length):
        if self.bump_position_mode == "随机分布":
            return self._generate_random_positions(screen_length)
        elif self.bump_position_mode == "网格排列":
            return self._generate_grid_positions(screen_length)
        elif self.bump_position_mode == "固定位置":
            return self.fixed_bump_positions
        else:
            return self._generate_grid_positions(screen_length)

    def _generate_random_positions(self, screen_length):
        positions = []
        for _ in range(max(self.bump_count, 10)):
            center_x = random.uniform(0.1 * screen_length, 0.9 * screen_length)
            center_y = random.uniform(0.1 * screen_length, 0.9 * screen_length)
            positions.append((center_x, center_y))
        return positions

    def _generate_grid_positions(self, screen_length):
        """生成网格排列的位置 - 增加行间间距版本"""
        positions = []
        grid_size = int(np.ceil(np.sqrt(self.bump_count)))

        # 行间距增加系数，使行之间分开更多
        row_spacing_factor = 1.3  # 增加30%的行间距

        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= self.bump_count:
                    break

                # X坐标保持原样
                center_x = (j + 1) * screen_length / (grid_size + 1)
                base_y_spacing = screen_length / (grid_size + 1)
                extra_spacing = (i) * (base_y_spacing * (row_spacing_factor - 1))
                center_y = (i + 1) * base_y_spacing + extra_spacing

                positions.append((center_x, center_y))

        return positions

    def get_defect_description(self):
        """获取瑕疵描述"""
        descriptions = {
            "无瑕疵": "无任何瑕疵的理想劈尖干涉",
            "灰尘污染": f"表面灰尘污染，密度{self.dust_density}%",
            "表面划痕": f"表面划痕，类型：{self.scratch_type}",
            "小突起": f"表面小突起，数量{self.bump_count}个，高度{self.bump_height}μm，位置模式：{self.bump_position_mode}",
            "小凹陷": f"表面小凹陷，数量{self.bump_count}个，深度{self.bump_height}μm，位置模式：{self.bump_position_mode}",
        }
        return descriptions.get(self.defect_type, "未知瑕疵类型")

    def update_from_control_panel(self, control_panel):
        """从控制面板更新参数"""
        self.lambda_nm = control_panel.lambda_spin.value()
        self.d0_um = control_panel.d0_spin.value()
        self.theta_rad = control_panel.theta_spin.value() * 1e-5
        self.n1 = control_panel.n1_spin.value()
        self.n2 = control_panel.n2_spin.value()
        self.n3 = control_panel.n3_spin.value()
        self.defect_type = control_panel.defect_combo.currentText()
        self.dust_density = control_panel.dust_spin.value()
        self.scratch_type = control_panel.scratch_type_combo.currentText()
        self.bump_count = control_panel.bump_spin.value()
        self.bump_height = control_panel.bump_height_spin.value()
        self.bump_position_mode = control_panel.bump_position_combo.currentText()
        self.manual_scratch_type = control_panel.manual_scratch_type_combo.currentText()
        self.brush_size = control_panel.brush_size_spin.value()
        self.brush_depth = control_panel.depth_slider.value() / 100.0
        self.eraser_mode = control_panel.eraser_button.isChecked()

    def get_parameter_summary(self):
        """获取参数摘要"""
        summary = f"""
        仿真参数摘要:
        波长: {self.lambda_nm} nm
        劈尖截距: {self.d0_um} μm
        劈尖夹角: {self.theta_rad * 1e5:.2f} ×10⁻⁵⁵ rad
        折射率: 上板={self.n1}, 薄膜={self.n2}, 下板={self.n3}
        瑕疵类型: {self.defect_type}
        """

        if self.defect_type in ["小突起", "小凹陷"]:
            summary += f"        突起/凹陷数量: {self.bump_count}个\n"
            summary += f"        高度/深度: {self.bump_height}μm\n"
            summary += f"        位置模式: {self.bump_position_mode}\n"
            # 显示实际效应高度
            min_effect, max_effect = self.get_bump_effect_height_range()
            summary += f"        实际效应高度: {min_effect:.1f}-{max_effect:.1f}μm\n"

        elif self.defect_type == "灰尘污染":
            summary += f"        灰尘密度: {self.dust_density}%\n"

        elif self.defect_type == "表面划痕":
            summary += f"        划痕类型: {self.scratch_type}\n"
            if self.use_manual_scratches:
                summary += f"        使用手动划痕: 是\n"
                summary += f"        手动划痕类型: {self.manual_scratch_type}\n"
                depth_range = self.get_manual_scratch_depth_range()
                min_depth = abs(depth_range[0]) * 1e6 if self.manual_scratch_type == "凹陷划痕" else depth_range[0] * 1e6
                max_depth = abs(depth_range[1]) * 1e6 if self.manual_scratch_type == "凹陷划痕" else depth_range[1] * 1e6
                summary += f"        深度范围: {min_depth:.1f}-{max_depth:.1f}μm\n"
            else:
                summary += f"        使用手动划痕: 否\n"

        return summary

    def validate_parameters(self):
        """验证参数有效性"""
        errors = []

        if self.lambda_nm < 380 or self.lambda_nm > 760:
            errors.append("波长应在380-760nm范围内")

        if self.d0_um < 0 or self.d0_um > 50:
            errors.append("劈尖截距应在0-50μm范围内")

        if self.n1 < 1.0 or self.n1 > 3.0:
            errors.append("上板折射率应在1.0-3.0范围内")

        if self.n2 < 1.0 or self.n2 > 3.0:
            errors.append("薄膜折射率应在1.0-3.0范围内")

        if self.n3 < 1.0 or self.n3 > 3.0:
            errors.append("下板折射率应在1.0-3.0范围内")

        if self.defect_type == "小突起" and self.bump_count > 10:
            errors.append("突起数量最多为10个")

        if self.defect_type == "小凹陷" and self.bump_count > 10:
            errors.append("凹陷数量最多为10个")

        if self.dust_density < 0 or self.dust_density > 30:
            errors.append("灰尘密度应在0-30%范围内")

        return errors

    def get_bump_positions_info(self):
        """获取突起/凹陷位置信息"""
        if self.bump_position_mode == "随机分布":
            return "随机分布位置"
        elif self.bump_position_mode == "网格排列":
            grid_size = int(np.ceil(np.sqrt(self.bump_count)))
            return f"网格排列 ({grid_size}×{grid_size})"
        elif self.bump_position_mode == "固定位置":
            positions_str = "; ".join([f"({x * 1000:.1f}mm,{y * 1000:.1f}mm)"
                                       for x, y in self.fixed_bump_positions[:self.bump_count]])
            return f"固定位置: {positions_str}"
        else:
            return "未知位置模式"

    def set_custom_bump_positions(self, positions):
        """设置自定义突起/凹陷位置"""
        if positions and isinstance(positions, list):
            self.fixed_bump_positions = positions
            self.bump_position_mode = "固定位置"

    def add_custom_bump_position(self, x, y):
        """添加自定义突起/凹陷位置"""
        self.fixed_bump_positions.append((x, y))

    def clear_bump_positions(self):
        """清空突起/凹陷位置"""
        self.fixed_bump_positions = [
            (0.005, 0.005),
            (0.010, 0.005),
            (0.005, 0.010),
        ]

    def get_bump_radius(self):
        """根据数量动态调整突起/凹陷半径"""
        base_radius = 0.04 * 0.015
        scale_factor = max(0.5, 1.0 - (self.bump_count - 1) * 0.1)
        return base_radius * scale_factor

    def export_parameters(self):
        """导出参数为字典"""
        return {
            'lambda_nm': self.lambda_nm,
            'd0_um': self.d0_um,
            'theta_rad': self.theta_rad,
            'n1': self.n1,
            'n2': self.n2,
            'n3': self.n3,
            'defect_type': self.defect_type,
            'scratch_type': self.scratch_type,
            'use_manual_scratches': self.use_manual_scratches,
            'manual_scratch_type': self.manual_scratch_type,
            'bump_count': self.bump_count,
            'bump_height': self.bump_height,
            'bump_type': self.bump_type,
            'bump_position_mode': self.bump_position_mode,
            'fixed_bump_positions': self.fixed_bump_positions,
            'dust_density': self.dust_density,
            'brush_size': self.brush_size,
            'brush_depth': self.brush_depth,
            'eraser_mode': self.eraser_mode
        }

    def import_parameters(self, params_dict):
        """从字典导入参数"""
        try:
            self.lambda_nm = params_dict.get('lambda_nm', 622)
            self.d0_um = params_dict.get('d0_um', 10)
            self.theta_rad = params_dict.get('theta_rad', 5e-5)
            self.n1 = params_dict.get('n1', 1.5)
            self.n2 = params_dict.get('n2', 2.0)
            self.n3 = params_dict.get('n3', 1.5)
            self.defect_type = params_dict.get('defect_type', "无瑕疵")
            self.scratch_type = params_dict.get('scratch_type', "凹陷划痕")
            self.use_manual_scratches = params_dict.get('use_manual_scratches', False)
            self.manual_scratch_type = params_dict.get('manual_scratch_type', "凹陷划痕")
            self.bump_count = params_dict.get('bump_count', 3)
            self.bump_height = params_dict.get('bump_height', 0.1)
            self.bump_type = params_dict.get('bump_type', "小突起")
            self.bump_position_mode = params_dict.get('bump_position_mode', "网格排列")
            self.fixed_bump_positions = params_dict.get('fixed_bump_positions', [
                (0.005, 0.005), (0.010, 0.005), (0.005, 0.010)
            ])
            self.dust_density = params_dict.get('dust_density', 5)
            self.brush_size = params_dict.get('brush_size', 15)
            self.brush_depth = params_dict.get('brush_depth', 0.5)
            self.eraser_mode = params_dict.get('eraser_mode', False)

            return True
        except Exception as e:
            print(f"导入参数错误: {e}")
            return False

    def __str__(self):
        return self.get_parameter_summary()

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def compare_with(self, other_params):
        """比较两个参数设置的差异"""
        differences = []

        if self.lambda_nm != other_params.lambda_nm:
            differences.append(f"波长: {self.lambda_nm}nm vs {other_params.lambda_nm}nm")

        if self.d0_um != other_params.d0_um:
            differences.append(f"劈尖截距: {self.d0_um}μm vs {other_params.d0_um}μm")

        if abs(self.theta_rad - other_params.theta_rad) > 1e-7:
            differences.append(f"劈尖夹角: {self.theta_rad:.2e}rad vs {other_params.theta_rad:.2e}rad")

        if self.defect_type != other_params.defect_type:
            differences.append(f"瑕疵类型: {self.defect_type} vs {other_params.defect_type}")

        if self.scratch_type != other_params.scratch_type:
            differences.append(f"划痕类型: {self.scratch_type} vs {other_params.scratch_type}")

        if self.bump_count != other_params.bump_count:
            differences.append(f"突起/凹陷数量: {self.bump_count}个 vs {other_params.bump_count}个")

        if self.bump_position_mode != other_params.bump_position_mode:
            differences.append(f"位置模式: {self.bump_position_mode} vs {other_params.bump_position_mode}")

        return differences

    def get_scratch_depth_range(self):
        """获取划痕深度范围"""
        if self.scratch_type == "凹陷划痕":
            return (-0.2e-6, -0.05e-6)
        else:
            return (0.05e-6, 0.2e-6)

    def get_manual_scratch_depth_range(self):
        """获取手动划痕深度范围 - 扩大范围"""
        if self.manual_scratch_type == "凹陷划痕":
            return (-1e-6, 0)
        else:
            return (0, 1e-6)

    def calculate_scratch_effectiveness(self):
        """计算划痕效果评估"""
        if self.defect_type != "表面划痕":
            return 0.0

        base_score = 1.0
        type_factor = 1.2 if self.scratch_type == "凹陷划痕" else 1.0
        manual_factor = 1.3 if self.use_manual_scratches else 1.0

        return base_score * type_factor * manual_factor

    def get_scratches_info(self):
        """获取划痕信息"""
        if self.defect_type != "表面划痕":
            return "当前未选择表面划痕瑕疵"

        info = f"划痕类型: {self.scratch_type}\n"

        if self.use_manual_scratches:
            info += f"使用方式: 手动绘制\n"
            info += f"手动划痕类型: {self.manual_scratch_type}\n"
            depth_range = self.get_manual_scratch_depth_range()
            if self.manual_scratch_type == "凹陷划痕":
                info += f"凹陷深度范围: {abs(depth_range[0]) * 1e6:.1f}μm 到 {abs(depth_range[1]) * 1e6:.1f}μm\n"
            else:
                info += f"突起高度范围: {depth_range[0] * 1e6:.1f}μm 到 {depth_range[1] * 1e6:.1f}μm\n"
        else:
            info += f"使用方式: 手动绘制\n"
            depth_range = self.get_manual_scratch_depth_range()
            if self.scratch_type == "凹陷划痕":
                info += f"凹陷深度范围: {abs(depth_range[0]) * 1e6:.1f}μm 到 {abs(depth_range[1]) * 1e6:.1f}μm\n"
            else:
                info += f"突起高度范围: {depth_range[0] * 1e6:.1f}μm 到 {depth_range[1] * 1e6:.1f}μm\n"

        effectiveness = self.calculate_scratch_effectiveness()
        info += f"效果评估: {effectiveness:.1%}"

        return info

    def get_bump_effect_height_range(self):
        """获取小突起/小凹陷的实际效应高度范围"""
        min_effect_height = self.bump_height * 2.0
        max_effect_height = self.bump_height * 3.0
        return min_effect_height, max_effect_height

    def get_parameter_status(self):
        """获取参数状态信息"""
        status = {
            'basic_parameters': {
                'lambda_nm': self.lambda_nm,
                'd0_um': self.d0_um,
                'theta_rad': self.theta_rad,
                'n1': self.n1,
                'n2': self.n2,
                'n3': self.n3
            },
            'defect_parameters': {
                'defect_type': self.defect_type,
                'dust_density': self.dust_density,
                'scratch_type': self.scratch_type,
                'bump_count': self.bump_count,
                'bump_height': self.bump_height,
                'bump_position_mode': self.bump_position_mode
            },
            'drawing_parameters': {
                'use_manual_scratches': self.use_manual_scratches,
                'manual_scratch_type': self.manual_scratch_type,
                'brush_size': self.brush_size,
                'brush_depth': self.brush_depth,
                'eraser_mode': self.eraser_mode
            },
            'simulation_settings': {
                'simulation_size': self.simulation_size,
                'has_canvas_state': self.canvas_state is not None,
                'has_manual_mask': self.manual_scratch_mask is not None
            }
        }
        return status

    def is_parameter_changed(self, other_params):
        """检查参数是否发生变化"""
        if not isinstance(other_params, SimulationParameters):
            return True

        current_dict = self.export_parameters()
        other_dict = other_params.export_parameters()

        return current_dict != other_dict

    def get_parameter_changes(self, other_params):
        """获取参数变化的详细信息"""
        changes = []

        if self.lambda_nm != other_params.lambda_nm:
            changes.append(f"波长: {other_params.lambda_nm}nm → {self.lambda_nm}nm")

        if self.d0_um != other_params.d0_um:
            changes.append(f"劈尖截距: {other_params.d0_um}μm → {self.d0_um}μm")

        if abs(self.theta_rad - other_params.theta_rad) > 1e-10:
            changes.append(f"劈尖夹角: {other_params.theta_rad:.2e}rad → {self.theta_rad:.2e}rad")

        if self.defect_type != other_params.defect_type:
            changes.append(f"瑕疵类型: {other_params.defect_type} → {self.defect_type}")

        if self.scratch_type != other_params.scratch_type:
            changes.append(f"划痕类型: {other_params.scratch_type} → {self.scratch_type}")

        if self.bump_count != other_params.bump_count:
            changes.append(f"突起/凹陷数量: {other_params.bump_count}个 → {self.bump_count}个")

        return changes

    def validate_current_parameters(self):
        """验证当前参数的有效性"""
        errors = self.validate_parameters()
        warnings = []

        # 检查波长是否在可见光范围内
        if self.lambda_nm < 400 or self.lambda_nm > 700:
            warnings.append("波长不在典型可见光范围(400-700nm)内")

        # 检查折射率合理性
        if self.n2 <= self.n1 or self.n2 <= self.n3:
            warnings.append("薄膜折射率应大于上下板折射率以获得明显干涉效果")

        # 检查灰尘密度是否过高
        if self.dust_density > 20:
            warnings.append("灰尘密度较高，可能会显著影响干涉图样")

        # 检查突起/凹陷数量
        if self.bump_count > 5:
            warnings.append("突起/凹陷数量较多，可能会使干涉图样复杂化")

        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0
        }

    def get_simulation_preview_info(self):
        """获取仿真预览信息"""
        preview_info = {
            'defect_type': self.defect_type,
            'defect_description': self.get_defect_description(),
            'wavelength_info': f"{self.lambda_nm}nm ({self.get_wavelength_color()})",
            'interference_type': self.get_interference_type(),
            'expected_effect': self.get_expected_effect(),
            'simulation_complexity': self.get_simulation_complexity()
        }
        return preview_info

    def get_wavelength_color(self):
        """获取波长对应的颜色描述"""
        lamda = self.get_wavelength_meters()
        if 380e-9 <= lamda < 450e-9:
            return "紫色/蓝色"
        elif 450e-9 <= lamda < 495e-9:
            return "青色"
        elif 495e-9 <= lamda < 570e-9:
            return "绿色"
        elif 570e-9 <= lamda < 590e-9:
            return "黄色"
        elif 590e-9 <= lamda < 620e-9:
            return "橙色"
        elif 620e-9 <= lamda < 750e-9:
            return "红色"
        else:
            return "不可见光"

    def get_interference_type(self):
        """获取干涉类型描述"""
        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            return "半波损失干涉"
        else:
            return "正常干涉"

    def get_expected_effect(self):
        """获取预期效果描述"""
        if self.defect_type == "无瑕疵":
            return "清晰的等厚干涉条纹"
        elif self.defect_type == "灰尘污染":
            return "局部条纹扭曲和亮度变化"
        elif self.defect_type == "表面划痕":
            return "沿划痕方向的条纹畸变"
        elif self.defect_type in ["小突起", "小凹陷"]:
            return "局部条纹弯曲和间距变化"
        else:
            return "未知效果"

    def get_simulation_complexity(self):
        """获取仿真复杂度评估"""
        complexity = 1  # 基础复杂度

        if self.defect_type != "无瑕疵":
            complexity += 1

        if self.defect_type == "灰尘污染" and self.dust_density > 10:
            complexity += 1

        if self.defect_type in ["小突起", "小凹陷"] and self.bump_count > 3:
            complexity += 1

        if self.use_manual_scratches and self.manual_scratch_mask is not None:
            complexity += 2

        return min(complexity, 5)  # 最大复杂度为5

    def create_parameter_preset(self, preset_name):
        """创建参数预设"""
        preset = {
            'name': preset_name,
            'timestamp': np.datetime64('now'),
            'parameters': self.export_parameters(),
            'description': self.get_defect_description()
        }
        return preset

    def apply_parameter_preset(self, preset):
        """应用参数预设"""
        if isinstance(preset, dict) and 'parameters' in preset:
            return self.import_parameters(preset['parameters'])
        return False

    def get_parameter_ranges(self):
        """获取参数范围信息"""
        ranges = {
            'lambda_nm': {'min': 380, 'max': 760, 'step': 1, 'unit': 'nm'},
            'd0_um': {'min': 0.1, 'max': 50.0, 'step': 0.1, 'unit': 'μm'},
            'theta_rad': {'min': 0.1e-5, 'max': 20e-5, 'step': 0.1e-5, 'unit': '×10⁻⁵ rad'},
            'n1': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'unit': ''},
            'n2': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'unit': ''},
            'n3': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'unit': ''},
            'dust_density': {'min': 0, 'max': 30, 'step': 1, 'unit': '%'},
            'bump_count': {'min': 1, 'max': 10, 'step': 1, 'unit': '个'},
            'bump_height': {'min': 0.01, 'max': 1.0, 'step': 0.01, 'unit': 'μm'},
            'brush_size': {'min': 1, 'max': 20, 'step': 1, 'unit': '像素'},
            'brush_depth': {'min': 0, 'max': 100, 'step': 1, 'unit': '%'}
        }
        return ranges

    def get_recommended_presets(self):
        """获取推荐参数预设"""
        presets = {
            'red_light': {
                'name': '红光标准劈尖',
                'lambda_nm': 622,
                'd0_um': 10,
                'theta_rad': 5e-5,
                'n1': 1.5,
                'n2': 2.0,
                'n3': 1.5,
                'description': '标准红光劈尖干涉'
            },
            'green_light': {
                'name': '绿光精密测量',
                'lambda_nm': 532,
                'd0_um': 5,
                'theta_rad': 2e-5,
                'n1': 1.5,
                'n2': 1.7,
                'n3': 1.5,
                'description': '绿光高精度干涉测量'
            },
            'blue_light': {
                'name': '蓝光小角度',
                'lambda_nm': 450,
                'd0_um': 15,
                'theta_rad': 1e-5,
                'n1': 1.6,
                'n2': 2.4,
                'n3': 1.6,
                'description': '蓝光小角度干涉'
            }
        }
        return presets

    def __repr__(self):
        return f"SimulationParameters(λ={self.lambda_nm}nm, d0={self.d0_um}μm, θ={self.theta_rad:.2e}rad, defect={self.defect_type})"