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
        self.bump_height = 0.01
        # 横向有效半径(mm)，默认与旧版 0.08×视场宽度(30mm) 相当
        self.bump_radius_mm = 2.4
        self.bump_type = "小突起"
        self.bump_position_mode = "网格排列"  # 先定义这个属性
        self.fixed_bump_positions = [  # 然后使用它
            (0.005, 0.005), (0.010, 0.005), (0.005, 0.010)
        ]

        # 头发丝遮挡参数
        self.hair_count = 3

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
        self.bump_height = 0.01
        self.bump_radius_mm = 2.4
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

        # 修改点1：定义屏幕物理尺寸 (单位：毫米)
        Screen_length_x_mm = 30  # X轴长度改为30毫米
        Screen_length_y_mm = 15  # Y轴长度保持15毫米
        # 转换为米用于内部计算
        Screen_length_x = Screen_length_x_mm * 0.001
        Screen_length_y = Screen_length_y_mm * 0.001

        # 修改点2：分别生成X和Y方向的坐标网格
        x = np.linspace(0, Screen_length_x, self.simulation_size)  # X方向：0-0.030米
        y = np.linspace(0, Screen_length_y, self.simulation_size)  # Y方向：0-0.015米
        X, Y = np.meshgrid(x, y)  # 生成二维网格

        # 理想厚度 (公式基于X坐标)
        ideal_thickness = d0 + X * np.tan(theta)

        # 有缺陷厚度
        defect_thickness = ideal_thickness.copy()
        if self.defect_type == "头发丝遮挡":  # 新增
            defect_thickness = self.add_hair_defect(ideal_thickness, X, Y)
        elif self.defect_type == "灰尘污染":
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

        # 修改点1：定义与显示一致的物理计算域 (X: 30mm, Y: 15mm)
        screen_length_x = 0.030  # 30毫米 -> 0.030米
        screen_length_y = 0.015  # 15毫米 -> 0.015米

        # 修改点2：生成用于一维光强曲线的X坐标 (0-30mm)
        x_curve = np.linspace(0, screen_length_x, self.simulation_size)
        # 计算一维厚度分布
        thickness_curve = d0 + x_curve * np.tan(theta)

        # 计算一维光程差和相位差
        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            delta_curve = 2 * self.n2 * thickness_curve + lamda / 2
        else:
            delta_curve = 2 * self.n2 * thickness_curve

        phase_diff_curve = (2 * np.pi * delta_curve) / lamda
        # 计算一维光强分布曲线
        intensity_curve = 4 * self.I0 * (np.cos(phase_diff_curve / 2)) ** 2

        # 修改点3：生成用于二维干涉图样的网格 (30mm x 15mm)
        x_2d = np.linspace(0, screen_length_x, self.simulation_size)
        y_2d = np.linspace(0, screen_length_y, self.simulation_size)
        X_2d, Y_2d = np.meshgrid(x_2d, y_2d)  # 网格尺寸: simulation_size x simulation_size
        # 计算二维厚度分布 (厚度只随X变化)
        thickness_2d = d0 + X_2d * np.tan(theta)

        # 计算二维光程差和相位差
        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            delta_2d = 2 * self.n2 * thickness_2d + lamda / 2
        else:
            delta_2d = 2 * self.n2 * thickness_2d

        phase_diff_2d = (2 * np.pi * delta_2d) / lamda
        # 计算二维干涉图样
        pattern = 4 * self.I0 * (np.cos(phase_diff_2d / 2)) ** 2

        return intensity_curve, pattern

    def add_hair_defect(self, thickness, X, Y):
        """添加头发丝遮挡缺陷 - 使用OpenCV进行连续曲线绘制"""
        import cv2
        import numpy as np
        import random
        
        hair_thickness = thickness.copy()
        height, width = thickness.shape

        # 获取屏幕尺寸（单位：米）
        screen_length_x = 0.030  # 30mm
        screen_length_y = 0.015  # 15mm

        # 用于存储头发丝的三维形貌数据
        hair_topography = np.zeros((height, width))

        for hair_index in range(self.hair_count):
            diameter_mm = random.uniform(0.04, 0.2)  # 毫米, 稍粗一点以确保可见度
            diameter_m = diameter_mm * 0.001
            radius_m = diameter_m / 2

            length_mm = random.uniform(5.0, 20.0)
            length_m = length_mm * 0.001

            start_x = random.uniform(diameter_m, screen_length_x - diameter_m)
            start_y = random.uniform(diameter_m, screen_length_y - diameter_m)

            control_points = self._generate_curved_hair_path(
                start_x, start_y,
                length_m=length_m,
                screen_length_x=screen_length_x,
                screen_length_y=screen_length_y,
                min_radius=diameter_m / 2
            )

            curve_points = self._sample_bezier_curve(
                control_points,
                num_samples=max(50, int(length_m / 0.0001))
            )

            pts_px = []
            for px, py in curve_points:
                qx = px / screen_length_x * (width - 1)
                qy = py / screen_length_y * (height - 1)
                pts_px.append([int(qx), int(qy)])

            if len(pts_px) == 0: continue
            pts_arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
            
            mask_layer = np.zeros((height, width), dtype=np.uint8)
            px_size = screen_length_x / width
            # 保证哪怕网格很稀疏也能显示，至少像素线宽为2
            thickness_px = max(2, int(diameter_m / px_size))
            cv2.polylines(mask_layer, [pts_arr], isClosed=False, color=1, thickness=thickness_px, lineType=cv2.LINE_AA)
            mask = mask_layer > 0
            
            if np.any(mask):
                hair_thickness[mask] = 1e-3
                
                # 添加形貌随机变化
                topography_heights = diameter_mm * np.random.uniform(0.8, 1.2, size=np.sum(mask))
                hair_topography[mask] = np.maximum(hair_topography[mask], topography_heights)

                if not hasattr(self, 'hair_mask') or self.hair_mask is None:
                    self.hair_mask = np.zeros((height, width), dtype=bool)
                self.hair_mask[mask] = True

        hair_thickness += hair_topography * 1e-6
        return hair_thickness

    def _generate_curved_hair_path(self, start_x, start_y, length_m, screen_length_x, screen_length_y, min_radius):
        """生成随机弯曲的头发丝路径（贝塞尔曲线）"""
        # 随机初始角度
        initial_angle = random.uniform(0, 2 * np.pi)

        # 随机弯曲程度（0-1，0为直线，1为高度弯曲）
        curvature = random.uniform(0.3, 0.9)

        # 随机弯曲方向
        curve_direction = random.choice([-1, 1])

        # 控制点数量（1-3个控制点产生不同程度的弯曲）
        num_control_points = random.randint(1, 3)

        control_points = [(start_x, start_y)]

        # 生成控制点
        segment_length = length_m / (num_control_points + 1)
        current_x, current_y = start_x, start_y
        current_angle = initial_angle

        for i in range(num_control_points):
            # 添加随机角度偏移
            angle_offset = curve_direction * curvature * random.uniform(np.pi / 6, np.pi / 3)
            current_angle += angle_offset

            # 控制点位置
            control_x = current_x + segment_length * np.cos(current_angle)
            control_y = current_y + segment_length * np.sin(current_angle)

            # 确保在屏幕内
            control_x = max(min_radius, min(screen_length_x - min_radius, control_x))
            control_y = max(min_radius, min(screen_length_y - min_radius, control_y))

            control_points.append((control_x, control_y))
            current_x, current_y = control_x, control_y

        # 终点
        end_angle_offset = curve_direction * curvature * random.uniform(np.pi / 8, np.pi / 4)
        end_angle = current_angle + end_angle_offset

        end_x = current_x + segment_length * np.cos(end_angle)
        end_y = current_y + segment_length * np.sin(end_angle)

        # 确保在屏幕内
        end_x = max(min_radius, min(screen_length_x - min_radius, end_x))
        end_y = max(min_radius, min(screen_length_y - min_radius, end_y))

        control_points.append((end_x, end_y))

        return control_points

    def _sample_bezier_curve(self, control_points, num_samples=50):
        """沿着贝塞尔曲线采样点"""
        if len(control_points) < 2:
            return control_points

        # 如果只有2个点，返回直线
        if len(control_points) == 2:
            t_values = np.linspace(0, 1, num_samples)
            points = []
            for t in t_values:
                x = (1 - t) * control_points[0][0] + t * control_points[1][0]
                y = (1 - t) * control_points[0][1] + t * control_points[1][1]
                points.append((x, y))
            return points

        # 二次贝塞尔曲线
        elif len(control_points) == 3:
            t_values = np.linspace(0, 1, num_samples)
            points = []
            for t in t_values:
                x = (1 - t) ** 2 * control_points[0][0] + \
                    2 * (1 - t) * t * control_points[1][0] + \
                    t ** 2 * control_points[2][0]
                y = (1 - t) ** 2 * control_points[0][1] + \
                    2 * (1 - t) * t * control_points[1][1] + \
                    t ** 2 * control_points[2][1]
                points.append((x, y))
            return points

        # 三次贝塞尔曲线
        elif len(control_points) == 4:
            t_values = np.linspace(0, 1, num_samples)
            points = []
            for t in t_values:
                x = (1 - t) ** 3 * control_points[0][0] + \
                    3 * (1 - t) ** 2 * t * control_points[1][0] + \
                    3 * (1 - t) * t ** 2 * control_points[2][0] + \
                    t ** 3 * control_points[3][0]
                y = (1 - t) ** 3 * control_points[0][1] + \
                    3 * (1 - t) ** 2 * t * control_points[1][1] + \
                    3 * (1 - t) * t ** 2 * control_points[2][1] + \
                    t ** 3 * control_points[3][1]
                points.append((x, y))
            return points

        return control_points

    def _point_to_line_distance(self, x, y, x1, y1, x2, y2):
        """计算点到线段的距离"""
        # 线段长度平方
        length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

        if length_sq == 0:
            # 点退化
            return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        # 计算投影参数
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / length_sq))

        # 投影点
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)

        # 返回距离
        return np.sqrt((x - projection_x) ** 2 + (y - projection_y) ** 2)

    def calculate_defect_interference(self):
        """计算有瑕疵干涉图样 - 专门处理头发丝遮挡"""
        lamda = self.get_wavelength_meters()
        d0 = self.get_d0_meters()
        theta = self.get_theta_radians()

        # 定义与显示和理想计算一致的物理计算域
        screen_length_x = 0.030  # 30毫米 -> 0.030米
        screen_length_y = 0.015  # 15毫米 -> 0.015米

        # 生成覆盖新范围的二维计算网格
        x = np.linspace(0, screen_length_x, self.simulation_size)
        y = np.linspace(0, screen_length_y, self.simulation_size)
        X, Y = np.meshgrid(x, y)  # 关键修改：网格范围现在是0.030m x 0.015m

        # 计算无缺陷时的基础厚度分布
        base_thickness = d0 + X * np.tan(theta)
        # 初始化有缺陷的厚度，开始时与基础厚度相同
        thickness = base_thickness.copy()

        # 根据选择的瑕疵类型，在基础厚度上添加缺陷
        if self.defect_type == "无瑕疵":
            # 厚度保持不变
            pass
        elif self.defect_type == "灰尘污染":
            thickness = self.add_dust_defect(thickness, X, Y)
        elif self.defect_type == "表面划痕":
            thickness = self.add_scratch_defect(thickness, X, Y)
        elif self.defect_type == "小突起":
            thickness = self.add_bump_defect(thickness, X, Y, bump_type="突起")
        elif self.defect_type == "小凹陷":
            thickness = self.add_bump_defect(thickness, X, Y, bump_type="凹陷")
        elif self.defect_type == "头发丝遮挡":
            thickness = self.add_hair_defect(thickness, X, Y)

        # 计算有缺陷时的光程差
        if (self.n2 > self.n1 and self.n2 > self.n3) or (self.n2 < self.n1 and self.n2 < self.n3):
            delta = 2 * self.n2 * thickness + lamda / 2
        else:
            delta = 2 * self.n2 * thickness

        # 计算相位差
        phase_diff = (2 * np.pi * delta) / lamda

        # 计算干涉图样
        pattern = 4 * self.I0 * (np.cos(phase_diff / 2)) ** 2

        # ==== 关键修改：对头发丝遮挡特殊处理 ====
        if self.defect_type == "头发丝遮挡":
            # 获取头发丝掩码
            if hasattr(self, 'hair_mask') and self.hair_mask is not None:
                # 在头发丝位置设置光强为0
                pattern[self.hair_mask] = 0
            else:
                # 如果没有掩码，通过厚度特征识别头发丝位置
                # 头发丝位置通常有特殊的厚度值
                hair_thickness_threshold = 1e-4  # 0.1毫米
                hair_mask = thickness > hair_thickness_threshold
                pattern[hair_mask] = 0

        # 提取中心行（Y轴中间位置）的光强分布作为一维曲线
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
                            base_depth = 1.0e-6  # 基准深度：-1.0微米
                            depth = base_depth * depth_factor  # 直接乘以深度系数
                        else:
                            base_depth = -1.0e-6  # 基准深度：+1.0微米
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
        """添加小突起/小凹陷缺陷。

        在劈尖平面内按圆形区域施加：点到圆心距离 ≤ 半径（物理坐标 X、Y 下欧氏距离），
        这样在干涉图样（等比例 mm 坐标）上显示为圆，而非椭圆。
        """
        bump_thickness = thickness.copy()
        height, width = thickness.shape

        screen_length_x = 0.030
        screen_length_y = 0.015

        base_depth = self.bump_height * 1e-6

        if bump_type == "突起":
            depth_sign = -1.0
        else:
            depth_sign = 1.0

        base_depth_m = base_depth * depth_sign

        positions = self._generate_bump_positions(screen_length_x, screen_length_y)

        # 用户可调半径（米），并限制在视场内以免数值过大
        radius_m = self.bump_radius_mm * 1e-3
        max_radius_m = min(screen_length_x, screen_length_y) * 0.48
        radius = float(np.clip(radius_m, 0.05e-3, max_radius_m))

        for center_x, center_y in positions[:self.bump_count]:

            for i in range(height):
                for j in range(width):
                    x = X[i, j]
                    y = Y[i, j]
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                    if dist <= radius:
                        normalized_dist = dist / radius
                        depth_coefficient = 1.0 - normalized_dist
                        edge_smooth_start = 0.7
                        if normalized_dist > edge_smooth_start:
                            t = (normalized_dist - edge_smooth_start) / (1.0 - edge_smooth_start)
                            depth_coefficient *= (1 - t ** 2)
                        depth = base_depth_m * depth_coefficient
                        bump_thickness[i, j] += depth

        return bump_thickness

    def _generate_bump_positions(self, screen_length_x, screen_length_y):  # 修改函数签名，接受两个参数
        if self.bump_position_mode == "随机分布":
            return self._generate_random_positions(screen_length_x, screen_length_y)  # 也需要修改，见下一步
        elif self.bump_position_mode == "网格排列":
            # 修正：传递两个参数
            return self._generate_grid_positions(screen_length_x, screen_length_y)
        elif self.bump_position_mode == "固定位置":
            return self.fixed_bump_positions
        else:
            # 修正：传递两个参数
            return self._generate_grid_positions(screen_length_x, screen_length_y)

    def _generate_random_positions(self, screen_length_x, screen_length_y):  # 修改函数签名
        positions = []
        for _ in range(max(self.bump_count, 10)):
            # 分别在X和Y方向的有效区域内随机生成位置
            center_x = random.uniform(0.1 * screen_length_x, 0.9 * screen_length_x)
            center_y = random.uniform(0.1 * screen_length_y, 0.9 * screen_length_y)
            positions.append((center_x, center_y))
        return positions

    def _generate_grid_positions(self, screen_length_x, screen_length_y):  # 注意：现在传入两个参数
        """生成网格排列的位置 - 增加行间间距版本"""
        positions = []
        grid_size = int(np.ceil(np.sqrt(self.bump_count)))

        # 行间距增加系数，使行之间分开更多
        row_spacing_factor = 1.3

        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= self.bump_count:
                    break

                # X坐标：在0-screen_length_x范围内均匀分布
                center_x = (j + 1) * screen_length_x / (grid_size + 1)
                # Y坐标：在0-screen_length_y范围内均匀分布，并增加行间距
                base_y_spacing = screen_length_y / (grid_size + 1)
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
            "小突起": f"表面小突起，数量{self.bump_count}个，高度{self.bump_height}μm，半径{self.bump_radius_mm}mm，位置模式：{self.bump_position_mode}",
            "小凹陷": f"表面小凹陷，数量{self.bump_count}个，深度{self.bump_height}μm，半径{self.bump_radius_mm}mm，位置模式：{self.bump_position_mode}",
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
        self.bump_radius_mm = control_panel.bump_radius_spin.value()
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
            summary += f"        横向半径: {self.bump_radius_mm}mm\n"
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
        """突起/凹陷横向有效半径（米），与 bump_radius_mm 一致并限制在视场内"""
        screen_length_x = 0.030
        screen_length_y = 0.015
        radius_m = self.bump_radius_mm * 1e-3
        max_radius_m = min(screen_length_x, screen_length_y) * 0.48
        return float(np.clip(radius_m, 0.05e-3, max_radius_m))

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
            'bump_radius_mm': self.bump_radius_mm,
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
            self.bump_height = params_dict.get('bump_height', 0.01)
            self.bump_radius_mm = params_dict.get('bump_radius_mm', 2.4)
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
                'bump_radius_mm': self.bump_radius_mm,
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
            'bump_radius_mm': {'min': 0.2, 'max': 7.0, 'step': 0.1, 'unit': 'mm'},
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