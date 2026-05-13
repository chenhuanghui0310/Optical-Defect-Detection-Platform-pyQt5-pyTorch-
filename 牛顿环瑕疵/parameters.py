import numpy as np
import random


class SimulationParameters:
    """仿真参数管理类"""

    def __init__(self):
        # 基本参数
        self.lambda_nm = 589
        self.d0_um = 5
        self.R_mm = 1000.0
        self.n = 1
        self.n1 = self.n
        self.n2 = self.n
        self.n3 = self.n
        self.I0 = 1.0
        self.screen_length_x = 0.006
        self.screen_length_y = 0.006

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
        self.fixed_bump_positions = [
            (-0.001, -0.001), (0.001, -0.001), (0.0, 0.001)
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

        self.simulation_size = 500
        self.canvas_state = None

    def reset(self):
        """重置所有参数"""
        self.lambda_nm = 589
        self.d0_um = 5
        self.R_mm = 1000.0
        self.n = 1
        self.n1 = self.n
        self.n2 = self.n
        self.n3 = self.n
        self.screen_length_x = 0.006
        self.screen_length_y = 0.006
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

    def get_R_meters(self):
        return self.R_mm / 1000.0

    def get_refractive_index(self):
        return self.n

    def _compute_reference_pattern(self, wavelength, radius, refractive_index, gap, X, Y):
        center_x = 0.0
        center_y = 0.0
        dx = X - center_x
        dy = Y - center_y

        # --- 依据图片物理公式更新 ---
        # 移除zoom_factor，使用实际尺寸
        r_squared = dx ** 2 + dy ** 2

        # 理想平凸透镜厚度: 2d = r^2 / R  =>  d = r^2 / (2R)
        d = r_squared / (2 * radius)

        # 截距 e (调用方传进来的 gap 已经包含基础厚度 d0 以及瑕疵高度 h)
        e = gap

        # 光程差 δ = 2nd + 2ne + λ/2
        # 当有瑕疵点 h 时 (e = d0 + h)，该公式与图片中 δ = 2n(d+h) + λ/2 完全相符
        delta = 2 * refractive_index * d + 2 * refractive_index * e + wavelength / 2.0

        # 光强公式: I = 4I0 * cos²(πδ/λ)
        # 为渲染框架适配 0~1 范围，I0归一化，使用 cos² 取值
        intensity = np.cos(np.pi * delta / wavelength) ** 2

        # 保持全视场显示
        background_level = 0.05
        contrast = 0.95
        intensity = background_level + contrast * intensity

        return np.clip(intensity, 0.0, 1.0)

    def get_thickness_data(self):
        """获取理想和有缺陷的厚度数据"""
        lamda = self.get_wavelength_meters()
        d0 = self.get_d0_meters()
        R_m = self.get_R_meters()

        x = np.linspace(-self.screen_length_x / 2, self.screen_length_x / 2, self.simulation_size)
        y = np.linspace(-self.screen_length_y / 2, self.screen_length_y / 2, self.simulation_size)
        X, Y = np.meshgrid(x, y)  # 生成二维网格

        # 理想厚度 (公式基于X坐标)
        ideal_thickness = d0 + (X**2 + Y**2) / (2 * R_m)

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
        R_m = self.get_R_meters()
        refractive_index = self.get_refractive_index()

        x_2d = np.linspace(-self.screen_length_x / 2, self.screen_length_x / 2, self.simulation_size)
        y_2d = np.linspace(-self.screen_length_y / 2, self.screen_length_y / 2, self.simulation_size)
        X_2d, Y_2d = np.meshgrid(x_2d, y_2d)  # 网格尺寸: simulation_size x simulation_size

        pattern = self._compute_reference_pattern(lamda, R_m, refractive_index, d0, X_2d, Y_2d)

        # 中心行一维光强分布
        intensity_curve = pattern[self.simulation_size // 2, :]

        return intensity_curve, pattern

    def add_hair_defect(self, thickness, X, Y):
        """添加头发丝遮挡缺陷 - 带随机弯曲效果的版本"""
        hair_thickness = thickness.copy()
        height, width = thickness.shape

        # 每次生成前重置缓存的掩码，避免多次调整数量时头发丝残影累加
        self.hair_mask = np.zeros((height, width), dtype=bool)

        # 获取屏幕尺寸（单位：米）
        screen_length_x = self.screen_length_x
        screen_length_y = self.screen_length_y

        # 用于存储头发丝的三维形貌数据
        hair_topography = np.zeros((height, width))

        # 将每根头发丝分配到相对独立的锚点，避免多根头发在视场内重叠成一根
        anchor_positions = self._generate_hair_anchor_positions(screen_length_x, screen_length_y, self.hair_count)

        for hair_index, (anchor_x, anchor_y) in enumerate(anchor_positions):
            # 随机生成头发丝直径：0.02-0.2毫米
            diameter_mm = random.uniform(0.02, 0.2)  # 毫米
            diameter_m = diameter_mm * 0.001  # 转换为米
            radius_m = diameter_m / 2

            # 缩短单根头发丝长度，确保输入几根就能清楚看到几根
            length_mm = random.uniform(2.5, 4.0)  # 毫米
            length_m = length_mm * 0.001  # 转换为米

            # 在锚点附近轻微抖动，避免机械化，同时保持彼此分离
            jitter_x = random.uniform(-0.2e-3, 0.2e-3)
            jitter_y = random.uniform(-0.2e-3, 0.2e-3)
            start_x = anchor_x + jitter_x
            start_y = anchor_y + jitter_y

            # 保证起点在视场内
            start_x = max(-screen_length_x / 2 + radius_m, min(screen_length_x / 2 - radius_m, start_x))
            start_y = max(-screen_length_y / 2 + radius_m, min(screen_length_y / 2 - radius_m, start_y))

            # ==== 关键修改：生成随机弯曲路径 ====
            # 使用贝塞尔曲线生成弯曲路径
            control_points = self._generate_curved_hair_path(
                start_x, start_y,
                length_m=length_m,
                screen_length_x=screen_length_x,
                screen_length_y=screen_length_y,
                min_radius=diameter_m / 2
            )

            # 沿着贝塞尔曲线采样点
            curve_points = self._sample_bezier_curve(
                control_points,
                num_samples=max(18, int(length_m / (radius_m * 0.9)))  # 采样点密度
            )

            # 沿着弯曲路径绘制头发丝
            for i in range(len(curve_points)):
                center_x, center_y = curve_points[i]

                # 计算每个网格点到头发丝中心的距离矩阵
                dist_matrix = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

                # 找到在半径内的像素
                mask = dist_matrix <= radius_m

                if np.any(mask):
                    # 1. 在干涉图上：设置为纯黑色遮挡
                    hair_thickness[mask] = 1e-3

                    # 2. 在三维形貌图上：突起高度为头发直径
                    # 计算高斯剖面，使边缘更平滑
                    normalized_dist = dist_matrix[mask] / radius_m
                    gaussian_factor = np.exp(-(normalized_dist ** 2) * 4)  # 更陡峭的高斯

                    # 添加随机噪点，使头发丝表面不完美
                    if random.random() < 0.3:  # 30%的像素添加噪点
                        noise = random.uniform(0.8, 1.2)
                        gaussian_factor = gaussian_factor * noise

                    height_values = diameter_mm * gaussian_factor

                    # 更新形貌数据，取最大值（防止过度覆盖）
                    current_heights = hair_topography[mask]
                    hair_topography[mask] = np.maximum(current_heights, height_values)

                    # 3. 记录头发丝掩码（用于干涉图黑色处理）
                    self.hair_mask[mask] = True

        # 将头发丝突起高度叠加到厚度上（用于三维显示）
        # 转换为米：1微米 = 1e-6米
        hair_thickness += hair_topography * 1e-6

        return hair_thickness

    def _generate_hair_anchor_positions(self, screen_length_x, screen_length_y, hair_count):
        """为每根头发丝生成相对分离的锚点位置。"""
        if hair_count <= 0:
            return []

        if hair_count == 1:
            return [(0.0, 0.0)]

        cols = int(np.ceil(np.sqrt(hair_count)))
        rows = int(np.ceil(hair_count / cols))

        x_positions = np.linspace(-0.35 * screen_length_x, 0.35 * screen_length_x, cols)
        y_positions = np.linspace(-0.35 * screen_length_y, 0.35 * screen_length_y, rows)

        anchors = []
        for row in range(rows):
            for col in range(cols):
                if len(anchors) >= hair_count:
                    break
                anchors.append((x_positions[col], y_positions[row]))

        return anchors

    def _generate_curved_hair_path(self, start_x, start_y, length_m, screen_length_x, screen_length_y, min_radius):
        """生成随机弯曲的头发丝路径（贝塞尔曲线）"""
        # 随机初始角度
        initial_angle = random.uniform(0, 2 * np.pi)

        # 随机弯曲程度（0-1，0为直线，1为高度弯曲）
        curvature = random.uniform(0.15, 0.45)

        # 随机弯曲方向
        curve_direction = random.choice([-1, 1])

        # 控制点数量（1-3个控制点产生不同程度的弯曲）
        num_control_points = random.randint(1, 2)

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

            # 确保在屏幕内（以原点为中心）
            control_x = max(-screen_length_x / 2 + min_radius, min(screen_length_x / 2 - min_radius, control_x))
            control_y = max(-screen_length_y / 2 + min_radius, min(screen_length_y / 2 - min_radius, control_y))

            control_points.append((control_x, control_y))
            current_x, current_y = control_x, control_y

        # 终点
        end_angle_offset = curve_direction * curvature * random.uniform(np.pi / 8, np.pi / 4)
        end_angle = current_angle + end_angle_offset

        end_x = current_x + segment_length * np.cos(end_angle)
        end_y = current_y + segment_length * np.sin(end_angle)

        # 确保在屏幕内（以原点为中心）
        end_x = max(-screen_length_x / 2 + min_radius, min(screen_length_x / 2 - min_radius, end_x))
        end_y = max(-screen_length_y / 2 + min_radius, min(screen_length_y / 2 - min_radius, end_y))

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
        R_m = self.get_R_meters()
        refractive_index = self.get_refractive_index()

        # 定义与显示和理想计算一致的物理计算域
        screen_length_x = self.screen_length_x
        screen_length_y = self.screen_length_y

        # 生成覆盖新范围的二维计算网格
        x = np.linspace(-screen_length_x/2, screen_length_x/2, self.simulation_size)
        y = np.linspace(-screen_length_y/2, screen_length_y/2, self.simulation_size)
        X, Y = np.meshgrid(x, y)  # 关键修改：网格范围现在是0.006m x 0.006m

        # 计算无缺陷时的基础厚度分布
        base_thickness = d0 + (X**2 + Y**2) / (2 * R_m)
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

        # 基于基准牛顿环模型计算干涉图样：
        # 使用曲率项 + 实际距离e项，瑕疵引起的厚度变化叠加到e项上。
        gap_perturbation = thickness - base_thickness
        local_gap = d0 + gap_perturbation
        pattern = self._compute_reference_pattern(lamda, R_m, refractive_index, local_gap, X, Y)

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

        return intensity_curve, pattern, gap_perturbation

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

        在牛顿环平面内按圆形区域施加：点到圆心距离 ≤ 半径（物理坐标 X、Y 下欧氏距离），
        这样在干涉图样（等比例 mm 坐标）上显示为圆，而非椭圆。
        """
        bump_thickness = thickness.copy()
        height, width = thickness.shape

        screen_length_x = self.screen_length_x
        screen_length_y = self.screen_length_y

        # 移除额外的可视化增益和物理限制，以保证与其他瑕疵（如表面划痕）的干涉条纹表现一致。
        base_depth_raw = self.bump_height * 1e-6
        base_depth = base_depth_raw

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
            # 使用余弦钟形轮廓使高度变化更均匀地分布在整个横向半径内
            # 这使得深度增加时，整个瑕疵区域的干涉条纹弯曲程度都会明显增加
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            
            mask = dist <= radius
            
            # 余弦钟形函数：从中心的 1 平滑过渡到边缘的 0
            profile = (1.0 + np.cos(np.pi * dist[mask] / radius)) / 2.0
            
            # 加上缺陷的形貌深度
            bump_thickness[mask] += base_depth_m * profile

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
            center_x = random.uniform(-0.4 * screen_length_x, 0.4 * screen_length_x)
            center_y = random.uniform(-0.4 * screen_length_y, 0.4 * screen_length_y)
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

                # X坐标：在以原点为中心的范围内均匀分布
                center_x = -screen_length_x / 2 + (j + 1) * screen_length_x / (grid_size + 1)
                # Y坐标：在以原点为中心的范围内均匀分布，并增加行间距
                base_y_spacing = screen_length_y / (grid_size + 1)
                extra_spacing = (i) * (base_y_spacing * (row_spacing_factor - 1))
                center_y = -screen_length_y / 2 + (i + 1) * base_y_spacing + extra_spacing
                center_y = np.clip(center_y, -0.45 * screen_length_y, 0.45 * screen_length_y)

                positions.append((center_x, center_y))

        return positions

    def get_defect_description(self):
        """获取瑕疵描述"""
        descriptions = {
            "无瑕疵": "无任何瑕疵的理想牛顿环干涉",
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
        self.R_mm = control_panel.R_spin.value()
        self.n = control_panel.n_spin.value()
        self.n1 = self.n
        self.n2 = self.n
        self.n3 = self.n
        self.defect_type = control_panel.defect_combo.currentText()
        self.dust_density = control_panel.dust_spin.value()
        self.scratch_type = control_panel.scratch_type_combo.currentText()
        self.bump_count = control_panel.bump_spin.value()
        self.bump_height = control_panel.bump_height_spin.value()
        self.bump_radius_mm = control_panel.bump_radius_spin.value()
        self.bump_position_mode = control_panel.bump_position_combo.currentText()
        self.manual_scratch_type = control_panel.scratch_type_combo.currentText()
        self.brush_size = control_panel.brush_size_spin.value()
        self.brush_depth = control_panel.depth_slider.value() / 100.0
        self.eraser_mode = control_panel.eraser_button.isChecked()

    def get_parameter_summary(self):
        """获取参数摘要"""
        summary = f"""
        仿真参数摘要:
        波长: {self.lambda_nm} nm
        实际距离e: {self.d0_um} μm
        曲率半径R: {self.R_mm:.1f} mm
        折射率n: {self.n}
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

        if self.lambda_nm < 380 or self.lambda_nm > 750:
            errors.append("波长应在380-750nm范围内")

        if self.d0_um < 1 or self.d0_um > 15:
            errors.append("实际距离e应在1-15μm范围内")

        if self.n < 1.0 or self.n > 5.0:
            errors.append("折射率n应在1.0-5.0范围内")

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
            (-0.001, -0.001),
            (0.001, -0.001),
            (0.0, 0.001),
        ]

    def get_bump_radius(self):
        """突起/凹陷横向有效半径（米），与 bump_radius_mm 一致并限制在视场内"""
        screen_length_x = self.screen_length_x
        screen_length_y = self.screen_length_y
        radius_m = self.bump_radius_mm * 1e-3
        max_radius_m = min(screen_length_x, screen_length_y) * 0.48
        return float(np.clip(radius_m, 0.05e-3, max_radius_m))

    def export_parameters(self):
        """导出参数为字典"""
        return {
            'lambda_nm': self.lambda_nm,
            'd0_um': self.d0_um,
            'R_mm': self.R_mm,
            'n': self.n,
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
            self.lambda_nm = params_dict.get('lambda_nm', 589)
            self.d0_um = params_dict.get('d0_um', 5)
            self.R_mm = params_dict.get('R_mm', 1000.0)
            self.n = params_dict.get('n', params_dict.get('n2', 1.0))
            self.n1 = self.n
            self.n2 = self.n
            self.n3 = self.n
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
                (-0.001, -0.001), (0.001, -0.001), (0.0, 0.001)
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
            differences.append(f"中心间隙: {self.d0_um}μm vs {other_params.d0_um}μm")

        if abs(self.R_mm - other_params.R_mm) > 0.1:
            differences.append(f"透镜曲率半径: {self.R_mm:.1f}mm vs {other_params.R_mm:.1f}mm")

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
        base_depth_raw = self.bump_height
        visibility_gain = 12.0
        effective_um = np.clip(base_depth_raw * visibility_gain, 0.02, 2.0)
        min_effect_height = effective_um * 0.85
        max_effect_height = effective_um * 1.0
        return min_effect_height, max_effect_height

    def get_parameter_status(self):
        """获取参数状态信息"""
        status = {
            'basic_parameters': {
                'lambda_nm': self.lambda_nm,
                'd0_um': self.d0_um,
                'R_mm': self.R_mm,
                'n': self.n
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
            changes.append(f"中心间隙: {other_params.d0_um}μm → {self.d0_um}μm")

        if abs(self.R_mm - other_params.R_mm) > 0.1:
            changes.append(f"透镜曲率半径: {other_params.R_mm:.1f}mm → {self.R_mm:.1f}mm")

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
        if self.n < 1.1:
            warnings.append("折射率n较低，干涉条纹对比度可能偏弱")

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
        return "牛顿环反射干涉"

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
            'lambda_nm': {'min': 380, 'max': 750, 'step': 1, 'unit': 'nm'},
            'd0_um': {'min': 1.0, 'max': 15.0, 'step': 1.0, 'unit': 'μm'},
            'R_mm': {'min': 1000, 'max': 5000, 'step': 100, 'unit': ' mm'},
            'n': {'min': 1.0, 'max': 5.0, 'step': 0.1, 'unit': ''},
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
                'name': '红光标准牛顿环',
                'lambda_nm': 622,
                'd0_um': 5,
                'R_mm': 1000.0,
                'n': 1.0,
                'description': '标准红光牛顿环干涉'
            },
            'green_light': {
                'name': '绿光精密测量',
                'lambda_nm': 532,
                'd0_um': 5,
                'R_mm': 1000.0,
                'n': 1.0,
                'description': '绿光高精度干涉测量'
            },
            'blue_light': {
                'name': '蓝光小角度',
                'lambda_nm': 450,
                'd0_um': 8,
                'R_mm': 1000.0,
                'n': 1.0,
                'description': '蓝光小角度干涉'
            }
        }
        return presets

    def __repr__(self):
        return f"SimulationParameters(λ={self.lambda_nm}nm, d0={self.d0_um}μm, R={self.R_mm:.1f}mm, defect={self.defect_type})"