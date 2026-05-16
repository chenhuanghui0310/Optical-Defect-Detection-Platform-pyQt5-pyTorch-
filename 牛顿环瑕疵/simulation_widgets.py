import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTabWidget, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QTimer

from drawing_canvas import DrawingCanvas


def wavelength_to_rgb(wavelength):
    """将波长转换为RGB颜色（0-1），兼容米和纳米输入。"""
    wavelength_nm = wavelength * 1e9 if wavelength < 1e-6 else wavelength

    if 380 <= wavelength_nm <= 760:
        if wavelength_nm < 440:
            r = -(wavelength_nm - 440) / (440 - 380)
            g = 0
            b = 1
        elif wavelength_nm < 490:
            r = 0
            g = (wavelength_nm - 440) / (490 - 440)
            b = 1
        elif wavelength_nm < 510:
            r = 0
            g = 1
            b = -(wavelength_nm - 510) / (510 - 490)
        elif wavelength_nm < 580:
            r = (wavelength_nm - 510) / (580 - 510)
            g = 1
            b = 0
        elif wavelength_nm < 645:
            r = 1
            g = -(wavelength_nm - 645) / (645 - 580)
            b = 0
        else:
            r = 1
            g = 0
            b = 0
        return (r, g, b)

    return (0.8, 0.8, 0.8)


class SpectrumPlot(FigureCanvas):
    """光谱图控件 - 固定尺寸版本"""

    def __init__(self, parent=None):
        # 创建单个干涉图子图 - 固定尺寸
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 3.5))  # 图案部分 600x300
        self.fig.set_facecolor('#FFFFFF')

        super().__init__(self.fig)
        # 设置固定尺寸策略
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(800, 450)  # 固定尺寸：600x300像素（图案部分）

    def get_color_mapping(self, wavelength):
        """根据波长获取颜色映射"""
        wavelength_color = wavelength_to_rgb(wavelength)

        colors = [(0, 'black'), (1, wavelength_color)]
        return LinearSegmentedColormap.from_list('custom_cmap', colors)

    def update_plot(self, pattern, title, lamda, is_defect=False):
        """更新单个干涉图像"""
        # 缓存数据，供 save_figure 以训练集一致格式保存
        self._last_pattern = pattern
        self._last_title = title
        self._last_lamda = lamda
        # 清除子图
        self.ax.clear()

        # 获取颜色映射
        cmap = self.get_color_mapping(lamda)

        # 固定字体大小
        title_fontsize = 20
        label_fontsize = 8

        # 显示干涉图像
        self.ax.imshow(pattern, extent=[-3, 3, -3, 3], cmap=cmap, origin='lower', aspect='equal')
        self.ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
        self.ax.set_xlabel('x (mm)', fontsize=label_fontsize)
        self.ax.set_ylabel('y (mm)', fontsize=label_fontsize)

        # 设置坐标轴刻度：X轴每3mm一格（0,3,6,...,30），Y轴每3mm一格（0,3,6,...,15）
        # 与三维缺陷形貌图的刻度设置保持一致
        xticks = np.linspace(-3, 3, 7)
        yticks = np.linspace(-3, 3, 7)
        self.ax.set_xticks(xticks)
        self.ax.set_yticks(yticks)

        # 调整布局，减少边距以突出图案部分
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def save_figure(self, filename):
        """保存图像到文件。
        使用与训练数据集一致的参数（figsize=(5,5), dpi=150，无标题无刻度），
        确保 YOLO 模型能正确识别平台保存的图片。
        """
        import matplotlib.pyplot as _plt
        import numpy as _np

        if hasattr(self, '_last_pattern') and self._last_pattern is not None:
            cmap = self.get_color_mapping(self._last_lamda)
            # TODO 2.1：dpi=150，与训练数据 _save_interference_image(dpi=150) 一致
            fig2, ax2 = _plt.subplots(figsize=(5, 5), dpi=150)
            fig2.set_facecolor('#FFFFFF')
            ax2.imshow(self._last_pattern, extent=[-3, 3, -3, 3],
                       cmap=cmap, origin='lower', aspect='equal')
            # 与训练数据集保持一致：无标题、无坐标轴标签、无刻度数值
            ax2.set_xticks([])
            ax2.set_yticks([])
            _plt.tight_layout(pad=0)
            fig2.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            _plt.close(fig2)
        else:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')


class Intensity3DPlot(FigureCanvas):
    """三维形貌分布图控件"""

    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(9, 4.5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.set_facecolor('#FFFFFF')

        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(800, 500)
        self.ax.view_init(elev=28, azim=-45)

    def get_color_mapping(self, wavelength):
        color = wavelength_to_rgb(wavelength)
        return LinearSegmentedColormap.from_list('intensity_cmap', [(0, 0, 0), color], N=256)

    def update_plot(self, pattern, title, lamda, defect_type=""):
        self.ax.clear()

        # 将形貌数据转换为微米(um) 并反转Z轴
        topography_um = -pattern * 1e6

        if defect_type in ["小突起", "小凹陷"]:
            # 已经无需反转Z轴刻度，也去除了原本粗暴的除以12
            pass

        step = max(1, topography_um.shape[0] // 80)

        x = np.linspace(-3, 3, topography_um.shape[1])
        y = np.linspace(-3, 3, topography_um.shape[0])
        X, Y = np.meshgrid(x, y)

        X_plot = X[::step, ::step]
        Y_plot = Y[::step, ::step]
        Z_plot = topography_um[::step, ::step]

        cmap = self.get_color_mapping(lamda)
        self.ax.plot_surface(X_plot, Y_plot, Z_plot, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)

        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('x (mm)', fontsize=10, labelpad=8)
        self.ax.set_ylabel('y (mm)', fontsize=10, labelpad=8)
        self.ax.set_zlabel('形貌高度 (um)', fontsize=10, labelpad=8)

        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        
        # 动态设置Z轴范围
        z_min, z_max = np.min(Z_plot), np.max(Z_plot)
        if abs(z_max - z_min) < 1e-4:
            z_min, z_max = -1.0, 1.0 # 默认范围如果没有瑕疵
        else:
            padding = (z_max - z_min) * 0.1
            z_min -= padding
            z_max += padding
        self.ax.set_zlim(z_min, z_max)
        
        self.ax.set_xticks(np.linspace(-3, 3, 7))
        self.ax.set_yticks(np.linspace(-3, 3, 7))
        self.ax.grid(True, alpha=0.3)
        self.ax.view_init(elev=28, azim=-45)

        self.fig.tight_layout(pad=1.0)
        self.draw()

    def save_figure(self, filename):
        """保存图像到文件"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white',
                         transparent=False)


class DefectTopography3D(FigureCanvas):
    """缺陷三维形貌图控件 - 增大尺寸版本"""

    def __init__(self, parent=None):
        # 创建三维图形 - 增大尺寸
        self.fig = plt.figure(figsize=(9, 4.5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.set_facecolor('#FFFFFF')

        super().__init__(self.fig)
        # 设置固定尺寸策略
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(800, 500)

        # 设置默认视角
        self.ax.view_init(elev=30, azim=-45)

        # 用于存储缺陷形貌数据
        self.defect_topography = None
        self.X = None
        self.Y = None
        self.scale_factor = 1e6  # 将米转换为微米显示

    def get_wavelength_color(self, wavelength, is_defect=False):
        """根据波长获取对应的颜色"""
        if 380e-9 <= wavelength < 435e-9:  # 紫光
            return 'purple'
        elif 435e-9 <= wavelength < 450e-9:  # 蓝光
            return 'blue'
        elif 450e-9 <= wavelength < 492e-9:  # 青光
            return 'cyan'
        elif 492e-9 <= wavelength < 577e-9:  # 绿光
            return 'green'
        elif 577e-9 <= wavelength < 597e-9:  # 黄光
            return 'goldenrod'
        elif 597e-9 <= wavelength < 622e-9:  # 橙光
            return 'darkorange'
        elif 622e-9 <= wavelength < 760e-9:  # 红光
            return 'red' if not is_defect else 'darkred'
        else:
            return 'gray' if not is_defect else 'darkgray'

    def create_topography_data(self, ideal_thickness, defect_thickness, params, simulation_size=200):
        """创建三维形貌数据 - 使用毫米单位，坐标范围为 -3mm 到 3mm。"""
        # 创建网格坐标 - 与光强图像完全一致
        screen_length_x_mm = 6
        screen_length_y_mm = 6
        screen_length_x_m = screen_length_x_mm * 0.001  # 转换为米用于厚度计算
        screen_length_y_m = screen_length_y_mm * 0.001

        # 生成网格坐标
        x_m = np.linspace(-screen_length_x_m / 2, screen_length_x_m / 2, simulation_size)
        y_m = np.linspace(-screen_length_y_m / 2, screen_length_y_m / 2, simulation_size)
        X_m, Y_m = np.meshgrid(x_m, y_m)

        # 计算缺陷形貌（缺陷导致的厚度变化）
        # 突起应为正值，凹陷应为负值。且数值保持与参数一致
        if params.defect_type == "小突起":
            # params.bump_height是突起高度(um)
            target_amp = params.bump_height
            defect_topography = (ideal_thickness - defect_thickness) * self.scale_factor
            current_amp = np.max(np.abs(defect_topography)) if np.max(np.abs(defect_topography)) > 1e-9 else 1.0
            defect_topography = (np.abs(defect_topography) / current_amp) * target_amp
        elif params.defect_type == "小凹陷":
            # 凹陷应该是负值
            target_amp = -params.bump_height
            defect_topography = (ideal_thickness - defect_thickness) * self.scale_factor
            current_amp = np.max(np.abs(defect_topography)) if np.max(np.abs(defect_topography)) > 1e-9 else 1.0
            defect_topography = -(np.abs(defect_topography) / current_amp) * np.abs(target_amp)
        else:
            defect_topography = (defect_thickness - ideal_thickness) * self.scale_factor  # 转换为微米

        # 将坐标转换为毫米用于显示
        X_mm = X_m * 1000  # 米 → 毫米
        Y_mm = Y_m * 1000  # 米 → 毫米

        return X_mm, Y_mm, defect_topography

    def calculate_defect_thickness(self, params):
        """根据参数计算缺陷形貌 - 确保使用与光强图相同的物理长度"""
        lamda = params.get_wavelength_meters()
        d0 = params.get_d0_meters()
        R_m = params.get_R_meters()

        # 使用与干涉图一致的居中视场：6mm x 6mm
        Screen_length_x = getattr(params, 'screen_length_x', 0.006)
        Screen_length_y = getattr(params, 'screen_length_y', 0.006)

        # 生成X和Y方向的坐标
        x = np.linspace(-Screen_length_x / 2, Screen_length_x / 2, params.simulation_size)
        y = np.linspace(-Screen_length_y / 2, Screen_length_y / 2, params.simulation_size)
        X, Y = np.meshgrid(x, y)  # 创建网格

        # 理想厚度
        ideal_thickness = d0 + (X**2 + Y**2) / (2 * R_m)

        # 有缺陷厚度
        defect_thickness = ideal_thickness.copy()

        if params.defect_type == "无瑕疵":
            pass
        elif params.defect_type == "灰尘污染":
            defect_thickness = params.add_dust_defect(ideal_thickness, X, Y)
        elif params.defect_type == "表面划痕":
            defect_thickness = params.add_scratch_defect(ideal_thickness, X, Y)
        elif params.defect_type == "小突起":
            defect_thickness = params.add_bump_defect(ideal_thickness, X, Y, bump_type="突起")
        elif params.defect_type == "小凹陷":
            defect_thickness = params.add_bump_defect(ideal_thickness, X, Y, bump_type="凹陷")

        return ideal_thickness, defect_thickness

    def update_plot(self, params, use_manual_scratches=False, manual_scratch_mask=None):
        """更新三维缺陷形貌图"""
        self.ax.clear()

        # 计算缺陷形貌
        ideal_thickness, defect_thickness = self.calculate_defect_thickness(params)

        # 创建三维形貌数据
        simulation_size = params.simulation_size
        self.X, self.Y, self.defect_topography = self.create_topography_data(
            ideal_thickness, defect_thickness, params, simulation_size
        )

        # 根据缺陷类型选择颜色
        if params.defect_type in ["小突起"] or (params.defect_type == "表面划痕" and params.scratch_type == "突起划痕"):
            # 突起类缺陷用暖色调
            color_map = 'hot'
            defect_color = 'red'
        elif params.defect_type in ["小凹陷"] or (
                params.defect_type == "表面划痕" and params.scratch_type == "凹陷划痕"):
            # 凹陷类缺陷用冷色调
            color_map = 'cool'
            defect_color = 'blue'
        elif params.defect_type == "灰尘污染":
            # 灰尘用棕色调
            color_map = 'copper'
            defect_color = 'brown'
        elif params.defect_type == "头发丝遮挡":  # 新增头发丝颜色
            # 头发丝用暗色调
            color_map = 'Greys'
            defect_color = 'black'
        else:
            # 无瑕疵用灰色
            color_map = 'gray'
            defect_color = 'gray'

        # 增大字体大小以适应更大的图形
        title_fontsize = 12
        label_fontsize = 10
        tick_fontsize = 9

        # 绘制三维表面
        if params.defect_type == "无瑕疵":
            # 无瑕疵时显示平面
            Z_flat = np.zeros_like(self.defect_topography)
            surf = self.ax.plot_surface(self.X, self.Y, Z_flat,
                                        color='lightgray', alpha=0.8,
                                        linewidth=0, antialiased=True)
        else:
            # 有缺陷时显示形貌
            surf = self.ax.plot_surface(self.X, self.Y, self.defect_topography,
                                        cmap=color_map, alpha=0.8,
                                        linewidth=0, antialiased=True)

        # 设置标题和标签
        title = f'缺陷三维形貌 ({params.defect_type})'
        if params.defect_type == "表面划痕":
            if use_manual_scratches and manual_scratch_mask is not None:
                title += " (手动绘制)"
            else:
                title += " (自动生成)"

        self.ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

        # 使用毫米单位，与光强图像一致
        self.ax.set_xlabel('x (mm)', fontsize=label_fontsize, labelpad=12)
        self.ax.set_ylabel('y (mm)', fontsize=label_fontsize, labelpad=12)
        self.ax.set_zlabel('形貌高度 (um)', fontsize=label_fontsize, labelpad=12)

        # 设置坐标轴范围：-3mm 到 3mm
        screen_length_x_mm = 6  # 毫米
        screen_length_y_mm = 6  # 毫米
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)

        # 设置刻度，X轴和Y轴都以每3mm为一格
        xticks = np.linspace(-3, 3, 7)
        yticks = np.linspace(-3, 3, 7)
        self.ax.set_xticks(xticks)
        self.ax.set_yticks(yticks)

        # ============== Z轴自适应最大值调整逻辑 ==============
        if params.defect_type != "无瑕疵":
            z_data = self.defect_topography
            z_min = np.nanmin(z_data)  # 使用nanmin避免NaN值
            z_max = np.nanmax(z_data)
            z_range = z_max - z_min

            # 基于最大高度自动调整Z轴范围
            z_abs_max = max(abs(z_min), abs(z_max))

            if z_range > 0:  # 如果有数据变化
                # 确定显示范围（以最大绝对值为基准）
                if z_abs_max > 0.01:  # 如果有显著的高度变化
                    display_margin = 0.1  # 10%边距
                    display_range = z_abs_max * (1 + 2 * display_margin)  # 双向边距

                    if params.defect_type == "表面划痕" or params.defect_type in ["小突起", "小凹陷"]:
                        # 对于表面划痕、小突起/小凹陷，优先显示完整数据范围
                        if z_min >= 0:  # 只有正值（突起类）
                            z_min = 0
                            z_max = max(z_max * (1 + display_margin), 0.1)  # 最小显示0.1微米
                        elif z_max <= 0:  # 只有负值（凹陷类）
                            z_min = min(z_min * (1 + display_margin), -0.1)  # 最小显示-0.1微米
                            z_max = 0
                        else:  # 正负值混合
                            z_min = -z_abs_max * (1 + display_margin)
                            z_max = z_abs_max * (1 + display_margin)
                    else:
                        # 对于其他缺陷类型（如灰尘污染），使用对称范围
                        z_min = -z_abs_max * (1 + display_margin)
                        z_max = z_abs_max * (1 + display_margin)
                else:
                    # 高度变化很小的情况，设置一个最小显示范围
                    min_display_range = 0.2  # 最小显示范围0.2微米
                    z_center = (z_min + z_max) / 2
                    z_half_range = max(z_abs_max, min_display_range / 2)
                    z_min = z_center - z_half_range
                    z_max = z_center + z_half_range
            else:
                # 无变化的情况（平面）
                z_min = -0.1
                z_max = 0.1

            # 确保Z轴范围合理，避免极端情况
            z_min = max(-10.0, min(10.0, z_min))  # 限制在±10微米内
            z_max = max(-10.0, min(10.0, z_max))

        else:
            # 无瑕疵时显示平面，设置较小的固定范围
            z_min = -0.1
            z_max = 0.1

        # 设置Z轴范围
        self.ax.set_zlim(z_min, z_max)

        # 根据数据范围动态调整Z轴刻度
        z_ticks = self._get_adaptive_z_ticks(z_min, z_max)
        self.ax.set_zticks(z_ticks)
        # ============== Z轴调整逻辑结束 ==============

        # ============== 关键修改：不使用set_box_aspect，但确保XY轴每一格长度一致 ==============
        # 计算XY轴物理比例（当前视场为1:1）
        x_range = screen_length_x_mm
        y_range = screen_length_y_mm
        xy_ratio = x_range / y_range

        # 禁用自动缩放，保持手動設置的範圍
        self.ax.autoscale(enable=False)

        # 重新确认坐标轴范围
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(z_min, z_max)

        # 设置优化后的视角以确保XY轴比例正确，azim设为负值使X轴从-3到3符合直觉
        # 较低的俯仰角有助于保持XY平面的比例
        self.ax.view_init(elev=18, azim=-60)

        # 强制重新计算坐标轴显示比例
        self.ax.figure.canvas.draw()

        # 获取当前显示比例
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        # 检查并调整显示比例
        current_ratio = (current_xlim[1] - current_xlim[0]) / (current_ylim[1] - current_ylim[0])
        target_ratio = xy_ratio

        # 如果比例偏差较大，微调视图
        if abs(current_ratio - target_ratio) > 0.2:
            # 计算调整系数
            adjust_factor = target_ratio / current_ratio if current_ratio > 0 else 1.0

            # 调整视角
            if adjust_factor > 1.1:
                # 如果X轴显示太短，降低俯仰角
                self.ax.view_init(elev=15, azim=-60)
            elif adjust_factor < 0.9:
                # 如果X轴显示太长，稍微提高俯仰角
                self.ax.view_init(elev=22, azim=-60)

        # 确保Z轴刻度标签可见
        self.ax.zaxis.label.set_visible(True)
        self.ax.tick_params(axis='z', which='both', length=5, width=1, labelsize=9)
        self.ax.zaxis.set_rotate_label(False)
        self.ax.zaxis.label.set_rotation(0)
        self.ax.zaxis.labelpad = 12
        # ============== 关键修改结束 ==============

        # 添加缺陷信息标注
        if params.defect_type != "无瑕疵":
            # 计算缺陷统计信息
            max_height = np.max(self.defect_topography)
            min_height = np.min(self.defect_topography)
            avg_height = np.mean(self.defect_topography)
            rms_height = np.sqrt(np.mean(self.defect_topography ** 2))

            defect_info = params.defect_type
            if params.defect_type == "表面划痕":
                scratch_type = "凹陷" if params.scratch_type == "凹陷划痕" else "突起"
                defect_info += f" ({scratch_type})"
                # 在缺陷信息标注部分添加头发丝信息
                if params.defect_type == "头发丝遮挡":
                    defect_info = f"头发丝遮挡 (数量: {params.hair_count})"

        # 设置网格
        self.ax.grid(True, alpha=0.3)

        # 调整布局
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def _get_adaptive_z_ticks(self, z_min, z_max):
        """根据Z轴范围自适应设置刻度"""
        z_range = z_max - z_min

        if z_range <= 0.2:
            step = 0.05
        elif z_range <= 4.0:
            step = 0.1
        elif z_range <= 6.0:
            step = 0.15
        else:
            step = 0.2

        # 计算合适的起始点
        start = np.ceil(z_min / step) * step
        end = np.floor(z_max / step) * step

        # 生成刻度
        if start <= end:
            ticks = np.arange(start, end + step / 2, step)
        else:
            # 如果范围太小，只显示两个端点
            ticks = [z_min, z_max]

        return ticks
    def save_figure(self, filename):
        """保存图像到文件"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white',
                         transparent=False)


class SimulationTab(QWidget):
    """仿真结果标签页 - 固定尺寸版本"""

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.params = parameters
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 15, 8, 8)  # 增加上边距到15
        layout.setSpacing(20)  # 增加行间距到20

        # 添加顶部弹性空间，增加第一行的上边距
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 第一行：显示理想干涉图和瑕疵干涉图（各保留单一图样）
        interference_row = QHBoxLayout()
        interference_row.setSpacing(12)
        interference_row.setContentsMargins(0, 0, 0, 0)

        # 创建理想干涉图控件
        self.ideal_plot = SpectrumPlot()
        interference_row.addWidget(self.ideal_plot)

        # 创建瑕疵干涉图控件
        self.defect_plot = SpectrumPlot()
        interference_row.addWidget(self.defect_plot)

        layout.addLayout(interference_row)

        # 添加行间弹性空间，增加第一行和第二行之间的间距
        layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 第二行：并排显示手动画板和三维缺陷形貌图
        second_row = QHBoxLayout()
        second_row.setSpacing(12)
        second_row.setContentsMargins(0, 0, 0, 0)

        # 创建手动画板部件
        self.create_drawing_widget(second_row)

        # 创建三维形貌图部件
        self.create_3d_topography_widget(second_row)

        layout.addLayout(second_row)

        # 添加底部弹性空间
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def create_drawing_widget(self, parent_layout):
        """创建手动画板部件"""
        drawing_container = QWidget()
        drawing_container_layout = QVBoxLayout(drawing_container)
        drawing_container_layout.setContentsMargins(0, 0, 0, 0)

        # 手动画板部件
        self.drawing_widget = QWidget()
        drawing_widget_layout = QVBoxLayout(self.drawing_widget)
        drawing_widget_layout.setContentsMargins(0, 0, 0, 0)
        drawing_widget_layout.setSpacing(6)

        drawing_title = QLabel("手动画板 - 绘制划痕")
        drawing_title.setAlignment(Qt.AlignCenter)
        drawing_title.setStyleSheet("""
            QLabel {
                font-size: 24px; 
                font-weight: bold; 
                color: #333333;
                padding: 4px;
            }
        """)
        drawing_widget_layout.addWidget(drawing_title)

        # 手动画板容器
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setAlignment(Qt.AlignCenter)

        # 手动画板 - 与光谱图图案尺寸一致（600x300）
        self.drawing_canvas = DrawingCanvas()
        self.drawing_canvas.canvas_changed.connect(self.handle_manual_canvas_changed)
        canvas_layout.addWidget(self.drawing_canvas)
        drawing_widget_layout.addWidget(canvas_container)

        # 添加提示标签
        self.canvas_hint = QLabel("手动画板已禁用（仅在表面划痕模式下可用）")
        self.canvas_hint.setAlignment(Qt.AlignCenter)
        self.canvas_hint.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                padding: 4px;
            }
        """)
        drawing_widget_layout.addWidget(self.canvas_hint)

        drawing_container_layout.addWidget(self.drawing_widget)
        parent_layout.addWidget(drawing_container)

    def create_3d_topography_widget(self, parent_layout):
        """创建三维形貌图部件"""
        topography_widget = QWidget()
        topography_layout = QVBoxLayout(topography_widget)
        topography_layout.setContentsMargins(0, 0, 0, 0)
        topography_layout.setSpacing(4)

        topography_title = QLabel("三维形貌分布")
        topography_title.setAlignment(Qt.AlignCenter)
        topography_title.setStyleSheet("""
            QLabel {
                font-size: 24px; 
                font-weight: bold; 
                color: #333333;
                padding: 4px;
            }
        """)
        topography_layout.addWidget(topography_title)

        # 三维形貌图控件
        self.topography_plot_3d = Intensity3DPlot()
        topography_layout.addWidget(self.topography_plot_3d)

        parent_layout.addWidget(topography_widget)

    def set_drawing_widget_enabled(self, enabled):
        """设置手动画板部件的启用状态"""
        self.drawing_canvas.setEnabled(enabled)

        # 更新提示文本
        if enabled:
            self.canvas_hint.setText("提示：在画板上绘制线条，颜色越深表示划痕越深")
        else:
            self.canvas_hint.setText("手动画板已禁用（仅在表面划痕模式下可用）")

    def get_drawing_canvas(self):
        """获取手动画板"""
        return self.drawing_canvas

    def handle_manual_canvas_changed(self):
        """手动画板变化时实时更新瑕疵结果与三维光强分布"""
        try:
            if self.params.defect_type != "表面划痕":
                return

            mask_size = (self.params.simulation_size, self.params.simulation_size)
            if self.drawing_canvas.has_drawing():
                self.params.manual_scratch_mask = self.drawing_canvas.get_scratches_mask(mask_size)
                self.params.use_manual_scratches = True
            else:
                self.params.manual_scratch_mask = None
                self.params.use_manual_scratches = False

            self.update_simulation()
        except Exception as e:
            print(f"手动画板实时更新错误: {e}")

    def update_simulation(self):
        """更新仿真图像"""
        try:
            # 计算干涉图
            ideal_intensity, ideal_pattern = self.params.calculate_ideal_interference()
            defect_intensity, defect_pattern, defect_topography = self.params.calculate_defect_interference()

            # 获取波长
            lamda = self.params.get_wavelength_meters()

            # 更新理想干涉图
            self.ideal_plot.update_plot(ideal_pattern, '理想牛顿环干涉图样', lamda, is_defect=False)

            # 更新瑕疵干涉图
            defect_title = f'瑕疵牛顿环干涉图样 ({self.params.defect_type})'
            if self.params.defect_type == "表面划痕":
                if self.params.use_manual_scratches and self.params.manual_scratch_mask is not None:
                    defect_title += " (手动绘制)"
                else:
                    defect_title += " (自动生成)"
            self.defect_plot.update_plot(defect_pattern, defect_title, lamda, is_defect=True)

            # 更新三维形貌图
            topography_title = f'瑕疵三维形貌图 ({self.params.defect_type})'
            if self.params.defect_type == "表面划痕":
                if self.params.use_manual_scratches and self.params.manual_scratch_mask is not None:
                    topography_title += " (手动绘制)"
                else:
                    topography_title += " (自动生成)"

            self.topography_plot_3d.update_plot(defect_topography, topography_title, lamda, self.params.defect_type)

        except Exception as e:
            print(f"更新仿真图像错误: {e}")

    def save_figures(self, base_filename):
        """保存图像"""
        try:
            # 保存理想干涉图
            ideal_filename = base_filename.replace('.png', '_理想干涉图.png')
            self.ideal_plot.save_figure(ideal_filename)

            # 保存瑕疵干涉图
            defect_filename = base_filename.replace('.png', '_瑕疵干涉图.png')
            self.defect_plot.save_figure(defect_filename)

            # 保存三维形貌分布图
            topography_filename = base_filename.replace('.png', '_三维形貌分布.png')
            self.topography_plot_3d.save_figure(topography_filename)

            return ideal_filename, defect_filename, topography_filename
        except Exception as e:
            print(f"保存图像错误: {e}")
            return None, None, None


class ImageContainer(QWidget):
    """图像显示容器（包含标签页）"""

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.params = parameters
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 仿真结果标签页
        self.simulation_tab = SimulationTab(self.params)
        self.tab_widget.addTab(self.simulation_tab, "仿真结果")

        # AI检测结果标签页
        self.ai_result_label = QLabel("暂无检测结果\n请点击“保存图片”或“检测外部图片”以运行检测")
        self.ai_result_label.setAlignment(Qt.AlignCenter)
        self.ai_result_label.setScaledContents(True)
        self.tab_widget.addTab(self.ai_result_label, "AI 检测结果")

        layout.addWidget(self.tab_widget)

    def show_ai_result_image(self, img_path: str):
        """显示 AI 推理后的结果图像"""
        from PyQt5.QtGui import QPixmap
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            self.ai_result_label.setPixmap(pixmap)
            # 自动切换到 AI 结果标签页
            if self.tab_widget.indexOf(self.ai_result_label) != -1:
                self.tab_widget.setCurrentWidget(self.ai_result_label)

    def get_simulation_tab(self):
        """获取仿真结果标签页"""
        return self.simulation_tab

    def update_simulation(self):
        """更新仿真图像"""
        if hasattr(self, 'simulation_tab') and self.simulation_tab:
            self.simulation_tab.update_simulation()

    def get_drawing_canvas(self):
        """获取手动画板"""
        if hasattr(self, 'simulation_tab') and self.simulation_tab:
            return self.simulation_tab.get_drawing_canvas()
        return None

    def set_drawing_widget_enabled(self, enabled):
        """设置手动画板部件的启用状态"""
        if hasattr(self, 'simulation_tab') and self.simulation_tab:
            self.simulation_tab.set_drawing_widget_enabled(enabled)

    def save_figures(self, base_filename):
        """保存图像"""
        if hasattr(self, 'simulation_tab') and self.simulation_tab:
            return self.simulation_tab.save_figures(base_filename)
        return None, None, None

    def update_stylesheet(self, scale_factor):
        """更新样式表"""
        try:
            # 这里可以根据缩放因子调整样式
            pass
        except Exception as e:
            print(f"更新样式表错误: {e}")