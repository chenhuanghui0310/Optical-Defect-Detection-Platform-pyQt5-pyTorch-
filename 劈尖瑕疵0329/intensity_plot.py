import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PyQt5.QtWidgets import QSizePolicy


class IntensityPlot3D(FigureCanvas):
    """三维光强分布图控件 - 自适应版本"""

    def __init__(self, parent=None):
        # 创建包含三维光强分布子图的figure
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.set_facecolor('#FFFFFF')

        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 设置默认视角
        self.ax.view_init(elev=30, azim=45)

    def update_figure_size(self, scale_factor):
        """根据缩放因子更新图形大小"""
        base_width = 10
        base_height = 6
        self.fig.set_size_inches(base_width * scale_factor, base_height * scale_factor)
        self.fig.tight_layout(pad=3.0 * scale_factor)
        self.draw()

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

    def create_surface_data(self, intensity_curve, length=0.015, width=0.005):
        """创建三维表面数据"""
        x = np.linspace(0, length, len(intensity_curve))
        y = np.linspace(-width / 2, width / 2, 50)

        X, Y = np.meshgrid(x, y)

        # 在y方向扩展光强数据，使用高斯分布
        Z = np.zeros_like(X)
        for i in range(len(x)):
            # 使用高斯分布在y方向平滑扩展
            gaussian_profile = np.exp(-(Y[0, :] / (width / 4)) ** 2)
            Z[:, i] = intensity_curve[i] * gaussian_profile

        return X, Y, Z

    def update_plot(self, ideal_intensity, defect_intensity, lamda, defect_type,
                    use_manual_scratches=False, manual_scratch_mask=None):
        """更新三维光强分布图"""
        self.ax.clear()

        # 创建三维表面数据
        X_ideal, Y_ideal, Z_ideal = self.create_surface_data(ideal_intensity)
        X_defect, Y_defect, Z_defect = self.create_surface_data(defect_intensity)

        # 获取颜色映射
        wavelength_color = self.get_wavelength_color(lamda)
        defect_color = self.get_wavelength_color(lamda, is_defect=True)

        # 根据缩放因子调整字体大小
        scale_factor = getattr(self, 'scale_factor', 1.0)
        title_fontsize = max(10, int(14 * scale_factor))
        label_fontsize = max(8, int(11 * scale_factor))
        legend_fontsize = max(7, int(10 * scale_factor))

        # 绘制理想干涉三维表面
        surf_ideal = self.ax.plot_surface(X_ideal, Y_ideal, Z_ideal,
                                          color=wavelength_color, alpha=0.7,
                                          label='理想干涉')

        # 绘制有瑕疵干涉三维表面
        surf_defect = self.ax.plot_surface(X_defect, Y_defect, Z_defect,
                                           color=defect_color, alpha=0.7,
                                           label=f'{defect_type}')

        # 设置标题和标签
        title = '三维光强分布对比'
        if defect_type == "表面划痕":
            title += " (手动绘制)"

        self.ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        self.ax.set_xlabel('x (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_ylabel('y (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_zlabel('相对光强 I/I0', fontsize=label_fontsize, labelpad=10)

        # 设置坐标轴范围
        self.ax.set_xlim(0, 0.015)
        self.ax.set_ylim(-0.0025, 0.0025)
        self.ax.set_zlim(0, 4.5)

        # 创建图例（由于3D图例比较复杂，使用文本标注）
        self.ax.text2D(0.02, 0.95, '理想干涉', transform=self.ax.transAxes,
                       color=wavelength_color, fontsize=legend_fontsize, fontweight='bold')

        defect_label = f'{defect_type}'
        if defect_type == "表面划痕":
            defect_label += " (手动绘制)"

        self.ax.text2D(0.02, 0.90, defect_label, transform=self.ax.transAxes,
                       color=defect_color, fontsize=legend_fontsize, fontweight='bold')

        # 添加RMS差异注释
        if defect_type != "无瑕疵":
            rms_diff = np.sqrt(np.mean((ideal_intensity - defect_intensity) ** 2))
            self.ax.text2D(0.02, 0.85, f'RMS差异: {rms_diff:.4f}',
                           transform=self.ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                           fontsize=legend_fontsize - 2)

        # 设置网格
        self.ax.grid(True, alpha=0.3)

        # 设置视角
        self.ax.view_init(elev=30, azim=45)

        # 调整布局
        self.fig.tight_layout(pad=3.0 * scale_factor)
        self.draw()

    def update_plot_alternative(self, ideal_pattern, defect_pattern, lamda, defect_type,
                                use_manual_scratches=False, manual_scratch_mask=None):
        """替代版本：直接使用二维干涉图数据创建三维图"""
        self.ax.clear()

        # 获取数据尺寸
        height, width = ideal_pattern.shape
        x = np.linspace(0, 0.015, width)
        y = np.linspace(0, 0.015, height)

        X, Y = np.meshgrid(x, y)

        # 获取颜色映射
        wavelength_color = self.get_wavelength_color(lamda)
        defect_color = self.get_wavelength_color(lamda, is_defect=True)

        # 根据缩放因子调整字体大小
        scale_factor = getattr(self, 'scale_factor', 1.0)
        title_fontsize = max(10, int(14 * scale_factor))
        label_fontsize = max(8, int(11 * scale_factor))
        legend_fontsize = max(7, int(10 * scale_factor))

        # 绘制理想干涉三维表面
        surf_ideal = self.ax.plot_surface(X, Y, ideal_pattern,
                                          cmap='viridis', alpha=0.8,
                                          linewidth=0, antialiased=True)

        # 绘制有瑕疵干涉三维表面（半透明覆盖）
        surf_defect = self.ax.plot_surface(X, Y, defect_pattern,
                                           color=defect_color, alpha=0.4,
                                           linewidth=0, antialiased=True)

        # 设置标题和标签
        title = '三维干涉光强分布'
        if defect_type == "表面划痕":
            title += " (手动绘制)"

        self.ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        self.ax.set_xlabel('x (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_ylabel('y (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_zlabel('相对光强 I/I0', fontsize=label_fontsize, labelpad=10)

        # 设置坐标轴范围
        self.ax.set_zlim(0, 4.5)

        # 创建图例
        self.ax.text2D(0.02, 0.95, '理想干涉', transform=self.ax.transAxes,
                       color='blue', fontsize=legend_fontsize, fontweight='bold')

        defect_label = f'{defect_type}'
        self.ax.text2D(0.02, 0.90, defect_label, transform=self.ax.transAxes,
                       color=defect_color, fontsize=legend_fontsize, fontweight='bold')

        # 添加RMS差异注释
        if defect_type != "无瑕疵":
            rms_diff = np.sqrt(np.mean((ideal_pattern - defect_pattern) ** 2))
            self.ax.text2D(0.02, 0.85, f'RMS差异: {rms_diff:.4f}',
                           transform=self.ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                           fontsize=legend_fontsize - 2)

        # 设置网格
        self.ax.grid(True, alpha=0.3)

        # 设置视角
        self.ax.view_init(elev=30, azim=45)

        # 调整布局
        self.fig.tight_layout(pad=3.0 * scale_factor)
        self.draw()

    def update_plot_simple(self, intensity_pattern, lamda, title="三维光强分布"):
        """简化版本：只显示单个三维表面"""
        self.ax.clear()

        # 获取数据尺寸
        height, width = intensity_pattern.shape
        x = np.linspace(0, 0.015, width)
        y = np.linspace(0, 0.015, height)

        X, Y = np.meshgrid(x, y)

        # 获取颜色映射
        wavelength_color = self.get_wavelength_color(lamda)

        # 根据缩放因子调整字体大小
        scale_factor = getattr(self, 'scale_factor', 1.0)
        title_fontsize = max(10, int(14 * scale_factor))
        label_fontsize = max(8, int(11 * scale_factor))

        # 绘制三维表面
        surf = self.ax.plot_surface(X, Y, intensity_pattern,
                                    cmap='viridis', alpha=0.8,
                                    linewidth=0, antialiased=True)

        # 设置标题和标签
        self.ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        self.ax.set_xlabel('x (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_ylabel('y (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_zlabel('相对光强 I/I0', fontsize=label_fontsize, labelpad=10)

        # 设置坐标轴范围
        self.ax.set_zlim(0, 4.5)

        # 添加颜色条
        self.fig.colorbar(surf, ax=self.ax, shrink=0.6, aspect=20, pad=0.1)

        # 设置网格
        self.ax.grid(True, alpha=0.3)

        # 设置视角
        self.ax.view_init(elev=30, azim=45)

        # 调整布局
        self.fig.tight_layout(pad=3.0 * scale_factor)
        self.draw()

    def update_plot_comparison(self, ideal_pattern, defect_pattern, lamda, defect_type):
        """对比版本：并排显示理想和有瑕疵的干涉图"""
        self.ax.clear()

        # 获取数据尺寸
        height, width = ideal_pattern.shape
        x = np.linspace(0, 0.015, width)
        y = np.linspace(0, 0.015, height)

        X, Y = np.meshgrid(x, y)

        # 获取颜色映射
        wavelength_color = self.get_wavelength_color(lamda)
        defect_color = self.get_wavelength_color(lamda, is_defect=True)

        # 根据缩放因子调整字体大小
        scale_factor = getattr(self, 'scale_factor', 1.0)
        title_fontsize = max(10, int(14 * scale_factor))
        label_fontsize = max(8, int(11 * scale_factor))

        # 创建子图位置偏移
        offset = 0.002  # 轻微的偏移量

        # 绘制理想干涉图（稍微向左偏移）
        surf_ideal = self.ax.plot_surface(X - offset, Y, ideal_pattern,
                                         color=wavelength_color, alpha=0.7,
                                         label='理想干涉')

        # 绘制有瑕疵干涉图（稍微向右偏移）
        surf_defect = self.ax.plot_surface(X + offset, Y, defect_pattern,
                                          color=defect_color, alpha=0.7,
                                          label=f'{defect_type}')

        # 设置标题和标签
        title = f'三维光强分布对比 - {defect_type}'
        if defect_type == "表面划痕":
            title += " (手动绘制)"

        self.ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        self.ax.set_xlabel('x (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_ylabel('y (mm)', fontsize=label_fontsize, labelpad=10)
        self.ax.set_zlabel('相对光强 I/I0', fontsize=label_fontsize, labelpad=10)

        # 设置坐标轴范围
        self.ax.set_xlim(-offset, 0.015 + offset)
        self.ax.set_ylim(0, 0.015)
        self.ax.set_zlim(0, 4.5)

        # 添加图例
        self.ax.text2D(0.02, 0.95, '理想干涉', transform=self.ax.transAxes,
                       color=wavelength_color, fontsize=10, fontweight='bold')
        self.ax.text2D(0.02, 0.90, defect_type, transform=self.ax.transAxes,
                       color=defect_color, fontsize=10, fontweight='bold')

        # 设置网格
        self.ax.grid(True, alpha=0.3)

        # 设置视角
        self.ax.view_init(elev=30, azim=45)

        # 调整布局
        self.fig.tight_layout(pad=3.0 * scale_factor)
        self.draw()

    def clear_plot(self):
        """清除图形"""
        self.ax.clear()
        self.draw()

    def set_scale_factor(self, scale_factor):
        """设置缩放因子"""
        self.scale_factor = scale_factor

    def get_scale_factor(self):
        """获取缩放因子"""
        return getattr(self, 'scale_factor', 1.0)

    def save_figure(self, filename, dpi=300, transparent=False):
        """保存图像到文件"""
        try:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                           facecolor='white', transparent=transparent)
            return True
        except Exception as e:
            print(f"保存图像错误: {e}")
            return False

    def set_view_angle(self, elev=None, azim=None):
        """设置视角"""
        if elev is not None:
            self.ax.elev = elev
        if azim is not None:
            self.ax.azim = azim
        self.draw()

    def reset_view(self):
        """重置视角到默认值"""
        self.ax.view_init(elev=30, azim=45)
        self.draw()

    def set_colormap(self, cmap_name='viridis'):
        """设置颜色映射"""
        self.ax.clear()
        # 需要在下次绘图时应用新的颜色映射
        self.current_cmap = cmap_name

    def get_plot_info(self):
        """获取绘图信息"""
        return {
            'elevation': getattr(self.ax, 'elev', 30),
            'azimuth': getattr(self.ax, 'azim', 45),
            'scale_factor': getattr(self, 'scale_factor', 1.0),
            'figure_size': self.fig.get_size_inches(),
            'axes_limits': {
                'xlim': self.ax.get_xlim(),
                'ylim': self.ax.get_ylim(),
                'zlim': self.ax.get_zlim()
            }
        }

    def set_axes_limits(self, xlim=None, ylim=None, zlim=None):
        """设置坐标轴范围"""
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        if zlim is not None:
            self.ax.set_zlim(zlim)
        self.draw()

    def enable_grid(self, enabled=True):
        """启用/禁用网格"""
        self.ax.grid(enabled, alpha=0.3)
        self.draw()

    def set_background_color(self, color='white'):
        """设置背景颜色"""
        self.fig.set_facecolor(color)
        self.ax.set_facecolor(color)
        self.draw()

    def set_title(self, title, fontsize=14, fontweight='bold'):
        """设置标题"""
        self.ax.set_title(title, fontsize=fontsize, fontweight=fontweight, pad=20)
        self.draw()

    def set_labels(self, xlabel='x (m)', ylabel='y (m)', zlabel='相对光强 I/I0'):
        """设置坐标轴标签"""
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_zlabel(zlabel)
        self.draw()

    def set_font_sizes(self, title_size=14, label_size=11, tick_size=10):
        """设置字体大小"""
        self.ax.title.set_fontsize(title_size)
        self.ax.xaxis.label.set_fontsize(label_size)
        self.ax.yaxis.label.set_fontsize(label_size)
        self.ax.zaxis.label.set_fontsize(label_size)
        self.ax.tick_params(axis='both', which='major', labelsize=tick_size)
        self.draw()

    def add_annotation(self, text, x, y, z, color='red', fontsize=10):
        """添加注释"""
        self.ax.text(x, y, z, text, color=color, fontsize=fontsize)
        self.draw()

    def remove_annotations(self):
        """移除所有注释"""
        for text in self.ax.texts:
            text.remove()
        self.draw()

    def get_current_plot_data(self):
        """获取当前绘图数据"""
        # 返回当前显示的数据（如果存在）
        collections = self.ax.collections
        if len(collections) > 0:
            return {
                'has_data': True,
                'collection_count': len(collections),
                'data_types': [type(col).__name__ for col in collections]
            }
        return {'has_data': False}

    def export_to_array(self, dpi=300):
        """将图形导出为numpy数组"""
        # 将图形渲染为RGB数组
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return buf

    def close(self):
        """关闭图形"""
        plt.close(self.fig)

    def __del__(self):
        """析构函数"""
        try:
            plt.close(self.fig)
        except:
            pass