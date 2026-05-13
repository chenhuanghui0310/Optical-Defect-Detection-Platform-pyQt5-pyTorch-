import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import QSizePolicy  # pyright: ignore[reportMissingImports]


class SpectrumPlot(FigureCanvas):
    """光谱图控件 - 自适应版本"""

    def __init__(self, parent=None):
        # 创建包含干涉图子图的figure
        self.fig, (self.ideal_ax, self.defect_ax) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.set_facecolor('#FFFFFF')
        self.fig.tight_layout(pad=3.0)

        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 设置DPI适应缩放
        self.dpi = 100

    def update_figure_size(self, scale_factor):
        """根据缩放因子更新图形大小"""
        base_width = 10
        base_height = 4
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

    def get_color_mapping(self, wavelength):
        """根据波长获取颜色映射"""
        wavelength_color = self.get_wavelength_color(wavelength)

        colors = [(0, 'black'), (1, wavelength_color)]
        return LinearSegmentedColormap.from_list('custom_cmap', colors)

    def update_plots(self, ideal_pattern, defect_pattern, lamda, defect_type,
                     use_manual_scratches=False, manual_scratch_mask=None):
        """更新干涉图像（并排显示理想和有瑕疵图像）"""
        # 清除子图
        self.ideal_ax.clear()
        self.defect_ax.clear()

        # 获取颜色映射
        cmap = self.get_color_mapping(lamda)

        # 根据缩放因子调整字体大小
        scale_factor = getattr(self, 'scale_factor', 1.0)
        title_fontsize = max(8, int(12 * scale_factor))
        label_fontsize = max(6, int(10 * scale_factor))

        # 显示理想干涉图像
        im1 = self.ideal_ax.imshow(ideal_pattern, extent=[0, 15, 0, 15],  # 0.015m -> 15mm
                                   cmap=cmap, origin='lower', aspect='auto')
        self.ideal_ax.set_title('理想劈尖干涉图样', fontsize=title_fontsize, fontweight='bold')
        self.ideal_ax.set_xlabel('x (mm)', fontsize=label_fontsize)  # m -> mm
        self.ideal_ax.set_ylabel('y (mm)', fontsize=label_fontsize)  # m -> mm

        im2 = self.defect_ax.imshow(defect_pattern, extent=[0, 15, 0, 15],  # 0.015m -> 15mm
                                    cmap=cmap, origin='lower', aspect='auto')
        self.defect_ax.set_title('理想劈尖干涉图样', fontsize=title_fontsize, fontweight='bold')
        self.defect_ax.set_xlabel('x (mm)', fontsize=label_fontsize)  # m -> mm
        self.defect_ax.set_ylabel('y (mm)', fontsize=label_fontsize)  # m -> mm

        # 调整布局以适应不同尺寸
        self.fig.tight_layout(pad=3.0 * scale_factor)
        self.draw()

    def save_figure(self, filename):
        """保存图像到文件"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')