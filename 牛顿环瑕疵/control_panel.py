from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QPushButton, QSizePolicy, QSlider)
from PyQt5.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """左侧控制面板 - 优化尺寸版本"""

    # 定义信号
    simulation_updated = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.params = parameters
        self.scale_factor = 1.0

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(550)
        self.setMaximumWidth(900)

        self._init_styles()
        self.init_ui()

    def _init_styles(self):
        """初始化所有样式字符串"""
        # 字体大小
        base_font_size = 18
        title_font_size = 22
        label_font_size = 18
        control_font_size = 18

        # 尺寸设置
        control_height = 30
        min_width = 120
        padding_vertical = 1
        padding_horizontal = 6
        border_width = 1
        border_radius = 4

        # SpinBox样式 - 使用Qt默认箭头样式
        self.spinbox_style = f"""
                    QSpinBox, QDoubleSpinBox {{
                        font-size: {control_font_size}px; 
                        padding: 0px 0px 0px 12px;
                        border: {border_width}px solid #BDC3C7; 
                        border-radius: {border_radius}px;
                        background-color: white;
                        color: #333333;
                        min-height: {control_height}px;
                        max-height: {control_height}px;
                        min-width: {min_width}px;
                        margin: 2px;
                    }}
                """

        self.disabled_widget_style = f"""
                QSpinBox, QDoubleSpinBox {{
                font-size: {control_font_size}px; 
                padding: 0px 0px 0px 12px;
                border: {border_width}px solid #BDC3C7; 
                border-radius: {border_radius}px;
                background-color: white;
                color: #333333;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
                margin: 2px;
            }}
        """

        # ComboBox样式
        # ComboBox样式 - 使用Qt默认箭头样式
        self.combo_style = f"""
            QComboBox {{
                font-size: {control_font_size}px; 
                padding: 0px 0px 0px 12px;
                border: {border_width}px solid #BDC3C7; 
                border-radius: {border_radius}px;
                background-color: white;
                color: #333333;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
                margin: 2px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #3498db;
                selection-color: white;
                outline: none;
                font-size: {control_font_size}px;
            }}
        """

        self.disabled_combo_style = f"""
            QComboBox {{
                font-size: {control_font_size}px; 
                padding: 0px 0px 0px 12px;
                border: {border_width}px solid #CCCCCC; 
                border-radius: {border_radius}px;
                background-color: #F5F5F5;
                color: #999999;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
                margin: 2px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                background-color: #F5F5F5;
                selection-background-color: #95a5a6;
                selection-color: white;
                outline: none;
                font-size: {control_font_size}px;
            }}
        """

        # 标签样式
        self.label_style = f"""
            QLabel {{
                font-size: {label_font_size}px; 
                font-weight: bold; 
                color: #333333;
                background-color: white;
                border: {border_width}px solid #5DADE2;
                border-radius: {border_radius}px;
                padding: {padding_vertical}px {padding_horizontal}px;
                margin: 2px;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
            }}
        """

        self.disabled_label_style = f"""
            QLabel {{
                font-size: {label_font_size}px; 
                font-weight: bold; 
                color: #CCCCCC;
                background-color: #F5F5F5;
                border: {border_width}px solid #CCCCCC; 
                border-radius: {border_radius}px;
                padding: {padding_vertical}px {padding_horizontal}px;
                margin: 2px;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
            }}
        """

        # 灰尘密度标签样式
        self.dust_label_style = f"""
            QLabel {{
                font-size: {label_font_size + 1}px;
                font-weight: bold; 
                color: #333333;
                background-color: white;
                border: {border_width}px solid #5DADE2;
                border-radius: {border_radius}px;
                padding: {padding_vertical}px {padding_horizontal}px;
                margin: 2px;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
            }}
        """

        self.disabled_dust_label_style = f"""
            QLabel {{
                font-size: {label_font_size + 1}px;
                font-weight: bold; 
                color: #CCCCCC;
                background-color: #F5F5F5;
                border: {border_width}px solid #CCCCCC; 
                border-radius: {border_radius}px;
                padding: {padding_vertical}px {padding_horizontal}px;
                margin: 2px;
                min-height: {control_height}px;
                max-height: {control_height}px;
                min-width: {min_width}px;
            }}
        """

        # GroupBox样式
        self.groupbox_style = f"""
            QGroupBox {{
                font-weight: bold;
                font-size: {title_font_size}px;
                border: {border_width}px solid #3498db;
                border-radius: {border_radius}px;
                margin-top: 12px;
                margin-bottom: 12px;
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #FFFFFF;
                margin-left: 4px;
                margin-right: 4px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
            }}
        """

        self.drawing_groupbox_style = f"""
            QGroupBox {{
                font-weight: bold;
                font-size: {title_font_size}px;
                border: {border_width}px solid #e74c3c;
                border-radius: {border_radius}px;
                margin-top: 12px;
                margin-bottom: 12px;
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #FFFFFF;
                margin-left: 4px;
                margin-right: 4px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
                color: #c0392b;
            }}
        """

        self.disabled_groupbox_style = f"""
            QGroupBox {{
                font-weight: bold;
                font-size: {title_font_size}px;
                border: {border_width}px solid #95a5a6;
                border-radius: {border_radius}px;
                margin-top: 12px;
                margin-bottom: 12px;
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #f8f9fa;
                color: #95a5a6;
                margin-left: 4px;
                margin-right: 4px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
                color: #95a5a6;
            }}
        """

        # 按钮样式
        self.button_style = f"""
            QPushButton {{
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 8px;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
                min-height: 36px;
                margin: 2px;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
        """

        self.save_button_style = f"""
            QPushButton {{
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 6px 8px;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
                min-height: 36px;
                margin: 2px;
            }}
            QPushButton:hover {{
                background-color: #27ae60;
            }}
        """

        self.disabled_button_style = f"""
            QPushButton {{
                background-color: #95a5a6;
                color: #cccccc;
                border: none;
                padding: 6px 8px;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
                min-height: 36px;
                margin: 2px;
            }}
        """

    def init_ui(self):
        """初始化用户界面"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)

        self.create_basic_parameters_group()
        self.create_defect_parameters_group()
        self.create_drawing_settings_group()
        self.create_control_buttons()

        self.layout.addStretch(1)

    def create_basic_parameters_group(self):
        """创建基本参数设置组"""
        self.basic_group = QGroupBox("基本参数设置")
        self.basic_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.basic_group.setStyleSheet(self.groupbox_style)

        basic_layout = QVBoxLayout()
        basic_layout.setContentsMargins(8, 6, 8, 8)
        basic_layout.setSpacing(6)

        self.create_basic_parameter_widgets(basic_layout)
        self.basic_group.setLayout(basic_layout)
        self.layout.addWidget(self.basic_group)

    def create_basic_parameter_widgets(self, layout):
        """创建基本参数控件"""
        label_width_ratio = 1
        control_width_ratio = 2
        control_height = 30
        spacing = 6

        # 波长
        lambda_layout = QHBoxLayout()
        lambda_layout.setSpacing(spacing)

        lambda_label = QLabel("波长(nm):")
        lambda_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lambda_label.setFixedHeight(control_height)
        lambda_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        lambda_label.setStyleSheet(self.label_style)

        self.lambda_spin = QSpinBox()
        self.lambda_spin.setRange(380, 750)
        self.lambda_spin.setValue(self.params.lambda_nm)
        self.lambda_spin.setStyleSheet(self.spinbox_style)
        self.lambda_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lambda_spin.setFixedHeight(control_height)

        lambda_layout.addWidget(lambda_label, label_width_ratio)
        lambda_layout.addWidget(self.lambda_spin, control_width_ratio)
        layout.addLayout(lambda_layout)

        # 实际距离 e
        d0_layout = QHBoxLayout()
        d0_layout.setSpacing(spacing)

        d0_label = QLabel("实际距离e(μm):")
        d0_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        d0_label.setFixedHeight(control_height)
        d0_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        d0_label.setStyleSheet(self.label_style)

        self.d0_spin = QDoubleSpinBox()
        self.d0_spin.setRange(0.000, 15.000)
        self.d0_spin.setValue(self.params.d0_um)
        self.d0_spin.setSingleStep(0.05)
        self.d0_spin.setDecimals(3)
        self.d0_spin.setStyleSheet(self.spinbox_style)
        self.d0_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.d0_spin.setFixedHeight(control_height)

        d0_layout.addWidget(d0_label, label_width_ratio)
        d0_layout.addWidget(self.d0_spin, control_width_ratio)
        layout.addLayout(d0_layout)

        # 透镜曲率半径
        R_layout = QHBoxLayout()
        R_layout.setSpacing(spacing)

        R_label = QLabel("曲率半径R(mm):")
        R_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        R_label.setFixedHeight(control_height)
        R_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        R_label.setStyleSheet(self.label_style)

        self.R_spin = QDoubleSpinBox()
        self.R_spin.setRange(1000, 5000)
        self.R_spin.setValue(self.params.R_mm)
        self.R_spin.setSingleStep(100.0)
        self.R_spin.setDecimals(0)
        self.R_spin.setStyleSheet(self.spinbox_style)
        self.R_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.R_spin.setFixedHeight(control_height)

        R_layout.addWidget(R_label, label_width_ratio)
        R_layout.addWidget(self.R_spin, control_width_ratio)
        layout.addLayout(R_layout)

        # 折射率 n
        n_layout = QHBoxLayout()
        n_layout.setSpacing(spacing)

        n_label = QLabel("折射率n:")
        n_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        n_label.setFixedHeight(control_height)
        n_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        n_label.setStyleSheet(self.label_style)

        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(1.0, 5.0)
        self.n_spin.setValue(self.params.n)
        self.n_spin.setSingleStep(0.1)
        self.n_spin.setDecimals(1)
        self.n_spin.setStyleSheet(self.spinbox_style)
        self.n_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.n_spin.setFixedHeight(control_height)

        n_layout.addWidget(n_label, label_width_ratio)
        n_layout.addWidget(self.n_spin, control_width_ratio)
        layout.addLayout(n_layout)

    def create_defect_parameters_group(self):
        """创建瑕疵参数设置组"""
        self.defect_group = QGroupBox("瑕疵类型设置")
        self.defect_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.defect_group.setStyleSheet(self.groupbox_style)

        self.defect_layout = QVBoxLayout()
        self.defect_layout.setContentsMargins(8, 6, 8, 8)
        self.defect_layout.setSpacing(6)

        self.create_defect_parameter_widgets()
        self.defect_group.setLayout(self.defect_layout)
        self.layout.addWidget(self.defect_group)

    def create_defect_parameter_widgets(self):
        """创建瑕疵参数控件"""
        label_width_ratio = 1
        control_width_ratio = 2
        control_height = 30
        spacing = 6

        # 瑕疵类型选择
        defect_type_layout = QHBoxLayout()
        defect_type_layout.setSpacing(spacing)

        defect_type_label = QLabel("瑕疵类型:")
        defect_type_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        defect_type_label.setFixedHeight(control_height)
        defect_type_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        defect_type_label.setStyleSheet(self.label_style)

        self.defect_combo = QComboBox()
        self.defect_combo.addItems(["无瑕疵", "灰尘污染", "表面划痕", "小凸起", "小凹陷", "头发丝遮挡"])  # 新增头发丝
        self.defect_combo.setCurrentText(self.params.defect_type)
        self.defect_combo.setStyleSheet(self.combo_style)
        self.defect_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.defect_combo.setFixedHeight(control_height)

        defect_type_layout.addWidget(defect_type_label, label_width_ratio)
        defect_type_layout.addWidget(self.defect_combo, control_width_ratio)
        self.defect_layout.addLayout(defect_type_layout)

        # 灰尘密度控件
        dust_layout = QHBoxLayout()
        dust_layout.setSpacing(spacing)

        self.dust_label = QLabel("灰尘密度(%):")
        self.dust_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dust_label.setFixedHeight(control_height)
        self.dust_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.dust_label.setStyleSheet(self.disabled_label_style)

        self.dust_spin = QSpinBox()
        self.dust_spin.setRange(0, 30)
        self.dust_spin.setValue(self.params.dust_density)
        self.dust_spin.setStyleSheet(self.disabled_widget_style)
        self.dust_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dust_spin.setFixedHeight(control_height)
        self.dust_spin.setEnabled(False)

        dust_layout.addWidget(self.dust_label, label_width_ratio)
        dust_layout.addWidget(self.dust_spin, control_width_ratio)
        self.defect_layout.addLayout(dust_layout)

        # 划痕类型控件
        scratch_type_layout = QHBoxLayout()
        scratch_type_layout.setSpacing(spacing)

        self.scratch_type_label = QLabel("划痕类型:")
        self.scratch_type_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.scratch_type_label.setFixedHeight(control_height)
        self.scratch_type_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.scratch_type_label.setStyleSheet(self.disabled_label_style)

        self.scratch_type_combo = QComboBox()
        self.scratch_type_combo.addItems(["凹陷划痕", "凸起划痕"])
        self.scratch_type_combo.setCurrentText(self.params.scratch_type)
        self.scratch_type_combo.setStyleSheet(self.disabled_combo_style)
        self.scratch_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.scratch_type_combo.setFixedHeight(control_height)
        self.scratch_type_combo.setEnabled(False)

        scratch_type_layout.addWidget(self.scratch_type_label, label_width_ratio)
        scratch_type_layout.addWidget(self.scratch_type_combo, control_width_ratio)
        self.defect_layout.addLayout(scratch_type_layout)

        # 突起/凹陷数量控件
        bump_layout = QHBoxLayout()
        bump_layout.setSpacing(spacing)

        self.bump_label = QLabel("突起/凹陷数量:")
        self.bump_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_label.setFixedHeight(control_height)
        self.bump_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.bump_label.setStyleSheet(self.disabled_label_style)

        self.bump_spin = QSpinBox()
        self.bump_spin.setRange(1, 10)
        self.bump_spin.setValue(self.params.bump_count)
        self.bump_spin.setStyleSheet(self.disabled_widget_style)
        self.bump_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_spin.setFixedHeight(control_height)
        self.bump_spin.setEnabled(False)

        bump_layout.addWidget(self.bump_label, label_width_ratio)
        bump_layout.addWidget(self.bump_spin, control_width_ratio)
        self.defect_layout.addLayout(bump_layout)

        # 突起高度/凹陷深度控件
        bump_height_layout = QHBoxLayout()
        bump_height_layout.setSpacing(spacing)

        self.bump_height_label = QLabel("高度/深度(μm):")
        self.bump_height_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_height_label.setFixedHeight(control_height)
        self.bump_height_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.bump_height_label.setStyleSheet(self.disabled_label_style)

        # 突起高度/凹陷深度控件
        self.bump_height_spin = QDoubleSpinBox()
        self.bump_height_spin.setRange(0.01, 5.0)  # 范围 0.01–5.0 μm
        self.bump_height_spin.setValue(self.params.bump_height)
        self.bump_height_spin.setSingleStep(0.01)  # 修改这里：从0.1改为0.01
        self.bump_height_spin.setDecimals(2)  # 修改这里：从1改为2，以显示两位小数
        self.bump_height_spin.setStyleSheet(self.disabled_widget_style)
        self.bump_height_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_height_spin.setFixedHeight(control_height)
        self.bump_height_spin.setEnabled(False)

        bump_height_layout.addWidget(self.bump_height_label, label_width_ratio)
        bump_height_layout.addWidget(self.bump_height_spin, control_width_ratio)
        self.defect_layout.addLayout(bump_height_layout)

        # 突起/凹陷横向尺寸（有效半径，与仿真中圆形区域一致）
        bump_radius_layout = QHBoxLayout()
        bump_radius_layout.setSpacing(spacing)

        self.bump_radius_label = QLabel("横向半径(mm):")
        self.bump_radius_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_radius_label.setFixedHeight(control_height)
        self.bump_radius_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.bump_radius_label.setStyleSheet(self.disabled_label_style)
        self.bump_radius_label.setToolTip("数值越大，突起/凹陷在视场上占据的范围越大")

        self.bump_radius_spin = QDoubleSpinBox()
        self.bump_radius_spin.setRange(0.2, 7.0)
        self.bump_radius_spin.setValue(self.params.bump_radius_mm)
        self.bump_radius_spin.setSingleStep(0.1)
        self.bump_radius_spin.setDecimals(1)
        self.bump_radius_spin.setStyleSheet(self.disabled_widget_style)
        self.bump_radius_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_radius_spin.setFixedHeight(control_height)
        self.bump_radius_spin.setEnabled(False)

        bump_radius_layout.addWidget(self.bump_radius_label, label_width_ratio)
        bump_radius_layout.addWidget(self.bump_radius_spin, control_width_ratio)
        self.defect_layout.addLayout(bump_radius_layout)

        # 突起/凹陷位置模式控件
        bump_position_layout = QHBoxLayout()
        bump_position_layout.setSpacing(spacing)

        self.bump_position_label = QLabel("位置模式:")
        self.bump_position_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_position_label.setFixedHeight(control_height)
        self.bump_position_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.bump_position_label.setStyleSheet(self.disabled_label_style)

        self.bump_position_combo = QComboBox()
        self.bump_position_combo.addItems(["随机分布", "网格排列", "固定位置"])
        self.bump_position_combo.setCurrentText(self.params.bump_position_mode)
        self.bump_position_combo.setStyleSheet(self.disabled_combo_style)
        self.bump_position_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bump_position_combo.setFixedHeight(control_height)
        self.bump_position_combo.setEnabled(False)

        bump_position_layout.addWidget(self.bump_position_label, label_width_ratio)
        bump_position_layout.addWidget(self.bump_position_combo, control_width_ratio)
        self.defect_layout.addLayout(bump_position_layout)

        # ============== 在现有瑕疵参数控件后添加头发丝相关控件 ==============

        # 头发丝数量
        hair_layout = QHBoxLayout()
        hair_layout.setSpacing(spacing)

        self.hair_label = QLabel("头发丝数量:")
        self.hair_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.hair_label.setFixedHeight(control_height)
        self.hair_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.hair_label.setStyleSheet(self.disabled_label_style)

        self.hair_spin = QSpinBox()
        self.hair_spin.setRange(1, 20)
        self.hair_spin.setValue(self.params.hair_count)
        self.hair_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.hair_spin.setFixedHeight(control_height)
        self.hair_spin.setEnabled(False)

        hair_layout.addWidget(self.hair_label, label_width_ratio)
        hair_layout.addWidget(self.hair_spin, control_width_ratio)
        self.defect_layout.addLayout(hair_layout)
        # 添加弹性空间以确保布局正确
        self.defect_layout.addStretch(1)

    def create_drawing_settings_group(self):
        """创建绘图设置组"""
        self.drawing_group = QGroupBox("绘图设置")
        self.drawing_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.drawing_group.setVisible(True)

        if self.params.defect_type == "表面划痕":
            self.drawing_group.setStyleSheet(self.drawing_groupbox_style)
        else:
            self.drawing_group.setStyleSheet(self.disabled_groupbox_style)

        drawing_settings_layout = QVBoxLayout()
        drawing_settings_layout.setContentsMargins(8, 6, 8, 8)
        drawing_settings_layout.setSpacing(6)

        control_height = 30
        label_width_ratio = 1
        control_width_ratio = 2

        # 画笔粗细
        brush_layout = QHBoxLayout()
        brush_layout.setSpacing(6)

        brush_label = QLabel("画笔粗细(px):")
        brush_label.setObjectName("brush_label")
        brush_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        brush_label.setFixedHeight(control_height)
        brush_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        brush_label.setStyleSheet(self.disabled_label_style)

        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(5, 30)
        self.brush_size_spin.setValue(self.params.brush_size)
        self.brush_size_spin.setStyleSheet(self.disabled_widget_style)
        self.brush_size_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.brush_size_spin.setFixedHeight(control_height)
        self.brush_size_spin.setEnabled(False)

        brush_layout.addWidget(brush_label, label_width_ratio)
        brush_layout.addWidget(self.brush_size_spin, control_width_ratio)
        drawing_settings_layout.addLayout(brush_layout)

        # 画笔深度控制
        depth_layout = QHBoxLayout()
        depth_layout.setSpacing(6)

        depth_label = QLabel("画笔深度:")
        depth_label.setObjectName("depth_label")
        depth_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        depth_label.setFixedHeight(control_height)
        depth_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        depth_label.setStyleSheet(self.disabled_label_style)

        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(0, 100)  # 0-100%
        # 归一化深度(0.0-1.0) 映射到滑块值(0-100)
        slider_value = int(self.params.brush_depth * 100)
        self.depth_slider.setValue(slider_value)
        self.depth_slider.setFixedHeight(control_height)
        self.depth_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.depth_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #999999;
                        height: 8px;
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #FFFFFF, stop:1 #000000);
                        border-radius: 4px;
                    }
                    QSlider::handle:horizontal {
                        background: #95a5a6;
                        border: 1px solid #7f8c8d;
                        width: 18px;
                        margin: -2px 0;
                        border-radius: 9px;
                    }
                """)
        self.depth_slider.setFixedHeight(control_height)
        self.depth_slider.setEnabled(False)

        self.depth_value_label = QLabel(f"{slider_value}%")  # 显示百分比
        self.depth_value_label.setFixedWidth(40)
        self.depth_value_label.setFixedHeight(control_height)
        self.depth_value_label.setAlignment(Qt.AlignCenter)
        self.depth_value_label.setStyleSheet("""
                    QLabel {
                        font-size: 12px;
                        font-weight: bold;
                        color: #95a5a6;
                        background-color: #ecf0f1;
                        border: 1px solid #bdc3c7;
                        border-radius: 4px;
                        padding: 2px;
                    }
                """)

        depth_layout.addWidget(depth_label, label_width_ratio)
        depth_layout.addWidget(self.depth_slider, control_width_ratio)
        depth_layout.addWidget(self.depth_value_label)
        drawing_settings_layout.addLayout(depth_layout)

        # 深度说明标签
        depth_hint_label = QLabel("颜色越深表示划痕越深（当前禁用）")
        depth_hint_label.setAlignment(Qt.AlignCenter)
        depth_hint_label.setStyleSheet("""
                    QLabel {
                        font-size: 10px;
                        color: #95a5a6;
                        font-style: italic;
                        padding: 2px;
                    }
                """)
        drawing_settings_layout.addWidget(depth_hint_label)

        # 橡皮擦模式
        eraser_layout = QHBoxLayout()
        self.eraser_button = QPushButton("橡皮擦模式")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setChecked(self.params.eraser_mode)
        self.eraser_button.setStyleSheet("""
                    QPushButton {
                        background-color: #95a5a6;
                        color: #cccccc;
                        border: none;
                        padding: 4px 6px;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: bold;
                        min-height: 32px;
                    }
                """)
        self.eraser_button.setEnabled(False)
        eraser_layout.addWidget(self.eraser_button)
        drawing_settings_layout.addLayout(eraser_layout)

        # 清空画布按钮
        self.clear_canvas_button = QPushButton("清空画布")
        self.clear_canvas_button.setStyleSheet("""
                    QPushButton {
                        background-color: #95a5a6;
                        color: #cccccc;
                        border: none;
                        padding: 4px 6px;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: bold;
                        min-height: 32px;
                    }
                """)
        self.clear_canvas_button.setEnabled(False)
        drawing_settings_layout.addWidget(self.clear_canvas_button)

        # 应用手动划痕按钮
        self.apply_button = QPushButton("应用手动划痕")
        self.apply_button.setStyleSheet("""
                    QPushButton {
                        background-color: #95a5a6;
                        color: #cccccc;
                        border: none;
                        padding: 6px 8px;
                        border-radius: 4px;
                        font-size: 14px;
                        font-weight: bold;
                        min-height: 36px;
                    }
                """)
        self.apply_button.setEnabled(False)
        drawing_settings_layout.addWidget(self.apply_button)

        self.drawing_group.setLayout(drawing_settings_layout)
        self.layout.addWidget(self.drawing_group)

    def create_control_buttons(self):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)

        self.reset_button = QPushButton("重置参数")
        self.reset_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.reset_button.setStyleSheet(self.button_style)

        self.save_button = QPushButton("保存图像")
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.save_button.setStyleSheet(self.save_button_style)

        button_layout.addWidget(self.reset_button, 1)
        button_layout.addWidget(self.save_button, 1)
        self.layout.addLayout(button_layout)

    def connect_signals(self, update_simulation_slot, reset_parameters_slot, save_images_slot,
                        defect_type_changed_slot, on_brush_size_changed_slot,
                        on_brush_depth_changed_slot, on_eraser_mode_changed_slot,
                        clear_drawing_canvas_slot, apply_manual_scratches_slot,
                        bump_parameters_changed_slot, scratch_type_changed_slot):
        """连接信号"""
        # 基本参数
        self.lambda_spin.valueChanged.connect(update_simulation_slot)
        self.d0_spin.valueChanged.connect(update_simulation_slot)
        self.R_spin.valueChanged.connect(update_simulation_slot)
        self.n_spin.valueChanged.connect(update_simulation_slot)

        # 瑕疵参数
        self.defect_combo.currentTextChanged.connect(defect_type_changed_slot)
        self.dust_spin.valueChanged.connect(update_simulation_slot)
        self.scratch_type_combo.currentTextChanged.connect(scratch_type_changed_slot)
        # 头发丝参数
        self.hair_spin.valueChanged.connect(update_simulation_slot)  # 新增
        # 小突起/小凹陷参数
        self.bump_spin.valueChanged.connect(bump_parameters_changed_slot)
        self.bump_height_spin.valueChanged.connect(bump_parameters_changed_slot)
        self.bump_radius_spin.valueChanged.connect(bump_parameters_changed_slot)
        self.bump_position_combo.currentTextChanged.connect(bump_parameters_changed_slot)

        # 绘图设置
        self.brush_size_spin.valueChanged.connect(on_brush_size_changed_slot)
        self.depth_slider.valueChanged.connect(on_brush_depth_changed_slot)
        self.eraser_button.toggled.connect(on_eraser_mode_changed_slot)
        self.clear_canvas_button.clicked.connect(clear_drawing_canvas_slot)
        self.apply_button.clicked.connect(apply_manual_scratches_slot)

        # 控制按钮
        self.reset_button.clicked.connect(reset_parameters_slot)
        self.save_button.clicked.connect(save_images_slot)

    def update_parameters_from_ui(self):
        """从UI更新参数"""
        self.params.lambda_nm = self.lambda_spin.value()
        self.params.d0_um = self.d0_spin.value()
        self.params.R_mm = self.R_spin.value()  # 乘以10⁻⁵，不是乘以10⁵
        self.params.n = self.n_spin.value()
        self.params.n1 = self.params.n
        self.params.n2 = self.params.n
        self.params.n3 = self.params.n
        self.params.hair_count = self.hair_spin.value()  # 新增
        raw_defect = self.defect_combo.currentText()
        self.params.defect_type = "小突起" if raw_defect == "小凸起" else raw_defect
        self.params.dust_density = self.dust_spin.value()
        raw_scratch = self.scratch_type_combo.currentText()
        self.params.scratch_type = "突起划痕" if raw_scratch == "凸起划痕" else raw_scratch
        self.params.bump_count = self.bump_spin.value()
        self.params.bump_height = self.bump_height_spin.value()
        self.params.bump_radius_mm = self.bump_radius_spin.value()
        self.params.bump_position_mode = self.bump_position_combo.currentText()
        self.params.manual_scratch_type = self.params.scratch_type
        self.params.brush_size = self.brush_size_spin.value()

        # 修改部分：滑块值(0-100) 转换为归一化深度(0.0-1.0)
        self.params.brush_depth = self.depth_slider.value() / 100.0  # 除以100

        self.params.eraser_mode = self.eraser_button.isChecked()

    def update_ui_from_parameters(self):
        """从参数更新UI"""
        self.lambda_spin.setValue(self.params.lambda_nm)
        self.d0_spin.setValue(self.params.d0_um)
        self.R_spin.setValue(self.params.R_mm)
        self.n_spin.setValue(self.params.n)
        self.hair_spin.setValue(self.params.hair_count)  # 新增
        ui_defect_type = "小凸起" if self.params.defect_type == "小突起" else self.params.defect_type
        self.defect_combo.setCurrentText(ui_defect_type)
        self.dust_spin.setValue(self.params.dust_density)
        ui_scratch_type = "凸起划痕" if self.params.scratch_type == "突起划痕" else self.params.scratch_type
        self.scratch_type_combo.setCurrentText(ui_scratch_type)
        self.bump_spin.setValue(self.params.bump_count)
        self.bump_height_spin.setValue(self.params.bump_height)
        self.bump_radius_spin.setValue(self.params.bump_radius_mm)
        self.bump_position_combo.setCurrentText(self.params.bump_position_mode)
        self.brush_size_spin.setValue(self.params.brush_size)

        # 修改部分：归一化深度(0.0-1.0) 映射到滑块值(0-100)
        slider_value = int(self.params.brush_depth * 100)  # 0.0-1.0 -> 0-100
        self.depth_slider.setValue(slider_value)
        self.depth_value_label.setText(f"{slider_value}%")  # 显示百分比

        self.eraser_button.setChecked(self.params.eraser_mode)

        self.update_defect_controls_visibility()

    def update_defect_controls_visibility(self):
        """根据选择的瑕疵类型更新控件的可见性和启用状态"""
        defect_type = self.params.defect_type

        # 定义各瑕疵类型对应的控件
        widgets = {
            "灰尘污染": [self.dust_spin],
            "表面划痕": [self.scratch_type_combo],
            "小突起": [self.bump_spin, self.bump_height_spin, self.bump_radius_spin, self.bump_position_combo],
            "小凸起": [self.bump_spin, self.bump_height_spin, self.bump_radius_spin, self.bump_position_combo],
            "小凹陷": [self.bump_spin, self.bump_height_spin, self.bump_radius_spin, self.bump_position_combo],
            "头发丝遮挡": [self.hair_spin],
        }

        # 所有瑕疵参数标签
        all_labels = [
            self.dust_label, self.scratch_type_label,
            self.bump_label, self.bump_height_label, self.bump_radius_label, self.bump_position_label,
            self.hair_label,
        ]

        # 所有瑕疵参数控件
        all_widgets = [
            self.dust_spin, self.scratch_type_combo,
            self.bump_spin, self.bump_height_spin, self.bump_radius_spin, self.bump_position_combo,
            self.hair_spin,
        ]

        # 禁用所有瑕疵参数控件
        for widget in all_widgets:
            widget.setEnabled(False)
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setStyleSheet(self.disabled_widget_style)
            elif isinstance(widget, QComboBox):
                widget.setStyleSheet(self.disabled_combo_style)

        # 禁用所有标签
        for label in all_labels:
            label.setStyleSheet(self.disabled_label_style)

        # 如果选择"无瑕疵"，直接返回
        if defect_type == "无瑕疵":
            return

        # 根据当前瑕疵类型启用相关控件
        if defect_type in widgets:
            enabled_widgets = widgets[defect_type]
            for widget in enabled_widgets:
                widget.setEnabled(True)
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setStyleSheet(self.spinbox_style)
                elif isinstance(widget, QComboBox):
                    widget.setStyleSheet(self.combo_style)

        # 启用对应标签
        if defect_type == "灰尘污染":
            self.dust_label.setStyleSheet(self.label_style)

        elif defect_type == "表面划痕":
            self.scratch_type_label.setStyleSheet(self.label_style)

        elif defect_type in ["小突起", "小凸起", "小凹陷"]:
            self.bump_label.setStyleSheet(self.label_style)
            self.bump_height_label.setStyleSheet(self.label_style)
            self.bump_radius_label.setStyleSheet(self.label_style)
            self.bump_position_label.setStyleSheet(self.label_style)

        elif defect_type == "头发丝遮挡":
            self.hair_label.setStyleSheet(self.label_style)

        # ==== 新增：更新绘图设置控件 ====
        if defect_type == "表面划痕":
            # 启用绘图设置控件
            self.brush_size_spin.setEnabled(True)
            self.brush_size_spin.setStyleSheet(self.spinbox_style)

            self.depth_slider.setEnabled(True)
            self.depth_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    border: 1px solid #999999;
                    height: 8px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FFFFFF, stop:1 #000000);
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #3498db;
                    border: 1px solid #2980b9;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 9px;
                }
                QSlider::handle:horizontal:hover {
                    background: #5dade2;
                }
            """)

            # 更新深度值标签
            slider_value = int(self.params.brush_depth * 100)
            self.depth_value_label.setText(f"{slider_value}%")
            self.depth_value_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-weight: bold;
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    border: 1px solid #3498db;
                    border-radius: 4px;
                    padding: 2px;
                }
            """)

            # 启用其他绘图控件
            self.eraser_button.setEnabled(True)
            self.eraser_button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 4px 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    min-height: 32px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:checked {
                    background-color: #e74c3c;
                }
            """)

            self.clear_canvas_button.setEnabled(True)
            self.clear_canvas_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 4px 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    min-height: 32px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)

            self.apply_button.setEnabled(True)
            self.apply_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    border: none;
                    padding: 6px 8px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                    min-height: 36px;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
            """)

            # 启用绘图设置组
            self.drawing_group.setStyleSheet(self.drawing_groupbox_style)

            # ==== 关键修改：修复标签查找和样式更新 ====
            # 方法1：通过findChild查找标签
            brush_label = self.drawing_group.findChild(QLabel, "brush_label")
            if brush_label:
                brush_label.setStyleSheet(self.label_style)
            else:
                # 如果通过对象名找不到，尝试通过父控件查找
                for child in self.drawing_group.children():
                    if isinstance(child, QLabel) and child.text() == "画笔粗细(px):":
                        child.setStyleSheet(self.label_style)
                        break

            depth_label = self.drawing_group.findChild(QLabel, "depth_label")
            if depth_label:
                depth_label.setStyleSheet(self.label_style)
            else:
                # 如果通过对象名找不到，尝试通过父控件查找
                for child in self.drawing_group.children():
                    if isinstance(child, QLabel) and child.text() == "画笔深度:":
                        child.setStyleSheet(self.label_style)
                        break

        else:
            # 禁用绘图设置控件
            self.brush_size_spin.setEnabled(False)
            self.brush_size_spin.setStyleSheet(self.disabled_widget_style)

            self.depth_slider.setEnabled(False)
            self.depth_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    border: 1px solid #999999;
                    height: 8px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FFFFFF, stop:1 #000000);
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #95a5a6;
                    border: 1px solid #7f8c8d;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 9px;
                }
            """)

            self.eraser_button.setEnabled(False)
            self.eraser_button.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: #cccccc;
                    border: none;
                    padding: 4px 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    min-height: 32px;
                }
            """)

            self.clear_canvas_button.setEnabled(False)
            self.clear_canvas_button.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: #cccccc;
                    border: none;
                    padding: 4px 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    min-height: 32px;
                }
            """)

            self.apply_button.setEnabled(False)
            self.apply_button.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: #cccccc;
                    border: none;
                    padding: 6px 8px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                    min-height: 36px;
                }
            """)

            # 禁用绘图设置组
            self.drawing_group.setStyleSheet(self.disabled_groupbox_style)

            # ==== 关键修改：修复标签查找和样式更新 ====
            # 禁用状态的标签样式
            brush_label = self.drawing_group.findChild(QLabel, "brush_label")
            if brush_label:
                brush_label.setStyleSheet(self.disabled_label_style)
            else:
                # 如果通过对象名找不到，尝试通过父控件查找
                for child in self.drawing_group.children():
                    if isinstance(child, QLabel) and child.text() == "画笔粗细(px):":
                        child.setStyleSheet(self.disabled_label_style)
                        break

            depth_label = self.drawing_group.findChild(QLabel, "depth_label")
            if depth_label:
                depth_label.setStyleSheet(self.disabled_label_style)
            else:
                # 如果通过对象名找不到，尝试通过父控件查找
                for child in self.drawing_group.children():
                    if isinstance(child, QLabel) and child.text() == "画笔深度:":
                        child.setStyleSheet(self.disabled_label_style)
                        break

    def get_parameters(self):
        """获取参数"""
        return self.params

    def set_parameters(self, parameters):
        """设置参数"""
        self.params = parameters
        self.update_ui_from_parameters()

    def resizeEvent(self, event):
        """重设大小事件"""
        super().resizeEvent(event)
        self.apply_fixed_styles()

    def apply_fixed_styles(self):
        """应用固定样式"""
        # 应用样式到所有控件
        self.update_defect_controls_visibility()
