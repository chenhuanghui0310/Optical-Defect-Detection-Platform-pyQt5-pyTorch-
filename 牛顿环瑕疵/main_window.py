import sys
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer

from parameters import SimulationParameters
from control_panel import ControlPanel
from simulation_widgets import ImageContainer
import numpy as np

class DefectInterferenceWindow(QMainWindow):
    """主窗口 - 手动画板保持2:1比例版本"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("牛顿环瑕疵图像仿真系统")

        # 设置最小尺寸
        self.setMinimumSize(2200, 1200)

        # 获取屏幕尺寸并设置合适的初始大小
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        initial_width = min(1400, screen_geometry.width() - 100)
        initial_height = min(900, screen_geometry.height() - 100)
        self.resize(initial_width, initial_height)

        # 居中显示窗口
        self.move((screen_geometry.width() - initial_width) // 2,
                  (screen_geometry.height() - initial_height) // 2)

        self.setStyleSheet("background-color: #FFFFFF;")

        # 设置窗口图标
        try:
            icon = QtGui.QIcon('image/logo.ico')
            self.setWindowIcon(icon)
        except:
            pass

        # 初始化参数
        self.params = SimulationParameters()

        # 当前缩放因子
        self.scale_factor = 1.0
        self.last_size = self.size()
        self.canvas_state = None
        self.is_fullscreen_transition = False

        # 初始化UI
        self.init_ui()

        # 连接信号
        self.connect_signals()

        # 状态栏
        self.statusBar().showMessage("就绪 - 手动画板保持2:1比例")

        # 初始化显示
        self.control_panel.update_ui_from_parameters()
        self.update_simulation()

        # 设置定时器检测窗口大小变化
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.handle_resize)

    def init_ui(self):
        """初始化用户界面 - 固定间距"""
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #FFFFFF;")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # 固定边距
        main_layout.setSpacing(10)  # 固定间距

        # 标题
        title_label = QLabel("牛顿环瑕疵图像仿真系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("title_label")
        title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 28px;
                font-family: 宋体;
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # 创建主布局
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)  # 固定间距
        main_layout.addLayout(content_layout, 1)

        # 左侧控制面板
        self.control_panel = ControlPanel(self.params)
        content_layout.addWidget(self.control_panel)

        # 右侧图像显示区域
        self.image_container = ImageContainer(self.params)
        content_layout.addWidget(self.image_container, 1)

    def changeEvent(self, event):
        """处理窗口状态变化（包括全屏切换）"""
        if event.type() == QtCore.QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowFullScreen:
                self.is_fullscreen_transition = True
                self.save_canvas_state()
                QTimer.singleShot(150, self.delayed_fullscreen_update)
            elif self.is_fullscreen_transition:
                self.is_fullscreen_transition = False
                self.save_canvas_state()
                QTimer.singleShot(150, self.delayed_fullscreen_update)
        super().changeEvent(event)

    def delayed_fullscreen_update(self):
        """延迟更新界面，避免全屏切换时的竞争条件"""
        try:
            self.resize_timer.stop()
            self.update_scale_factor()
            self.safe_update_stylesheet()
            self.restore_canvas_state()
            self.sync_canvas_with_spectrum()
            self.resize_timer.start(100)
            self.is_fullscreen_transition = False
        except Exception as e:
            print(f"全屏切换处理错误: {e}")
            self.is_fullscreen_transition = False

    def resizeEvent(self, event):
        """重写窗口大小改变事件"""
        super().resizeEvent(event)
        if not self.is_fullscreen_transition:
            self.resize_timer.start(100)

    def handle_resize(self):
        """处理窗口大小变化"""
        try:
            current_size = self.size()
            if current_size != self.last_size:
                self.last_size = current_size
                self.update_scale_factor()
                self.safe_update_stylesheet()
                self.sync_canvas_with_spectrum()
        except Exception as e:
            print(f"处理窗口大小变化错误: {e}")

    def update_scale_factor(self):
        """根据窗口大小计算缩放因子"""
        try:
            base_width = 1200
            current_width = self.width()
            self.scale_factor = max(0.7, min(1.5, current_width / base_width))
        except Exception as e:
            print(f"更新缩放因子错误: {e}")
            self.scale_factor = 1.0

    def safe_update_stylesheet(self):
        """安全更新样式表 - 固定值"""
        try:
            title_style = """
                #title_label {
                    font-weight: bold;
                    font-size: 28px;
                    font-family: 宋体;
                    color: #2c3e50;
                    margin: 10px;
                    padding: 10px;
                }
            """

            title_label = self.findChild(QLabel, "title_label")
            if title_label:
                title_label.setStyleSheet(title_style)

            # 更新图像容器的样式
            if hasattr(self, 'image_container'):
                self.image_container.update_stylesheet(self.scale_factor)

        except Exception as e:
            print(f"安全更新样式表错误: {e}")

    def sync_canvas_with_spectrum(self):
        """同步画板与光谱图尺寸"""
        try:
            if hasattr(self, 'image_container'):
                simulation_tab = self.image_container.get_simulation_tab()
                if simulation_tab and hasattr(simulation_tab, 'sync_canvas_size'):
                    simulation_tab.sync_canvas_size()

        except Exception as e:
            print(f"同步画板尺寸错误: {e}")

    def save_canvas_state(self):
        """保存画板状态"""
        try:
            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas:
                self.canvas_state = drawing_canvas.save_canvas_state()
        except Exception as e:
            print(f"保存画板状态错误: {e}")
            self.canvas_state = None

    def restore_canvas_state(self):
        """恢复画板状态"""
        try:
            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas and self.canvas_state:
                drawing_canvas.restore_canvas_state(self.canvas_state)
        except Exception as e:
            print(f"恢复画板状态错误: {e}")

    def connect_signals(self):
        """连接信号"""
        try:
            self.control_panel.connect_signals(
                update_simulation_slot=self.update_simulation,
                reset_parameters_slot=self.reset_parameters,
                save_images_slot=self.save_images,
                defect_type_changed_slot=self.defect_type_changed,
                on_brush_size_changed_slot=self.on_brush_size_changed,
                on_brush_depth_changed_slot=self.on_brush_depth_changed,
                on_eraser_mode_changed_slot=self.on_eraser_mode_changed,
                clear_drawing_canvas_slot=self.clear_drawing_canvas,
                apply_manual_scratches_slot=self.apply_manual_scratches,
                bump_parameters_changed_slot=self.bump_parameters_changed,
                scratch_type_changed_slot=self.scratch_type_changed
            )
        except Exception as e:
            print(f"连接信号错误: {e}")

    def defect_type_changed(self, text):
        """瑕疵类型改变时的处理"""
        try:
            self.params.defect_type = text
            self.control_panel.update_defect_controls_visibility()

            # 更新手动画板的启用状态
            if self.params.defect_type == "表面划痕":
                # 启用手动画板
                self.image_container.set_drawing_widget_enabled(True)
                # 自动切换到手动划痕模式
                self.params.use_manual_scratches = True
                # 同步手动划痕类型与划痕类型
                self.params.manual_scratch_type = self.params.scratch_type

                self.statusBar().showMessage("表面划痕模式已启用 - 手动画板可用")
            elif self.params.defect_type == "头发丝遮挡":  # 新增
                # 禁用手动画板
                self.image_container.set_drawing_widget_enabled(False)
                self.params.use_manual_scratches = False

                self.statusBar().showMessage("头发丝遮挡模式已启用 - 手动画板已禁用")
            else:
                # 禁用手动画板
                self.image_container.set_drawing_widget_enabled(False)
                self.params.manual_scratch_mask = None
                self.params.use_manual_scratches = False

                if self.params.defect_type in ["小突起", "小凹陷"]:
                    bump_type = "突起" if self.params.defect_type == "小突起" else "凹陷"
                    min_effect, max_effect = self.params.get_bump_effect_height_range()
                    self.statusBar().showMessage(
                        f"{bump_type}模式已启用 - 手动画板已禁用")
                else:
                    self.statusBar().showMessage(f"已选择{self.params.defect_type}瑕疵模式 - 手动画板已禁用")

            self.update_simulation()

        except Exception as e:
            print(f"瑕疵类型改变处理错误: {e}")

    def scratch_type_changed(self, text):
        """划痕类型改变时的处理"""
        try:
            self.params.scratch_type = text
            if self.params.defect_type == "表面划痕":
                # 如果当前是表面划痕模式，同步手动划痕类型
                self.params.manual_scratch_type = text
                self.update_simulation()

                scratch_type_desc = "凹陷" if text == "凹陷划痕" else "突起"
                depth_range = self.params.get_scratch_depth_range()
                min_depth = abs(depth_range[0]) * 1e6 if scratch_type_desc == "凹陷" else depth_range[0] * 1e6
                max_depth = abs(depth_range[1]) * 1e6 if scratch_type_desc == "凹陷" else depth_range[1] * 1e6

                self.statusBar().showMessage(
                    f"已切换到{scratch_type_desc}划痕模式 - 深度范围: {min_depth:.1f}-{max_depth:.1f}μm")
        except Exception as e:
            print(f"划痕类型改变处理错误: {e}")

    def bump_parameters_changed(self, value):
        """小突起/小凹陷参数变化"""
        try:
            if self.params.defect_type in ["小突起", "小凹陷"]:
                self.params.bump_count = self.control_panel.bump_spin.value()
                self.params.bump_height = self.control_panel.bump_height_spin.value()
                self.params.bump_radius_mm = self.control_panel.bump_radius_spin.value()
                self.params.bump_position_mode = self.control_panel.bump_position_combo.currentText()
                self.update_simulation()

                # 更新状态栏显示详细信息
                position_info = {
                    "随机分布": "随机分布位置",
                    "网格排列": "网格均匀排列",
                    "固定位置": "固定位置(5mm,5mm; 10mm,5mm; 5mm,10mm)"
                }
                mode_info = position_info.get(self.params.bump_position_mode, "未知模式")
                bump_type = "突起" if self.params.defect_type == "小突起" else "凹陷"

                # 计算实际高度范围（考虑放大效应）
                min_effect, max_effect = self.params.get_bump_effect_height_range()

                self.statusBar().showMessage(
                    f"{bump_type}参数已更新: 数量{self.params.bump_count}个, "
                    f"高度{self.params.bump_height}μm, 半径{self.params.bump_radius_mm}mm, "
                    f"效应高度{min_effect:.1f}-{max_effect:.1f}μm"
                )
        except Exception as e:
            print(f"小突起/小凹陷参数变化处理错误: {e}")

    def on_brush_size_changed(self, value):
        """画笔粗细改变"""
        try:
            self.params.brush_size = value
            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas:
                # 保持原有的缩放逻辑，但范围现在是 5-30
                scaled_size = max(1, int(value * self.scale_factor))
                drawing_canvas.set_brush_size(scaled_size)

                # 获取当前的画笔剖面信息
                from drawing_canvas import DrawingCanvas
                brush_info = f"画笔粗细: {value}像素"

                # 添加剖面信息
                if self.params.defect_type == "表面划痕":
                    profile = drawing_canvas._create_half_circle_brush_profile(value, self.params.brush_depth)
                    profile_size = profile.shape[0]
                    max_depth = self.params.brush_depth

                    brush_info += f" | 剖面尺寸: {profile_size}×{profile_size}"
                    brush_info += f" | 最大深度: {max_depth:.2f}"

                    if self.params.scratch_type == "凹陷划痕":
                        depth_range = self.params.get_manual_scratch_depth_range()
                        max_depth_um = abs(depth_range[0]) * 1e6
                        actual_depth = max_depth_um * max_depth
                        brush_info += f" | 实际凹陷深度: {actual_depth:.1f}μm"
                    else:
                        depth_range = self.params.get_manual_scratch_depth_range()
                        max_depth_um = depth_range[1] * 1e6
                        actual_depth = max_depth_um * max_depth
                        brush_info += f" | 实际突起高度: {actual_depth:.1f}μm"

                self.statusBar().showMessage(brush_info)
        except Exception as e:
            print(f"画笔粗细改变处理错误: {e}")

    def on_brush_depth_changed(self, value):
        """画笔深度改变"""
        try:
            # 滑块值 (0-100) -> 归一化深度系数 (0.0-1.0)
            # 这里保持原有的映射：100% 对应深度系数1.0
            normalized_depth = value / 100.0
            self.params.brush_depth = normalized_depth  # 范围 0.0-1.0

            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas:
                # 传递给画布使用
                drawing_canvas.set_brush_depth(self.params.brush_depth)

            # 更新深度显示标签，显示百分比
            if hasattr(self.control_panel, 'depth_value_label'):
                self.control_panel.depth_value_label.setText(f"{value}%")

            # 提供深度信息
            if self.params.defect_type == "表面划痕" and self.params.use_manual_scratches:
                scratch_type_desc = "凹陷" if self.params.scratch_type == "凹陷划痕" else "突起"
                # 计算物理深度：归一化系数 × 1.0μm
                physical_depth = normalized_depth  # 单位：μm
                depth_info = f"{scratch_type_desc}划痕深度: {physical_depth:.2f}μm (对应滑块{value}%)"
                self.statusBar().showMessage(depth_info)

        except Exception as e:
            print(f"画笔深度改变处理错误: {e}")

    def on_eraser_mode_changed(self, checked):
        """橡皮擦模式改变"""
        try:
            self.params.eraser_mode = checked
            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas:
                drawing_canvas.set_eraser_mode(checked)

            if checked:
                self.statusBar().showMessage("橡皮擦模式已启用 - 可以擦除已绘制的划痕")
            else:
                if self.params.defect_type == "表面划痕" and self.params.use_manual_scratches:
                    scratch_type_desc = "凹陷" if self.params.scratch_type == "凹陷划痕" else "突起"
                    depth_range = self.params.get_manual_scratch_depth_range()

                    # 修正：凹陷深度显示绝对值
                    if scratch_type_desc == "凹陷":
                        min_depth_abs = abs(depth_range[0]) * 1e6
                        max_depth_abs = abs(depth_range[1]) * 1e6
                        self.statusBar().showMessage(
                            f"橡皮擦模式已关闭 - 返回{scratch_type_desc}划痕绘制模式 "
                            f"(深度范围: 0-{min_depth_abs:.1f}μm)"
                        )
                    else:
                        min_depth = depth_range[0] * 1e6
                        max_depth = depth_range[1] * 1e6
                        self.statusBar().showMessage(
                            f"橡皮擦模式已关闭 - 返回{scratch_type_desc}划痕绘制模式 "
                            f"(深度范围: {min_depth:.1f}-{max_depth:.1f}μm)"
                        )
                else:
                    self.statusBar().showMessage("橡皮擦模式已关闭")
        except Exception as e:
            print(f"橡皮擦模式改变处理错误: {e}")

    def clear_drawing_canvas(self):
        """清空手动画板"""
        try:
            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas:
                drawing_canvas.clear_canvas()
                self.params.manual_scratch_mask = None
                self.update_simulation()
                self.statusBar().showMessage("手动画板已清空")
        except Exception as e:
            print(f"清空手动画板错误: {e}")

    def apply_manual_scratches(self):
        """应用手动绘制的划痕"""
        try:
            if self.params.defect_type == "表面划痕":
                drawing_canvas = self.image_container.get_drawing_canvas()
                if drawing_canvas and drawing_canvas.has_drawing():
                    # 获取划痕掩码
                    mask_size = (self.params.simulation_size, self.params.simulation_size)
                    self.params.manual_scratch_mask = drawing_canvas.get_scratches_mask(mask_size)
                    self.params.use_manual_scratches = True
                    self.update_simulation()

                    # 计算划痕覆盖面积
                    mask_sum = np.sum(self.params.manual_scratch_mask > 0.01)
                    total_pixels = self.params.simulation_size * self.params.simulation_size
                    coverage_percent = (mask_sum / total_pixels) * 100

                    scratch_type_desc = "凹陷" if self.params.scratch_type == "凹陷划痕" else "突起"
                    self.statusBar().showMessage(
                        f"手动划痕已应用 - 类型: {scratch_type_desc}, 覆盖面积: {coverage_percent:.1f}%"
                    )
                else:
                    self.params.manual_scratch_mask = None
                    self.params.use_manual_scratches = False
                    self.update_simulation()
                    self.statusBar().showMessage("手动画板为空，已清除手动划痕")
            else:
                self.statusBar().showMessage("只有表面划痕模式下才能应用手动划痕")
        except Exception as e:
            print(f"应用手动划痕错误: {e}")

    def update_simulation(self):
        """更新仿真图像"""
        try:
            # 更新控制面板参数
            self.control_panel.update_parameters_from_ui()

            # 更新图像显示
            self.image_container.update_simulation()

            # 更新状态栏信息
            if self.params.defect_type == "表面划痕" and self.params.use_manual_scratches:
                scratch_type_desc = "凹陷" if self.params.scratch_type == "凹陷划痕" else "突起"
                self.statusBar().showMessage(f"仿真已更新 - 表面划痕模式（{scratch_type_desc}，手动绘制）")
            else:
                self.statusBar().showMessage(f"仿真已更新 - {self.params.defect_type}")
        except Exception as e:
            print(f"更新仿真错误: {e}")
            self.statusBar().showMessage("更新仿真时发生错误")

    def reset_parameters(self):
        """重置所有参数"""
        try:
            # 显示确认对话框
            reply = QMessageBox.question(
                self, '确认重置',
                '确定要重置所有参数到默认值吗？\n这将清除所有当前设置。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # 保存手动画板状态
                drawing_canvas = self.image_container.get_drawing_canvas()
                canvas_image = None
                if drawing_canvas:
                    canvas_image = drawing_canvas.get_canvas_image()

                # 重置参数
                self.params.reset()

                # 恢复手动画板图像
                if drawing_canvas and canvas_image:
                    drawing_canvas.set_canvas_image(canvas_image)

                # 更新UI
                self.control_panel.update_ui_from_parameters()
                self.update_simulation()

                # 禁用手动画板
                self.image_container.set_drawing_widget_enabled(False)

                self.statusBar().showMessage("所有参数已重置到默认值")
        except Exception as e:
            print(f"重置参数错误: {e}")

    def save_images(self):
        """保存当前图像"""
        try:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(
                self, "保存图片", "", "PNG文件 (*.png);;所有文件 (*)", options=options)
            if fileName:
                if not fileName.lower().endswith('.png'):
                    fileName += '.png'

                # 保存仿真图像
                ideal_filename, defect_filename, topography_filename = self.image_container.save_figures(fileName)

                if ideal_filename and topography_filename:
                    # 显示保存成功消息
                    detail_lines = [
                        f"1. 理想干涉图: {ideal_filename}",
                        f"2. 三维光强分布图: {topography_filename}"
                    ]
                    if defect_filename:
                        detail_lines.insert(1, f"2. 有瑕疵干涉图: {defect_filename}")
                        detail_lines[2] = f"3. 三维光强分布图: {topography_filename}"

                    QMessageBox.information(
                        self, "保存成功",
                        "图片已成功保存：\n" + "\n".join(detail_lines),
                        QMessageBox.Ok
                    )

                    saved_items = [ideal_filename, topography_filename]
                    if defect_filename:
                        saved_items.insert(1, defect_filename)
                    self.statusBar().showMessage(f"图片已保存: {', '.join(saved_items)}")
                else:
                    QMessageBox.warning(
                        self, "保存失败",
                        "保存图片失败，请检查文件路径和权限。",
                        QMessageBox.Ok
                    )
        except Exception as e:
            print(f"保存图片错误: {e}")
            self.statusBar().showMessage("保存图片失败")

    def closeEvent(self, event):
        """关闭窗口事件"""
        try:
            # 检查是否有未保存的手动绘制
            drawing_canvas = self.image_container.get_drawing_canvas()
            if drawing_canvas and drawing_canvas.has_drawing() and self.params.defect_type == "表面划痕":
                reply = QMessageBox.question(
                    self, '确认关闭',
                    '手动画板上有未应用的绘制内容，确定要关闭吗？',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.No:
                    event.ignore()
                    return

            event.accept()
        except Exception as e:
            print(f"关闭窗口错误: {e}")
            event.accept()

    def get_scale_factor(self):
        """获取缩放因子"""
        return self.scale_factor

    def set_scale_factor(self, scale_factor):
        """设置缩放因子"""
        self.scale_factor = max(0.5, min(2.0, scale_factor))
        self.safe_update_stylesheet()
        self.sync_canvas_with_spectrum()

    def get_current_parameters(self):
        """获取当前参数"""
        return self.params

    def update_control_panel_visibility(self):
        """更新控制面板可见性"""
        if hasattr(self.control_panel, 'update_defect_controls_visibility'):
            self.control_panel.update_defect_controls_visibility()

