# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false

"""
虚拟实验平台启动入口（根目录启动文件）。

用法：
    python virtual_main.py
"""

import io
import os
import subprocess
import sys

from VirtualExperiment.utils import resource_path

from PyQt5.QtWidgets import (  # type: ignore
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStackedWidget,
    QScrollArea,
    QStatusBar,
    QMessageBox,
    QDialog,
    QListWidget,
    QHBoxLayout,
    QPushButton,
    QAction,
    QFrame,
    QShortcut,
)
from PyQt5 import QtCore  # type: ignore
from PyQt5.QtCore import Qt, QEvent, QObject, QTimer  # type: ignore
from PyQt5.QtGui import QFont, QDesktopServices, QIcon, QKeySequence  # type: ignore

# 视频播放
from VirtualExperiment.video_window import VideoWindow
# 实验演示
from VirtualExperiment.newton_rings_window import NewtonRingsApp
from VirtualExperiment.michelson_window import CombinedInterface
from VirtualExperiment.wedge_window import PolarizationSimulator
from VirtualExperiment.experiment_window import ExperimentWindow
# 数据处理
from VirtualExperiment.data_work import PolarizationExperimentApp
# 思考拓展
from VirtualExperiment.extended_reflection_window import ThinkingExtensionApp


def _ask_quit_and_maybe_exit(app, parent_widget):
    reply = QMessageBox.question(
        parent_widget, "退出", "是否退出程序？",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    )
    if reply == QMessageBox.Yes:
        app.quit()


class _EscExitFilter(QObject):
    """拦截 Esc：弹出退出确认框，并阻止默认行为（如退出全屏）。"""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app

    def eventFilter(self, obj, event):
        is_esc = (
            event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape
        ) or (
            event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Escape
        )
        if is_esc:
            parent = self._app.activeWindow() if hasattr(self._app, "activeWindow") else None
            _ask_quit_and_maybe_exit(self._app, parent)
            event.accept()
            return True
        return super().eventFilter(obj, event)


class _FullScreenEnforcer(QObject):
    """强制保持全屏，禁止退出全屏/缩放窗口。"""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.WindowStateChange:
            try:
                if hasattr(obj, "isVisible") and obj.isVisible() and hasattr(obj, "isFullScreen"):
                    if not obj.isFullScreen() and hasattr(obj, "showFullScreen"):
                        QTimer.singleShot(0, obj.showFullScreen)
            except Exception:
                pass
        return super().eventFilter(obj, event)


# 修复中文路径问题
if getattr(sys, 'frozen', False):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']

    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()
else:
    try:
        if sys.stdout is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if sys.stderr is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("波动光学虚拟实验平台")
        self.setGeometry(100, 100, 1600, 900)

        # 设置应用程序图标
        self.setWindowIcon(QIcon(resource_path('images/app.ico')))

        # 初始化实验模块
        self.experiments = {
            '光的双缝干涉实验': ExperimentWindow(),
            '凸透镜曲率半径的测量': NewtonRingsApp(),
            '迈克尔逊干涉仪测量光波长': CombinedInterface(),
            '光的偏振实验': PolarizationSimulator(),
        }

        # 初始化视频窗口
        self.video_windows = {}

        self.initUI()

    def initUI(self):
        # 用滚动区域承载全部内容：保证全屏下不被截断（超出屏幕可滚动）
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.central_widget = QWidget()
        self._scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(self._scroll_area)

        # 主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # 创建菜单栏
        self.createMenuBar()

        # 创建状态栏
        self.createStatusBar()

        # 创建实验堆叠窗口
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget, 1)  # 添加伸缩因子

        # 创建主界面（欢迎页面）
        self.createWelcomePage()

        # 添加各实验页面
        for name, widget in self.experiments.items():
            self.stacked_widget.addWidget(widget)

        # 默认显示欢迎页面
        self.stacked_widget.setCurrentIndex(0)

    def createWelcomePage(self):
        """创建欢迎页面"""
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setContentsMargins(50, 50, 50, 50)
        welcome_layout.setSpacing(40)

        # 标题
        title_label = QLabel("波动光学虚拟实验平台")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #1E3A5F;
                font-size: 48px;
                font-weight: bold;
                letter-spacing: 4px;
            }
        """)
        welcome_layout.addWidget(title_label)

        # 副标题
        subtitle_label = QLabel("波动光学实验教学与研究的创新平台")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #2A4A7F;
                font-size: 22px;
                font-weight: normal;
                margin-top: -20px;
            }
        """)
        welcome_layout.addWidget(subtitle_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #5A8EC0;")
        separator.setFixedHeight(2)
        welcome_layout.addWidget(separator)

        # 平台介绍
        intro_label = QLabel(
            "<p style='text-align:center; font-size: 24px; color: #333333; line-height: 1.8;'>"
            "欢迎使用波动光学虚拟实验平台！本平台提供先进的波动光学实验虚拟环境，<br>"
            "通过计算机技术模拟重现真实实验场景，帮助学生理解光学原理、掌握实验技能。<br>"
            "平台包含以下四个核心波动光学实验：</p>"
            "<ul style='margin-left: 100px; font-size: 28px; color: #1E3A5F; line-height: 1.6;'>"
            "<li><b>光的双缝干涉实验</b> - 研究光的干涉与双缝间距测量</li>"
            "<li><b>凸透镜曲率半径的测量</b> - 牛顿环干涉法测量凸透镜曲率半径</li>"
            "<li><b>迈克尔逊干涉仪测量光波长</b> - 利用等倾干涉条纹变化测量光波长</li>"
            "<li><b>光的偏振实验</b> - 研究偏振光特性与马吕斯定律验证</li>"
            "</ul>"
            "<p style='text-align:center; font-size: 24px; color: #333333; line-height: 1.8;'>"
            "<b>平台主要功能：</b><br>"
            "1. 虚拟实验操作 - 高度仿真的实验仪器和操作界面<br>"
            "2. 实验预习与理论学习 - 提供详细的实验原理和预习材料<br>"
            "3. 实验动画演示 - 3D动画呈现实验仪器和实验操作<br>"
            "4. 实验报告生成 - 自动生成规范的实验报告<br>"
            "5. 数据采集与分析 - 自动采集实验数据并提供分析工具</p>"
            "<p style='text-align:center; font-size: 24px; color: #1E3A5F; font-weight: bold; line-height: 1.8;'>"
            "请从顶部菜单栏选择一个实验开始您的探索之旅！</p>"
        )
        intro_label.setAlignment(Qt.AlignCenter)
        intro_label.setWordWrap(True)
        welcome_layout.addWidget(intro_label, 1)

        # 底部信息
        footer_label = QLabel("© 2025 波动光学虚拟实验平台 | 版本 1.0.0 | 开发团队：XXXX实验室")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: #666666; font-size: 18px; margin-top: 20px;")
        welcome_layout.addWidget(footer_label)

        # 添加到堆叠窗口
        self.stacked_widget.addWidget(welcome_widget)

    def createMenuBar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 设置菜单栏样式为蓝底白字
        menubar.setStyleSheet("""
                QMenuBar {
                    background-color: #4682B4;
                    color: white;
                    padding: 5px;
                    border-bottom: 1px solid #3A6EA0;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 5px 15px;
                    border-radius: 4px;
                }
                QMenuBar::item:selected {
                    background-color: #5A8EC0;
                }
                QMenuBar::item:pressed {
                    background-color: #3A6EA0;
                }
            """)

        # 实验选择菜单
        experiment_menu = menubar.addMenu('选择实验')
        for name in self.experiments.keys():
            action = experiment_menu.addAction(name)
            action.triggered.connect(lambda _, n=name: self.switch_experiment(n))

        # 预习报告菜单
        report_menu = menubar.addMenu('预习报告')
        for name in self.experiments.keys():
            report_menu.addAction(name).triggered.connect(lambda _, n=name: self.open_pdf(n))

        # 实验动画菜单
        video_menu = menubar.addMenu('实验动画')
        for name in self.experiments.keys():
            video_menu.addAction(name).triggered.connect(lambda _, n=name: self.open_video_window(n))

        # 查看报告菜单
        report_menu = menubar.addMenu('查看报告')
        view_report_action = QAction('打开报告目录', self)
        view_report_action.triggered.connect(self.view_historical_pdf)
        report_menu.addAction(view_report_action)

        # 数据处理菜单
        data_software_menu = menubar.addMenu('数据分析')
        data_processing_action = data_software_menu.addAction('数据处理')
        data_processing_action.triggered.connect(self.run_data_work)

        guide = data_software_menu.addAction('操作指引')
        guide.triggered.connect(lambda: self.open_guide_pdf("docts/data_word.pdf"))

        # 操作指引菜单
        guide_menu = menubar.addMenu('操作指引')

        newton_rings_guide = guide_menu.addAction('凸透镜曲率半径的测量')
        newton_rings_guide.triggered.connect(
            lambda: self.open_guide_pdf("docts/newton_rings_operating_instructions.pdf"))

        michelson_guide = guide_menu.addAction('迈克尔逊干涉仪测量光波长')
        michelson_guide.triggered.connect(
            lambda: self.open_guide_pdf("docts/michelson_operating_instructions.pdf"))

        polarization_guide = guide_menu.addAction('光的偏振实验')
        polarization_guide.triggered.connect(
            lambda: self.open_guide_pdf("docts/wedge_operating_instructions.pdf"))

        experiment_guide = guide_menu.addAction('光的双缝干涉实验')
        experiment_guide.triggered.connect(
            lambda: self.open_guide_pdf("docts/experiment_operating_instructions.pdf"))

        # 思考拓展菜单
        extend_menu = menubar.addMenu('思考拓展')
        extend_action = extend_menu.addAction('思考拓展')
        extend_action.triggered.connect(self.run_extend)

        # ================= 帮助菜单 =================
        help_menu = menubar.addMenu('帮助')
        help_menu.addAction('帮助', lambda: QMessageBox.information(
            self,
            "帮助",
            "<h3>使用说明</h3>"
            "<p>1. 本平台包含四个波动光学实验和数据分析工具。</p>"
            "<p>2. 使用前请先阅读'操作指引'。</p>"
            "<p>更多详情请阅读随软件附带的用户手册。</p>"
        ))
        help_menu.addAction('关于项目', lambda: QMessageBox.about(
            self,
            "关于项目",
            "<h3>波动光学虚拟实验平台</h3>"
            "<p>版本：v1.0.0</p>"
            "<p>版权所有：XX大学XX学院</p>"
            "<p>作者：  XXX<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;XXX<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;XXX<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;XXX<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;XXX</p>"
            "<p>指导老师：XXX</p>"
            "<p>使用反馈：xxxx@xxxx</p>"
            "<p>本软件基于 PyQt5 + Python 3 开发</p>"
            "<p>实验动画用Solid Works和C4D制作</p>"
            "<p>源代码遵循 MIT 许可证，托管于 GitHub</p>"
        ))

    def createStatusBar(self):
        """创建状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # 设置状态栏样式为蓝底白字
        self.statusBar.setStyleSheet("""
                QStatusBar {
                    background-color: #3A6EA0;
                    color: white;
                    padding: 5px;
                    border-top: 1px solid #4682B4;
                }
            """)

        # 添加状态栏文本
        status_label = QLabel("波动光学虚拟实验平台")
        status_label.setFont(QFont("微软雅黑", 10))
        self.statusBar.addPermanentWidget(status_label)

    def switch_experiment(self, name):
        """切换实验界面"""
        if name in self.experiments:
            index = list(self.experiments.keys()).index(name) + 1  # +1因为第一个是欢迎页
            self.stacked_widget.setCurrentIndex(index)
            self.setWindowTitle(f"波动光学虚拟实验平台 - {name}")

    def open_pdf(self, name):
        pdf_path = resource_path(f"docts/{name}预习报告.pdf")
        if os.path.exists(pdf_path):
            QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(pdf_path))
        else:
            print(f"{name}预习报告PDF文件未找到！")

    def open_guide_pdf(self, filename):
        """通用PDF打开方法（跨平台）"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pdf_path = os.path.join(current_dir, filename)

            if os.path.exists(pdf_path):
                if sys.platform == 'win32':
                    os.startfile(pdf_path)
                elif sys.platform == 'darwin':
                    subprocess.run(['open', pdf_path], check=False)
                else:
                    subprocess.run(['xdg-open', pdf_path], check=False)
            else:
                QMessageBox.warning(self, "文件未找到", f"找不到PDF文件:\n{pdf_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开PDF失败:\n{str(e)}")

    def open_video_window(self, name):
        """打开实验动画窗口"""
        video_map = {
            # 菜单实验名 -> resources/videos 内的视频文件名(不含后缀)
            "光的偏振实验": "光的偏振",
            "迈克尔逊干涉仪测量光波长": "迈克尔逊干涉",
            "凸透镜曲率半径的测量": "牛顿环干涉",
            "光的双缝干涉实验": "双缝衍射",
        }
        video_stem = video_map.get(name, name)
        if name not in self.video_windows:
            self.video_windows[name] = VideoWindow(video_stem, display_name=name)
        self.video_windows[name].show()

    def view_historical_pdf(self):
        """查看历史生成的PDF文档 - 先选择实验项目"""
        experiment_dirs = {
            '凸透镜曲率半径的测量': "reports_pdf/newton_rings_pdf",
            '迈克尔逊干涉仪测量光波长': "reports_pdf/michelson_pdf",
            '光的偏振实验': "reports_pdf/wedge_pdf",
            '光的双缝干涉实验': "reports_pdf/experiment_pdf"
        }

        project_dialog = QDialog(self)
        project_dialog.setWindowTitle("选择实验项目")
        project_dialog.setFixedSize(300, 200)

        project_layout = QVBoxLayout(project_dialog)
        project_layout.addWidget(QLabel("请选择要查看的实验项目:"))

        project_list = QListWidget()
        project_list.addItems(experiment_dirs.keys())
        project_layout.addWidget(project_list)

        btn_layout = QHBoxLayout()
        select_btn = QPushButton("选择")
        select_btn.clicked.connect(project_dialog.accept)
        btn_layout.addWidget(select_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(project_dialog.reject)
        btn_layout.addWidget(cancel_btn)
        project_layout.addLayout(btn_layout)

        if project_dialog.exec_() != QDialog.Accepted or not project_list.currentItem():
            return

        selected_project = project_list.currentItem().text()
        pdf_dir = experiment_dirs[selected_project]
        pdf_dir = os.path.join(os.path.dirname(__file__), pdf_dir)

        if not os.path.exists(pdf_dir):
            QMessageBox.information(self, "提示", f"尚未生成{selected_project}的PDF报告")
            return

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            QMessageBox.information(self, "提示", "目录中没有PDF文件")
            return

        file_dialog = QDialog(self)
        file_dialog.setWindowTitle(f"选择{selected_project}的PDF文档")
        file_dialog.setMinimumWidth(600)

        file_layout = QVBoxLayout(file_dialog)
        file_layout.addWidget(QLabel(f"请选择{selected_project}的PDF文档:"))

        pdf_list = QListWidget()
        pdf_list.addItems(sorted(pdf_files, reverse=True))
        file_layout.addWidget(pdf_list)

        file_btn_layout = QHBoxLayout()
        view_btn = QPushButton("查看")
        view_btn.clicked.connect(lambda: self.open_pdf(os.path.join(pdf_dir, pdf_list.currentItem().text())))
        file_btn_layout.addWidget(view_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(file_dialog.close)
        file_btn_layout.addWidget(cancel_btn)
        file_layout.addLayout(file_btn_layout)
        file_dialog.exec_()

    def run_data_work(self):
        """打开数据处理界面"""
        self.data_word_window = PolarizationExperimentApp(self)
        self.data_word_window.show()

    def run_extend(self):
        """打开思考拓展界面"""
        self.extend_window = ThinkingExtensionApp(self)
        self.extend_window.show()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()

    # Esc 退出确认 + 强制全屏（取消可缩放/退出全屏）
    window.installEventFilter(_EscExitFilter(app, window))
    window.installEventFilter(_FullScreenEnforcer(window))
    app.installEventFilter(_EscExitFilter(app))
    app.installEventFilter(_FullScreenEnforcer())
    _esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), window)
    _esc_shortcut.setContext(Qt.ApplicationShortcut)
    _esc_shortcut.activated.connect(lambda: _ask_quit_and_maybe_exit(app, app.activeWindow()))

    # window.showFullScreen()
    window.showMaximized()  # 程序启动即最大化，允许用户调整窗口大小，但不强制全屏
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
