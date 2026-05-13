import sys
import os
import matplotlib.pyplot as plt

# 设置 Matplotlib 中文字体，解决图表上的中文方块乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# --- 修复 opencv-python 与 PyQt5 的 Qt 插件冲突 ---
import PyQt5
qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QAction, QWidget
from PyQt5.QtCore import Qt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 加载劈尖模块 ---
sys.path.insert(0, os.path.join(BASE_DIR, '劈尖瑕疵0329'))
import main_window as wedge_module
wedge_window_class = wedge_module.DefectInterferenceWindow
# 清除已被加载的关键模块缓存，防止污染
for k in list(sys.modules.keys()):
    if k in ['parameters', 'control_panel', 'simulation_widgets', 'drawing_canvas', 'intensity_plot', 'spectrum_plot', 'main_window', 'defect_inference']:
        del sys.modules[k]
sys.path.pop(0)

# --- 加载牛顿环模块 ---
sys.path.insert(0, os.path.join(BASE_DIR, '牛顿环瑕疵'))
import main_window as newton_module
newton_window_class = newton_module.DefectInterferenceWindow
sys.path.pop(0)
# --------------------

class UnifiedPlatformWindow(QMainWindow):
    """
    统一的多实验瑕疵检测平台主窗口
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("统一光学干涉瑕疵检测平台")
        self.setMinimumSize(1400, 900)

        # 核心：使用 QStackedWidget 切换不同的实验界面
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.wedge_app = wedge_window_class()
        self.newton_app = newton_window_class()
        
        # 移除子窗口边框，使其表现为普通控件
        self.wedge_app.setWindowFlags(Qt.Widget)
        self.newton_app.setWindowFlags(Qt.Widget)

        self.stacked_widget.addWidget(self.wedge_app)
        self.stacked_widget.addWidget(self.newton_app)

        self.create_global_menu()

        # 默认显示劈尖实验
        self.switch_to_wedge()

    def create_global_menu(self):
        menubar = self.menuBar()
        
        # 实验切换菜单
        exp_menu = menubar.addMenu("实验选择")
        
        wedge_action = QAction("劈尖干涉瑕疵检测", self)
        wedge_action.triggered.connect(self.switch_to_wedge)
        exp_menu.addAction(wedge_action)
        
        newton_action = QAction("牛顿环干涉瑕疵检测", self)
        newton_action.triggered.connect(self.switch_to_newton)
        exp_menu.addAction(newton_action)

    def switch_to_wedge(self):
        self.stacked_widget.setCurrentWidget(self.wedge_app)
        self.setWindowTitle("统一光学干涉瑕疵检测平台 - 劈尖干涉")

    def switch_to_newton(self):
        self.stacked_widget.setCurrentWidget(self.newton_app)
        self.setWindowTitle("统一光学干涉瑕疵检测平台 - 牛顿环干涉")

def main():
    app = QApplication(sys.path)
    app.setStyle("Fusion")
    
    window = UnifiedPlatformWindow()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
