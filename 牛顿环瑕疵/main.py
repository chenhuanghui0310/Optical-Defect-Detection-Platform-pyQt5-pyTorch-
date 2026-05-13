import sys
import matplotlib
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = False

from main_window import DefectInterferenceWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序的高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = DefectInterferenceWindow()
    window.show()
    sys.exit(app.exec_())