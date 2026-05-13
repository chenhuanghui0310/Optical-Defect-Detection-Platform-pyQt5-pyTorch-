import sys
import os

import matplotlib
from PyQt5.QtWidgets import QApplication  # type: ignore
from PyQt5.QtCore import Qt  # type: ignore

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = False

from OpticalDefectExperiment.main_window import DefectInterferenceWindow


def _setup_qt_platform_plugin_path() -> None:
    """兼容中文路径/虚拟环境，避免 qwindows 插件找不到。"""
    venv_base = os.path.dirname(os.path.dirname(sys.executable))
    plugin_path = os.path.join(venv_base, "Lib", "site-packages", "PyQt5", "Qt5", "plugins", "platforms")
    if os.path.exists(plugin_path):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
        return

    fallback = os.path.join(venv_base, "Lib", "site-packages", "PyQt5", "Qt", "plugins", "platforms")
    if os.path.exists(fallback):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = fallback


def main() -> int:
    _setup_qt_platform_plugin_path()

    app = QApplication(sys.argv)

    window = DefectInterferenceWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())