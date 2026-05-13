#!/usr/bin/env python
# pyright: reportMissingImports=false

import sys
import warnings

# 屏蔽 PyQt5/SIP 的 sipPyTypeDict 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")
# 屏蔽 matplotlib 打开图形过多的运行时警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*More than 20 figures.*")

from PyQt5.QtCore import Qt, QEvent, QObject, QTimer, qInstallMessageHandler  # type: ignore
from PyQt5.QtGui import QKeySequence  # type: ignore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QShortcut, QScrollArea, QFrame  # type: ignore

# 屏蔽 Qt 输出的"Unknown property writing-mode"及"QLayout"重复添加警告
_SUPPRESSED_QT_MSGS = (
    "Unknown property writing-mode",
    "QLayout: Attempting to add QLayout",
)


def _qt_message_handler(msg_type, context, message):
    if any(s in message for s in _SUPPRESSED_QT_MSGS):
        return


qInstallMessageHandler(_qt_message_handler)


from SimulationExperiment.ProjectIntroduction import (
    Ui_IntroductionWindow,
    Ui_LoginWindow,
    Ui_SelectionWindow,
)


def _ask_quit_and_maybe_exit(app, parent_widget):
    """弹出“是否退出”确认框，选是则退出程序。"""
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
            return True  # 已处理，不继续传递
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


def _install_esc_exit(app, window):
    """为顶层窗口安装 Esc 退出确认（事件过滤器）。"""
    esc_filter = _EscExitFilter(app, window)
    window.installEventFilter(esc_filter)


def _switch_window(from_win, to_win):
    """页面跳转：始终以全屏显示，不保留可拖拽/调整大小的窗口状态。"""
    from_win.hide()
    to_win.showFullScreen()


def _wrap_mainwindow_with_scroll(win: QMainWindow):
    """把 QMainWindow 的 centralWidget 包到 QScrollArea，避免底部被硬截断。"""
    try:
        cw = win.centralWidget()
        if cw is None or isinstance(cw, QScrollArea):
            return
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(cw)
        win.setCentralWidget(scroll)
    except Exception:
        return


from SimulationExperiment.DoubleSlitInterference_1 import InterferenceFringeCalculator
from SimulationExperiment.DoubleSlitDiffraction_2 import InterferenceFringeCalculator_2
from SimulationExperiment.NewtonRingsInterference_3 import Ui_NewtonWindow
from SimulationExperiment.SingleSlitDiffraction_4 import Ui_SingleWindow
from SimulationExperiment.GratingDiffraction_5 import Ui_GratingWindow
from SimulationExperiment.WedgeInterference_6 import Ui_WedgeWindow
from SimulationExperiment.MichelsonInterferometer_7 import Ui_MicheIsonWindow
from SimulationExperiment.CircularApertureDiffraction_8 import Ui_CircularWindow
from SimulationExperiment.Polarization_9 import Ui_PolarizationWindow


def main() -> int:
    app = QApplication(sys.argv)

    # 登录界面
    login_window = QMainWindow()
    ui_login = Ui_LoginWindow()
    ui_login.setupUi(login_window)
    _wrap_mainwindow_with_scroll(login_window)

    # 项目介绍界面
    introduction_window = QMainWindow()
    ui_introduction = Ui_IntroductionWindow()
    ui_introduction.setupUi(introduction_window)
    _wrap_mainwindow_with_scroll(introduction_window)

    # 实验选择界面
    selection_window = QMainWindow()
    ui_selection = Ui_SelectionWindow()
    ui_selection.setupUi(selection_window)
    _wrap_mainwindow_with_scroll(selection_window)

    # 1-双缝干涉
    ui_double_slit = InterferenceFringeCalculator()
    ui_selection.switch_to_double_slit.connect(lambda: _switch_window(selection_window, ui_double_slit))
    ui_double_slit.return_button.clicked.connect(lambda: _switch_window(ui_double_slit, selection_window))

    # 2-双缝衍射
    ui_double_slit_2 = InterferenceFringeCalculator_2()
    ui_selection.switch_to_double_slit_diffraction.connect(lambda: _switch_window(selection_window, ui_double_slit_2))
    ui_double_slit_2.back_button.clicked.connect(lambda: _switch_window(ui_double_slit_2, selection_window))

    # 3-牛顿环
    newton_window = QMainWindow()
    ui_newton = Ui_NewtonWindow()
    ui_newton.setupUi(newton_window)
    _wrap_mainwindow_with_scroll(newton_window)

    # 4-单缝衍射
    single_window = QMainWindow()
    ui_single = Ui_SingleWindow()
    ui_single.setupUi(single_window)
    _wrap_mainwindow_with_scroll(single_window)

    # 5-光栅衍射
    grating_window = QMainWindow()
    ui_grating = Ui_GratingWindow()
    ui_grating.setupUi(grating_window)
    _wrap_mainwindow_with_scroll(grating_window)

    # 6-劈尖干涉
    wedge_window = QMainWindow()
    ui_wedge = Ui_WedgeWindow()
    ui_wedge.setupUi(wedge_window)
    _wrap_mainwindow_with_scroll(wedge_window)

    # 7-迈克尔逊干涉
    micheison_window = QMainWindow()
    ui_micheison = Ui_MicheIsonWindow()
    ui_micheison.setupUi(micheison_window)
    _wrap_mainwindow_with_scroll(micheison_window)

    # 8-圆孔衍射
    circular_window = QMainWindow()
    ui_circular = Ui_CircularWindow(circular_window)
    ui_circular.setupUi(circular_window)
    _wrap_mainwindow_with_scroll(circular_window)

    # 9-偏振
    polarization_window = QMainWindow()
    ui_polarization = Ui_PolarizationWindow()
    ui_polarization.setupUi(polarization_window)
    _wrap_mainwindow_with_scroll(polarization_window)

    # 连接信号和槽函数
    ui_login.switch_to_introduction.connect(lambda: _switch_window(login_window, introduction_window))
    ui_introduction.switch_to_login.connect(lambda: _switch_window(introduction_window, login_window))
    ui_introduction.switch_to_main.connect(lambda: _switch_window(introduction_window, selection_window))
    ui_selection.switch_to_introduction.connect(lambda: _switch_window(selection_window, introduction_window))

    ui_selection.switch_to_newton.connect(lambda: _switch_window(selection_window, newton_window))
    ui_newton.switch_to_main.connect(lambda: _switch_window(newton_window, selection_window))

    ui_selection.switch_to_single.connect(lambda: _switch_window(selection_window, single_window))
    ui_single.switch_to_main.connect(lambda: _switch_window(single_window, selection_window))

    ui_selection.switch_to_grating.connect(lambda: _switch_window(selection_window, grating_window))
    ui_grating.switch_to_main.connect(lambda: _switch_window(grating_window, selection_window))

    ui_selection.switch_to_wedge.connect(lambda: _switch_window(selection_window, wedge_window))
    ui_wedge.switch_to_main.connect(lambda: _switch_window(wedge_window, selection_window))

    ui_selection.switch_to_micheison.connect(lambda: _switch_window(selection_window, micheison_window))
    ui_micheison.switch_to_main.connect(lambda: _switch_window(micheison_window, selection_window))

    ui_selection.switch_to_circular.connect(lambda: _switch_window(selection_window, circular_window))
    ui_circular.switch_to_main.connect(lambda: _switch_window(circular_window, selection_window))

    ui_selection.switch_to_polarization.connect(lambda: _switch_window(selection_window, polarization_window))
    ui_polarization.switch_to_main.connect(lambda: _switch_window(polarization_window, selection_window))

    # 为所有顶层窗口安装 Esc → 退出确认（窗口级过滤）
    _all_windows = [
        login_window, introduction_window, selection_window,
        newton_window, single_window, grating_window, wedge_window,
        micheison_window, circular_window, polarization_window,
        ui_double_slit, ui_double_slit_2,
    ]
    for win in _all_windows:
        _install_esc_exit(app, win)
        win.installEventFilter(_FullScreenEnforcer(win))
    # 应用级过滤，确保任意焦点下按 Esc 都能被拦截
    app.installEventFilter(_EscExitFilter(app))
    app.installEventFilter(_FullScreenEnforcer())
    # 备用：全局 Esc 快捷键（ApplicationShortcut 在任意界面生效）
    _esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), login_window)
    _esc_shortcut.setContext(Qt.ApplicationShortcut)
    _esc_shortcut.activated.connect(lambda: _ask_quit_and_maybe_exit(app, app.activeWindow()))

    # login_window.showFullScreen()  # 程序启动即全屏，取消小窗口/自由拖拽大小
    login_window.showMaximized()  # 程序启动即最大化，允许用户调整窗口大小，但不强制全屏
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

