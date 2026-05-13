# -*- coding: utf-8 -*-
"""
物理实验综合平台 —— 主启动器
提供虚拟实验、仿真实验、光学瑕疵检测三大平台的统一入口。
"""

import sys
import os

# 修复因中文路径或环境冲突导致找不到 qwindows.dll 的问题
venv_base = os.path.dirname(os.path.dirname(sys.executable))
plugin_path = os.path.join(venv_base, "Lib", "site-packages", "PyQt5", "Qt5", "plugins", "platforms")
if os.path.exists(plugin_path):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
else:
    # 兼容另外一些可能安装到的路径
    plugin_path = os.path.join(venv_base, "Lib", "site-packages", "PyQt5", "Qt", "plugins", "platforms")
    if os.path.exists(plugin_path):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

import importlib

from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, pyqtSignal, QProcess  # type: ignore
from PyQt5.QtGui import QColor, QLinearGradient, QPainter  # type: ignore

# ── 平台数据定义 ────────────────────────────────────────────────────────────────
PLATFORMS = [
    {
        "title": "虚拟实验平台",
        "icon": "🔬",
        "script": "virtual_main.py",
        "accent": "#4A90D9",
        "accent_dark": "#2C6FAC",
        "tag": "Virtual Lab",
        "desc": (
            "提供牛顿环、迈克尔逊干涉仪、劈尖干涉等经典光学实验的"
            "虚拟操作界面。支持数据记录与处理，真实还原实验操作流程，"
            "并提供偏振等拓展思考模块，辅助深入理解物理原理。"
        ),
        "features": ["牛顿环 · 迈克尔逊 · 劈尖干涉", "数据记录与误差分析", "偏振实验 · 视频演示"],
    },
    {
        "title": "仿真实验平台",
        "icon": "⚗️",
        "script": "simulation_main.py",
        "accent": "#7B5EA7",
        "accent_dark": "#56407A",
        "tag": "Simulation",
        "desc": (
            "集成双缝干涉、单缝衍射、光栅衍射、圆孔衍射、偏振等"
            "九项光学仿真实验，通过可调参数实时演示波动光学现象，"
            "直观呈现干涉与衍射的规律，适合课堂演示与自主探究。"
        ),
        "features": ["9 项光学仿真实验", "参数实时可调 · 波形动态演示", "双缝 · 光栅 · 圆孔 · 偏振"],
    },
    {
        "title": "光学瑕疵检测平台",
        "icon": "🔭",
        "script": "opticaldefectt_main.py",
        "accent": "#2E9E6B",
        "accent_dark": "#1D7050",
        "tag": "Defect Detection",
        "desc": (
            "基于干涉原理对光学元件表面瑕疵进行模拟检测，直观展示"
            "划痕、凸起、凹陷等缺陷对干涉条纹形态的影响，"
            "帮助理解光学加工质量控制与精密检测方法。"
        ),
        "features": ["表面瑕疵干涉模拟", "划痕 · 凸起 · 凹陷缺陷分析", "条纹形变可视化"],
    },
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── onefile 分发模式 ─────────────────────────────────────────────────────────────
def _dispatch_mode_from_argv() -> None:
    """
    统一入口 main.py 在打包(onefile)后仍然只有一个 EXE。
    通过 `--mode` 参数分发到不同平台入口，避免直接启动 .py 文件在 onefile 下失效。
    """
    if "--mode" not in sys.argv:
        return
    try:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1].strip().lower()
    except Exception:
        mode = ""

    mapping = {
        "virtual": ("virtual_main", "main"),
        "simulation": ("simulation_main", "main"),
        "defect": ("opticaldefectt_main", "main"),
    }
    if mode not in mapping:
        return

    # 清理 sys.argv，去掉 --mode <value>，避免子模块因未知参数出错
    try:
        idx = sys.argv.index("--mode")
        sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]
    except (ValueError, IndexError):
        pass

    mod_name, fn_name = mapping[mode]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    raise SystemExit(fn())


# 在创建 Launcher 的 QApplication 前先分发
_dispatch_mode_from_argv()

# ── 样式表 ──────────────────────────────────────────────────────────────────────
MAIN_STYLE = """
QWidget#root {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #0D1B2A, stop:0.5 #12263A, stop:1 #071220);
}
QLabel#app_title {
    color: #E8F4FD;
    font-size: 32px;
    font-weight: bold;
    letter-spacing: 4px;
}
QLabel#app_subtitle {
    color: #7FB3D3;
    font-size: 13px;
    letter-spacing: 2px;
}
QFrame#divider {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 transparent, stop:0.3 #2E6FA0, stop:0.7 #2E6FA0, stop:1 transparent);
    max-height: 1px;
    min-height: 1px;
}
QFrame#card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
}
QLabel#card_icon {
    font-size: 42px;
    background: transparent;
}
QLabel#card_tag {
    color: #A0BDD0;
    font-size: 10px;
    letter-spacing: 2px;
    background: transparent;
}
QLabel#card_title {
    color: #EAF4FB;
    font-size: 20px;
    font-weight: bold;
    background: transparent;
}
QLabel#card_desc {
    color: #AABDCC;
    font-size: 12px;
    line-height: 1.6;
    background: transparent;
}
QLabel#feature_item {
    color: #7FBFDA;
    font-size: 11px;
    background: transparent;
}
QLabel#footer {
    color: #3E6070;
    font-size: 11px;
}
"""

LAUNCH_BTN_STYLE = """
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent}, stop:1 {accent_dark});
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent_light}, stop:1 {accent});
}}
QPushButton:pressed {{
    background: {accent_dark};
}}
"""

QUIT_BTN_STYLE = """
QPushButton {
    background: rgba(255,255,255,0.06);
    color: #7090A0;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 8px 32px;
    font-size: 12px;
    letter-spacing: 1px;
}
QPushButton:hover {
    background: rgba(220,60,60,0.25);
    color: #F08080;
    border-color: rgba(220,60,60,0.40);
}
QPushButton:pressed {
    background: rgba(180,40,40,0.40);
}
"""


def _lighten(hex_color: str, factor: float = 1.3) -> str:
    c = QColor(hex_color)
    h, s, v, a = c.getHsvF()
    v = min(1.0, v * factor)
    c.setHsvF(h, max(0.0, s * 0.85), v, a)
    return c.name()


def _add_shadow(widget, color: str, blur: int = 24, offset: int = 4):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, offset)
    shadow.setColor(QColor(color))
    widget.setGraphicsEffect(shadow)


# ── 平台卡片组件 ────────────────────────────────────────────────────────────────
class PlatformCard(QFrame):
    launch_requested = pyqtSignal(str)  # 发出要启动的 script 名称

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self._build_ui(data)
        _add_shadow(self, data["accent"] + "66", blur=30, offset=6)

    def _build_ui(self, d: dict):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 26)
        layout.setSpacing(0)

        # ── 顶部标签行
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        icon_lbl = QLabel(d["icon"])
        icon_lbl.setObjectName("card_icon")
        top_row.addWidget(icon_lbl)
        top_row.addStretch()

        tag_lbl = QLabel(d["tag"].upper())
        tag_lbl.setObjectName("card_tag")
        top_row.addWidget(tag_lbl)

        layout.addLayout(top_row)
        layout.addSpacing(12)

        # ── 标题
        title = QLabel(d["title"])
        title.setObjectName("card_title")
        title.setStyleSheet(f"QLabel{{ color: {d['accent']}; font-size:20px; font-weight:bold; background:transparent; }}")
        layout.addWidget(title)
        layout.addSpacing(2)

        # ── 彩色短横线装饰
        line = QFrame()
        line.setFixedSize(36, 3)
        line.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {d['accent']}, stop:1 transparent);"
            f"border-radius:2px;"
        )
        layout.addWidget(line)
        layout.addSpacing(14)

        # ── 描述
        desc = QLabel(d["desc"])
        desc.setObjectName("card_desc")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(desc)
        layout.addSpacing(16)

        # ── 功能特性列表
        for feat in d["features"]:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)

            dot = QLabel("▸")
            dot.setStyleSheet(f"color:{d['accent']}; font-size:10px; background:transparent;")
            dot.setFixedWidth(14)
            row.addWidget(dot)

            feat_lbl = QLabel(feat)
            feat_lbl.setObjectName("feature_item")
            row.addWidget(feat_lbl)
            row.addStretch()

            layout.addLayout(row)
            layout.addSpacing(4)

        layout.addSpacing(20)
        layout.addStretch()

        # ── 启动按钮
        btn = QPushButton(f"启动  {d['title']}")
        accent_light = _lighten(d["accent"], 1.25)
        btn.setStyleSheet(
            LAUNCH_BTN_STYLE.format(
                accent=d["accent"],
                accent_dark=d["accent_dark"],
                accent_light=accent_light,
            )
        )
        btn.setFixedHeight(42)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _, s=d["script"]: self.launch_requested.emit(s))
        layout.addWidget(btn)


# ── 主启动器窗口 ────────────────────────────────────────────────────────────────
class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle("物理实验综合平台")
        self.setMinimumSize(1100, 680)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 50, 60, 36)
        root.setSpacing(0)

        # ── 顶部标题区
        header = QVBoxLayout()
        header.setSpacing(6)

        title = QLabel("物 理 实 验 综 合 平 台")
        title.setObjectName("app_title")
        title.setAlignment(Qt.AlignCenter)
        header.addWidget(title)

        subtitle = QLabel("Physics Experimental Integrated Platform  ·  三大模块  ·  多维探究")
        subtitle.setObjectName("app_subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        header.addWidget(subtitle)

        root.addLayout(header)
        root.addSpacing(24)

        # ── 分割线
        divider = QFrame()
        divider.setObjectName("divider")
        root.addWidget(divider)
        root.addSpacing(40)

        # ── 三张卡片横排
        cards_row = QHBoxLayout()
        cards_row.setSpacing(28)
        cards_row.addStretch(1)
        for platform in PLATFORMS:
            card = PlatformCard(platform)
            card.launch_requested.connect(self._launch)
            cards_row.addWidget(card)
        cards_row.addStretch(1)

        root.addLayout(cards_row)
        root.addStretch(1)

        # ── 底部说明与退出按钮
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)

        footer = QLabel("点击对应平台按钮以独立启动各模块 · 各平台运行互不影响")
        footer.setObjectName("footer")
        bottom.addWidget(footer)
        bottom.addStretch()

        quit_btn = QPushButton("退 出")
        quit_btn.setObjectName("quit_btn")
        quit_btn.setStyleSheet(QUIT_BTN_STYLE)
        quit_btn.setFixedHeight(38)
        quit_btn.setFixedWidth(100)
        quit_btn.setCursor(Qt.PointingHandCursor)
        quit_btn.clicked.connect(QApplication.quit)
        bottom.addWidget(quit_btn)

        root.addSpacing(16)
        root.addLayout(bottom)

    def _launch(self, script: str):
        """启动子平台进程，隐藏启动器；子进程退出后重新显示启动器。"""
        script_to_mode = {
            "virtual_main.py": "virtual",
            "simulation_main.py": "simulation",
            "opticaldefectt_main.py": "defect",
        }
        mode = script_to_mode.get(script, "")

        proc = QProcess(self)

        if not mode:
            # 回退：直接运行对应 .py 脚本
            path = os.path.join(BASE_DIR, script)
            proc.start(sys.executable, [path])
        elif getattr(sys, "frozen", False):
            # 打包 EXE 模式：sys.executable 就是 EXE 本身
            proc.start(sys.executable, ["--mode", mode])
        else:
            # 开发模式：python main.py --mode <mode>
            main_script = os.path.join(BASE_DIR, "main.py")
            proc.start(sys.executable, [main_script, "--mode", mode])

        # 隐藏启动器，等子进程退出后恢复
        self.hide()
        proc.finished.connect(self.showMaximized)
        # 保持引用，防止被 GC 回收
        self._active_proc = proc

    def paintEvent(self, event):
        """绘制渐变背景（弥补 QSS gradient 在某些平台的兼容问题）"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QColor("#0D1B2A"))
        grad.setColorAt(0.5, QColor("#12263A"))
        grad.setColorAt(1.0, QColor("#071220"))
        painter.fillRect(self.rect(), grad)
        super().paintEvent(event)


# ── 入口 ────────────────────────────────────────────────────────────────────────
def main():
    # 高 DPI 属性必须在 QApplication 构造之前以类方法设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(MAIN_STYLE)

    window = LauncherWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
