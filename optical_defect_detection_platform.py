# -*- coding: utf-8 -*-
"""光学元件表面瑕疵检测平台。

一个可独立启动的 PyQt5 页面，用于向训练好的 YOLO 模型上传图片并查看
光学元件表面瑕疵识别结果。

运行：
    python optical_defect_detection_platform.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from defect_ai_module import DefectAIService


BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.70


def inference_device() -> str:
    """开发环境用 GPU，发布的独立 EXE 固定使用 CPU 推理。"""
    return "cpu" if getattr(sys, "frozen", False) else "0"


def preferred_model_path() -> Path:
    """返回项目内优先使用的训练权重，并兼容旧版目录结构。"""
    candidates = [
        BASE_DIR / "dl_main_model_yolo" / "runs" / "train" / "yolo11s_v3_1280" / "weights" / "best.pt",
        BASE_DIR / "dl_main_model_yolo" / "runs" / "train" / "yolo11s_final_split94_focus_v4_mixed1280" / "weights" / "best.pt",
        BASE_DIR / "dl_main_model_yolo\\runs\\train\\yolo11s_final_split94_focus_v2_ft1280\\weights\\best.pt",
        BASE_DIR / "dl_main_model_yolo" / "runs" / "train" / "yolo11s_final_split94" / "weights" / "best.pt",
        BASE_DIR / "dl_main_model_yolo" / "runs" / "train" / "yolo11s_final_split94_smalltarget_ft1280" / "weights" / "best.pt",
        BASE_DIR / "dl_main_model_yolo" / "runs" / "train" / "defect_localization_smoke" / "weights" / "best.pt",
        BASE_DIR / "yolo11s.pt",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


class DetectionWorker(QThread):
    """将耗时的权重加载和推理放到后台线程，保持页面可操作。"""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image_path: str, model_path: str) -> None:
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path

    def run(self) -> None:
        try:
            service = DefectAIService(self.model_path)
            # 上传的通常是原始相机图像，不裁切，确保检测框与原图一致。
            result = service.predict_image(
                self.image_path,
                conf=DEFAULT_CONFIDENCE,
                iou=DEFAULT_IOU,
                device=inference_device(),
                # The GUI renders an in-memory preview after inference.  Do
                # not create a sibling "*_检测结果.png" merely by detecting.
                save_visualization=False,
                crop_plot_area=False,
            )
            self.completed.emit(result)
        except Exception as exc:  # 将模型与图像错误反馈到 GUI
            self.failed.emit(str(exc))


class ImagePanel(QFrame):
    """带占位状态的图片预览面板。"""

    def __init__(self, title: str, hint: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self._image_path = ""
        self._source_pixmap = QPixmap()
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 18)
        layout.setSpacing(10)

        heading = QLabel(title)
        heading.setObjectName("panelTitle")
        layout.addWidget(heading)

        self.canvas = QLabel(hint)
        self.canvas.setObjectName("imageCanvas")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setWordWrap(True)
        self.canvas.setMinimumSize(350, 390)
        layout.addWidget(self.canvas, 1)

    def set_image(self, image_path: str) -> None:
        self._image_path = image_path
        self._source_pixmap = QPixmap(image_path)
        self._refresh_pixmap()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._image_path = ""
        self._source_pixmap = pixmap
        self._refresh_pixmap()

    def clear_image(self, text: str) -> None:
        self._image_path = ""
        self._source_pixmap = QPixmap()
        self.canvas.setPixmap(QPixmap())
        self.canvas.setText(text)

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        margins = self.canvas.contentsMargins()
        available_width = max(1, self.canvas.width() - margins.left() - margins.right() - 8)
        available_height = max(1, self.canvas.height() - margins.top() - margins.bottom() - 8)
        self.canvas.setPixmap(
            self._source_pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.canvas.setText("")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 回调命名
        super().resizeEvent(event)
        self._refresh_pixmap()


class OpticalDefectDetectionWindow(QMainWindow):
    """光学元件表面瑕疵检测平台主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.image_path = ""
        self.result_image_path = ""
        self.result_preview = QPixmap()
        self.model_path = str(preferred_model_path())
        self.worker: DetectionWorker | None = None

        self.setWindowTitle("光学元件表面瑕疵检测平台")
        self.setMinimumSize(1260, 800)
        self._build_ui()
        self._apply_styles()
        self._set_model_display()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(34, 26, 34, 24)
        layout.setSpacing(18)

        layout.addLayout(self._build_header())

        content = QHBoxLayout()
        content.setSpacing(18)
        content.addWidget(self._build_control_column(), 0)

        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(18)
        preview_grid.setVerticalSpacing(12)
        self.original_panel = ImagePanel("01 / 待检原图", "上传一张光学元件表面图像\n开始 AI 瑕疵检测")
        self.result_panel = ImagePanel("02 / 检测结果", "检测完成后将在这里显示\n瑕疵位置与类别标注")
        preview_grid.addWidget(self.original_panel, 0, 0)
        preview_grid.addWidget(self.result_panel, 0, 1)
        preview_grid.setColumnStretch(0, 1)
        preview_grid.setColumnStretch(1, 1)

        result_box = QFrame()
        result_box.setObjectName("resultBox")
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(18, 14, 18, 16)
        result_layout.setSpacing(10)

        result_top = QHBoxLayout()
        result_label = QLabel("检测明细")
        result_label.setObjectName("panelTitle")
        result_top.addWidget(result_label)
        result_top.addStretch()
        self.result_summary = QLabel("等待上传图像")
        self.result_summary.setObjectName("resultSummary")
        result_top.addWidget(self.result_summary)
        result_layout.addLayout(result_top)

        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["序号", "瑕疵类别", "置信度", "检测框坐标 (x1, y1, x2, y2)"])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setMinimumHeight(175)
        result_layout.addWidget(self.result_table)
        preview_grid.addWidget(result_box, 1, 0, 1, 2)
        preview_grid.setRowStretch(0, 1)

        preview_host = QWidget()
        preview_host.setLayout(preview_grid)
        content.addWidget(preview_host, 1)
        layout.addLayout(content, 1)

        status = QStatusBar()
        status.setObjectName("statusBar")
        status.showMessage("就绪 · 请选择待检测图像")
        self.setStatusBar(status)

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(14)
        brand = QLabel("OpticalLab")
        brand.setObjectName("brand")
        header.addWidget(brand)

        divider = QFrame()
        divider.setObjectName("headerDivider")
        divider.setFixedWidth(1)
        header.addWidget(divider)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        title = QLabel("光学元件表面瑕疵检测平台")
        title.setObjectName("appTitle")
        subtitle = QLabel("基于 YOLO 的表面缺陷识别与定位 · 上传图像，查看可视化检测结果")
        subtitle.setObjectName("appSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch()

        self.state_badge = QLabel("● 系统就绪")
        self.state_badge.setObjectName("stateBadge")
        header.addWidget(self.state_badge)
        return header

    def _build_control_column(self) -> QFrame:
        card = QFrame()
        card.setObjectName("controlCard")
        card.setFixedWidth(292)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        heading = QLabel("AI 缺陷识别")
        heading.setObjectName("sideHeading")
        layout.addWidget(heading)
        tip = QLabel("上传光学元件图像，系统将通过已训练模型完成瑕疵定位与类别识别。")
        tip.setObjectName("sideTip")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        layout.addSpacing(10)

        upload_btn = QPushButton("＋  上传待检图像")
        upload_btn.setObjectName("primaryButton")
        upload_btn.setCursor(Qt.PointingHandCursor)
        upload_btn.clicked.connect(self.select_image)
        layout.addWidget(upload_btn)

        self.detect_btn = QPushButton("◈  开始瑕疵检测")
        self.detect_btn.setObjectName("detectButton")
        self.detect_btn.setCursor(Qt.PointingHandCursor)
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detection)
        layout.addWidget(self.detect_btn)

        layout.addWidget(self._rule())
        model_caption = QLabel("当前模型权重")
        model_caption.setObjectName("fieldCaption")
        layout.addWidget(model_caption)
        self.model_value = QLabel()
        self.model_value.setObjectName("pathValue")
        self.model_value.setWordWrap(True)
        layout.addWidget(self.model_value)

        model_btn = QPushButton("更换权重文件 (.pt)")
        model_btn.setObjectName("secondaryButton")
        model_btn.setCursor(Qt.PointingHandCursor)
        model_btn.clicked.connect(self.select_model)
        layout.addWidget(model_btn)

        layout.addStretch()
        save_btn = QPushButton("⌄  保存检测结果图")
        save_btn.setObjectName("secondaryButton")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self.save_result)
        layout.addWidget(save_btn)

        support = QLabel("支持 JPG · PNG · BMP · TIFF · WEBP")
        support.setObjectName("supportText")
        support.setAlignment(Qt.AlignCenter)
        layout.addWidget(support)
        return card

    @staticmethod
    def _rule() -> QFrame:
        rule = QFrame()
        rule.setObjectName("rule")
        rule.setFixedHeight(1)
        return rule

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QWidget#root { background: #F4F8FC; color: #182B48; }
            QLabel { font-family: "Microsoft YaHei"; background: transparent; }
            QLabel#brand { color: #1677FF; font-family: "Microsoft YaHei"; font-size: 18px; font-weight: bold; letter-spacing: 0.4px; }
            QFrame#headerDivider { background: #D5E3F4; }
            QLabel#appTitle { color: #142B4B; font-size: 25px; font-weight: 700; letter-spacing: 1px; }
            QLabel#appSubtitle { color: #7889A1; font-size: 11px; }
            QLabel#stateBadge { color: #18895F; background: #EDF9F4; border: 1px solid #B9E7D4; border-radius: 12px; padding: 6px 11px; font-family: "Microsoft YaHei"; font-size: 10px; font-weight: bold; }
            QFrame#controlCard, QFrame#imagePanel, QFrame#resultBox { background: white; border: 1px solid #E1EAF5; border-radius: 10px; }
            QFrame#controlCard { border-top: 3px solid #1677FF; }
            QLabel#sideHeading { color: #183457; font-size: 18px; font-weight: bold; }
            QLabel#sideTip { color: #71829A; font-size: 12px; line-height: 1.55; }
            QLabel#fieldCaption { color: #507098; font-size: 10px; font-weight: bold; letter-spacing: 1px; }
            QLabel#pathValue { color: #52657D; background: #F7FAFE; border: 1px solid #E1EAF5; border-radius: 6px; padding: 8px; font-family: Consolas; font-size: 10px; }
            QLabel#supportText { color: #9AA8BA; font-family: Consolas; font-size: 9px; }
            QFrame#rule { background: #E7EEF7; }
            QLabel#panelTitle { color: #4D6687; font-family: "Microsoft YaHei"; font-size: 11px; font-weight: bold; letter-spacing: 1px; }
            QLabel#imageCanvas { background: #F8FBFF; color: #8092AA; border: 1px dashed #B9CCE5; border-radius: 7px; font-size: 13px; line-height: 1.6; }
            QLabel#resultSummary { color: #1677FF; font-size: 11px; font-weight: bold; }
            QPushButton { min-height: 40px; border-radius: 6px; font-family: "Microsoft YaHei"; font-size: 12px; font-weight: bold; padding: 0 12px; }
            QPushButton#primaryButton { background: #1677FF; border: 1px solid #1677FF; color: white; }
            QPushButton#primaryButton:hover { background: #4096FF; border-color: #4096FF; }
            QPushButton#detectButton { background: #13A8A8; border: 1px solid #13A8A8; color: white; }
            QPushButton#detectButton:hover { background: #36CFC9; border-color: #36CFC9; }
            QPushButton#detectButton:disabled { background: #E7EDF5; border-color: #E1E8F1; color: #A5B2C2; }
            QPushButton#secondaryButton { background: white; border: 1px solid #C9D9EB; color: #426286; }
            QPushButton#secondaryButton:hover { background: #F0F7FF; color: #1677FF; border-color: #8BB8F7; }
            QTableWidget { background: white; alternate-background-color: #F8FBFF; border: 1px solid #E1EAF5; border-radius: 6px; color: #405572; gridline-color: #E9F0F7; font-size: 11px; }
            QHeaderView::section { background: #F2F7FD; color: #527096; border: none; border-right: 1px solid #E1EAF5; padding: 8px; font-family: "Microsoft YaHei"; font-size: 10px; font-weight: bold; }
            QTableWidget::item:selected { background: #E6F4FF; color: #1554AD; }
            QStatusBar#statusBar { background: white; color: #71829A; border-top: 1px solid #DFE9F4; font-family: "Microsoft YaHei"; font-size: 11px; }
        """)

    def _set_model_display(self) -> None:
        path = Path(self.model_path)
        try:
            self.model_value.setText(str(path.relative_to(BASE_DIR)))
        except ValueError:
            self.model_value.setText(str(path))

    def select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择待检测图像", str(BASE_DIR), IMAGE_FILTER)
        if not path:
            return
        self.image_path = path
        self.result_image_path = ""
        self.result_preview = QPixmap()
        self.original_panel.set_image(path)
        self.result_panel.clear_image("点击“开始瑕疵检测”\n生成带定位框的结果图")
        self.result_table.setRowCount(0)
        self.result_summary.setText("图像已载入，等待检测")
        self.detect_btn.setEnabled(True)
        self.statusBar().showMessage(f"已载入图像 · {Path(path).name}")

    def select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择训练权重", str(Path(self.model_path).parent), "PyTorch 权重 (*.pt)")
        if path:
            self.model_path = path
            self._set_model_display()
            self.statusBar().showMessage(f"已切换模型权重 · {Path(path).name}")

    def start_detection(self) -> None:
        if not self.image_path:
            return
        if not Path(self.model_path).exists():
            QMessageBox.warning(self, "权重不可用", f"未找到模型权重：\n{self.model_path}\n\n请通过“更换权重文件”选择有效的 .pt 文件。")
            return
        self.detect_btn.setEnabled(False)
        self.state_badge.setText("◌ 正在检测")
        self.state_badge.setStyleSheet("color:#1677FF; background:#EAF3FF; border:1px solid #B8D7FF; border-radius:12px; padding:6px 11px; font-family:'Microsoft YaHei'; font-size:10px; font-weight:bold;")
        self.result_summary.setText("模型推理中…")
        self.statusBar().showMessage("正在加载权重并执行检测，请稍候…")
        self.worker = DetectionWorker(self.image_path, self.model_path)
        self.worker.completed.connect(self._on_detection_completed)
        self.worker.failed.connect(self._on_detection_failed)
        self.worker.finished.connect(self._release_worker)
        self.worker.start()

    def _on_detection_completed(self, result) -> None:
        predictions = result.predictions
        self.result_image_path = ""
        self.result_preview = self._render_result_preview(predictions)
        self.result_panel.set_pixmap(self.result_preview)
        self._fill_result_table(predictions)
        if predictions:
            self.result_summary.setText(f"发现 {len(predictions)} 处疑似瑕疵")
            self.statusBar().showMessage(f"检测完成 · 共识别 {len(predictions)} 处疑似瑕疵")
        else:
            self.result_summary.setText("未发现明显瑕疵")
            self.statusBar().showMessage("检测完成 · 未发现明显瑕疵")
        self._set_ready_state()

    def _render_result_preview(self, predictions) -> QPixmap:
        """Draw detection boxes in memory; disk I/O happens only on Save."""
        preview = QPixmap(self.image_path).copy()
        if preview.isNull():
            return preview
        colors = {
            "scratch": QColor(255, 165, 0),
            "bump": QColor(0, 255, 0),
            "dent": QColor(0, 120, 255),
            "dust": QColor(255, 255, 0),
            "hair": QColor(255, 0, 255),
        }
        painter = QPainter(preview)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Microsoft YaHei", 10)
        painter.setFont(font)
        for item in predictions:
            if not item.bbox_xyxy:
                continue
            x1, y1, x2, y2 = [round(value) for value in item.bbox_xyxy]
            color = colors.get(item.label, QColor(0, 200, 200))
            painter.setPen(QPen(color, 2))
            painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            painter.setPen(QPen(color, 1))
            painter.drawText(x1 + 3, max(14, y1 - 4), f"{item.label_cn} {item.confidence:.2f}")
        painter.end()
        return preview

    def _fill_result_table(self, predictions) -> None:
        self.result_table.setRowCount(len(predictions))
        for row, item in enumerate(predictions, start=0):
            bbox = item.bbox_xyxy or []
            bbox_text = ", ".join(str(round(value, 1)) for value in bbox) if bbox else "—"
            values = [str(row + 1), item.label_cn, f"{item.confidence:.1%}", bbox_text]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignCenter if col != 3 else Qt.AlignLeft | Qt.AlignVCenter)
                self.result_table.setItem(row, col, cell)
        self.result_table.resizeColumnsToContents()

    def _on_detection_failed(self, message: str) -> None:
        self.result_summary.setText("检测失败")
        self.statusBar().showMessage("检测失败，请检查模型、图像及运行环境")
        self._set_ready_state()
        QMessageBox.critical(self, "检测失败", f"无法完成瑕疵识别：\n\n{message}")

    def _release_worker(self) -> None:
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _set_ready_state(self) -> None:
        self.state_badge.setText("● 系统就绪")
        self.state_badge.setStyleSheet("")
        self.detect_btn.setEnabled(bool(self.image_path))

    def save_result(self) -> None:
        if self.result_preview.isNull():
            QMessageBox.information(self, "暂无结果", "请先完成一次检测，再保存带定位框的结果图。")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测结果图",
            str(Path(self.image_path).with_name(f"{Path(self.image_path).stem}_检测结果.png")),
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg)",
        )
        if not target:
            return
        try:
            if not self.result_preview.save(target):
                raise OSError("无法写入结果图，请检查文件扩展名和目标目录权限。")
            self.statusBar().showMessage(f"已保存检测结果图 · {Path(target).name}")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))


def main() -> int:
    # 避免某些 Windows 环境中 OpenCV 与 PyQt5 的平台插件路径冲突。
    try:
        import PyQt5
        plugin_path = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins" / "platforms"
        if plugin_path.exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_path))
    except Exception:
        pass

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = OpticalDefectDetectionWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
