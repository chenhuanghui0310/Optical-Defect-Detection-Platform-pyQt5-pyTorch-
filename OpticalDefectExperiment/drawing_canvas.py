import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy  # pyright: ignore[reportMissingImports]
from PyQt5.QtCore import Qt, QPoint  # pyright: ignore[reportMissingImports]
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap  # pyright: ignore[reportMissingImports]


class DrawingCanvas(QWidget):
    """手动画布控件 - 与光谱图图案部分尺寸一致（600×300像素）"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置与光谱图图案部分相同的固定尺寸
        self.canvas_width = 710
        self.canvas_height = 350

        # 设置固定尺寸策略
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(self.canvas_width, self.canvas_height)

        # 初始为禁用状态样式
        self.setStyleSheet("""
            DrawingCanvas {
                background-color: #f5f5f5; 
                border: 2px solid #cccccc; 
                border-radius: 5px;
                color: #999999;
            }
        """)

        # 创建画布 - 使用与显示区域相同的尺寸
        self.image = QPixmap(self.canvas_width, self.canvas_height)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brush_size = 15
        self.brush_depth = 0.15
        self.eraser_mode = False
        self.last_point = QPoint()
        self.current_point = QPoint()
        self.show_grid = True
        self.grid_size = 30
        self.is_enabled = False
        self.brush_profile_cache = {}  # 缓存画笔剖面

        self.setMouseTracking(True)

    def _create_half_circle_brush_profile(self, size, depth=0.5):
        """创建半圆形画笔剖面 - 使用幂函数和余弦函数增强平滑过渡"""
        cache_key = (size, depth)
        if cache_key in self.brush_profile_cache:
            return self.brush_profile_cache[cache_key]

        diameter = size * 2
        if diameter <= 0:
            return np.zeros((1, 1))

        profile = np.zeros((diameter, diameter))
        center = size

        # 根据画笔大小选择不同的过渡函数参数
        if size > 15:  # 大尺寸画笔
            transition_type = 'cosine'  # 使用余弦函数，更适合大尺寸
            power_factor = 1.5  # 幂函数指数（值越小越平滑）
        elif size > 8:  # 中等尺寸画笔
            transition_type = 'blend'  # 混合函数
            power_factor = 1.8
        else:  # 小尺寸画笔
            transition_type = 'power'  # 使用幂函数，边缘更柔和
            power_factor = 2.0

        # 创建平滑的半圆形剖面
        for i in range(diameter):
            for j in range(diameter):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= size:
                    normalized_dist = distance / size

                    # 根据选择的过渡类型计算剖面值
                    if transition_type == 'cosine':
                        # 使用余弦函数创建平滑过渡
                        # 余弦函数在边缘处有更平缓的导数，过渡自然
                        profile_value = 0.5 * (1 + np.cos(np.pi * normalized_dist))

                    elif transition_type == 'power':
                        # 使用幂函数创建平滑过渡
                        # 幂函数提供了可调节的衰减速率
                        profile_value = (1 - normalized_dist) ** power_factor

                    elif transition_type == 'blend':
                        # 混合函数：结合余弦和幂函数的优点
                        cosine_value = 0.5 * (1 + np.cos(np.pi * normalized_dist))
                        power_value = (1 - normalized_dist) ** power_factor

                        # 使用加权平均，中心区域更多使用幂函数，边缘更多使用余弦函数
                        blend_weight = normalized_dist  # 距离越大，余弦权重越大
                        profile_value = (1 - blend_weight) * power_value + blend_weight * cosine_value

                    # 应用深度参数
                    profile[i, j] = profile_value * depth

                    # 【可选】添加轻微的中心增强，但要保持平滑
                    if normalized_dist < 0.3:
                        # 使用更平滑的中心增强函数
                        center_boost = 1 + 0.5 * (1 - normalized_dist / 0.3)  # 线性增强
                        profile[i, j] *= center_boost

        # 归一化到0-1范围，确保最大值不超过1
        if profile.max() > 0:
            profile = profile / profile.max()

        # 应用后期平滑处理
        try:
            from scipy import ndimage
            if profile.shape[0] > 1 and profile.shape[1] > 1:
                sigma = 1.2  # 中等尺寸
                # 应用高斯滤波进行最终平滑
                profile = ndimage.gaussian_filter(profile, sigma=sigma)

                # 确保平滑后数值仍在有效范围内
                profile = np.clip(profile, 0, 1)

        except ImportError:
            # 如果scipy不可用，使用简单的移动平均平滑
            if size > 1:
                kernel_size = min(3, size)
                profile = self._simple_smooth(profile, kernel_size)

        # 缓存结果以提高性能
        self.brush_profile_cache[cache_key] = profile
        return profile

    def _simple_smooth(self, profile, kernel_size=3):
        """简单的移动平均平滑（当scipy不可用时使用）"""
        h, w = profile.shape
        smoothed = np.zeros_like(profile)

        half_k = kernel_size // 2

        for i in range(h):
            for j in range(w):
                # 计算邻域平均值
                i_min = max(0, i - half_k)
                i_max = min(h, i + half_k + 1)
                j_min = max(0, j - half_k)
                j_max = min(w, j + half_k + 1)

                neighborhood = profile[i_min:i_max, j_min:j_max]
                smoothed[i, j] = np.mean(neighborhood)

        return smoothed
    def _apply_brush_stroke_with_profile(self, center_x, center_y, profile, is_eraser=False):
        """使用半圆形剖面应用画笔笔画 - 支持0-100%滑块映射到白-黑渐变"""
        if not self.is_enabled or self.image.isNull():
            return

        try:
            # 检查坐标有效性
            if center_x < 0 or center_y < 0:
                return

            # 将QPixmap转换为QImage以便像素级操作
            qimage = self.image.toImage()
            if qimage.isNull():
                return

            height, width = qimage.height(), qimage.width()

            # 检查图像尺寸有效性
            if height <= 0 or width <= 0:
                return

            profile_size = profile.shape[0]
            if profile_size <= 0:
                return

            half_size = profile_size // 2

            # 预计算画笔区域边界
            start_x = int(center_x - half_size)
            start_y = int(center_y - half_size)
            end_x = int(center_x + half_size)
            end_y = int(center_y + half_size)

            # 如果整个画笔区域都在画布外，直接返回
            if (end_x < 0 or start_x >= width or
                    end_y < 0 or start_y >= height):
                return

            for i in range(profile_size):
                for j in range(profile_size):
                    px = int(center_x - half_size + i)
                    py = int(center_y - half_size + j)

                    # 严格边界检查
                    if 0 <= px < width and 0 <= py < height:
                        weight = profile[i, j]
                        # 计算该点相对于画笔中心的归一化距离
                        dx = i - half_size
                        dy = j - half_size
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        normalized_dist = distance / self.brush_size if self.brush_size > 0 else 1.0
                        # 如果距离中心较近，增加权重
                        if normalized_dist < 0.3:
                            weight *= 1.2
                        if weight > 1.0:  # 确保权重不超过1
                            weight = 1.0
                        if weight > 0.01:  # 忽略权重过小的像素
                            try:
                                current_color = QColor(qimage.pixel(px, py))

                                if is_eraser:
                                    # 橡皮擦模式：混合白色
                                    erase_amount = weight
                                    new_red = int(255 * erase_amount + current_color.red() * (1 - erase_amount))
                                    new_green = int(255 * erase_amount + current_color.green() * (1 - erase_amount))
                                    new_blue = int(255 * erase_amount + current_color.blue() * (1 - erase_amount))
                                    new_color = QColor(new_red, new_green, new_blue)
                                else:
                                    # 绘图模式：计算新颜色
                                    depth_value = self.brush_depth * weight  # brush_depth 现在是0.0-1.0
                                    # 修改映射公式：滑块0-100% (深度系数0.0-1.0) 映射到白-黑渐变
                                    # 深度范围：0.0-1.0
                                    # 滑块0% -> depth_value=0 -> 白色(255)
                                    # 滑块100% -> depth_value=1.0 -> 黑色(0)
                                    # 线性映射：gray = 255 * (1 - depth_value)  # 因为 depth_value 现在就是归一化的深度
                                    if depth_value > 1.0:
                                        depth_value = 1.0
                                    new_gray_value = int(255 * (1 - depth_value))
                                    new_gray_value = max(0, min(255, new_gray_value))
                                    # 与现有颜色混合（取较深的值，模拟划痕累积）
                                    # 现有颜色的灰度值
                                    current_gray = (
                                                               current_color.red() + current_color.green() + current_color.blue()) / 3

                                    # 取较小的灰度值（较深的颜色）
                                    final_gray = min(current_gray, new_gray_value)

                                    # 创建新颜色
                                    new_color = QColor(int(final_gray), int(final_gray), int(final_gray))

                                qimage.setPixelColor(px, py, new_color)
                            except Exception as e:
                                # 单个像素处理出错时跳过，继续处理其他像素
                                continue

            # 将QImage转换回QPixmap
            self.image = QPixmap.fromImage(qimage)

        except Exception as e:
            print(f"_apply_brush_stroke_with_profile错误: {e}")
            import traceback
            traceback.print_exc()

    def draw_point(self, point):
        """在指定位置绘制一个具有半圆形剖面的点"""
        if not self.is_enabled or self.image.isNull() or point.isNull():
            return

        # 获取半圆形剖面
        profile = self._create_half_circle_brush_profile(self.brush_size, self.brush_depth)

        # 应用画笔剖面
        self._apply_brush_stroke_with_profile(point.x(), point.y(), profile, self.eraser_mode)
        self.update()

    def draw_line(self, from_point, to_point):
        """绘制具有半圆形剖面的线段"""
        if not self.is_enabled or self.image.isNull() or from_point.isNull() or to_point.isNull():
            return

        try:
            # 计算线段上的点
            x1, y1 = from_point.x(), from_point.y()
            x2, y2 = to_point.x(), to_point.y()

            dx = x2 - x1
            dy = y2 - y1
            distance = max(1, np.sqrt(dx * dx + dy * dy))

            # 根据距离和画笔大小计算采样步长
            step = max(1.0, self.brush_size / 4.0)
            steps = int(distance / step) + 1

            if steps > 100:  # 限制最大步数
                step = distance / 100
                steps = 100

            # 获取半圆形剖面
            profile = self._create_half_circle_brush_profile(self.brush_size, self.brush_depth)

            # 批量处理所有点
            for i in range(steps + 1):
                t = i / steps
                x = int(x1 + dx * t)
                y = int(y1 + dy * t)
                self._apply_brush_stroke_with_profile(x, y, profile, self.eraser_mode)

            # 只在最后更新一次
            self.update()

        except Exception as e:
            print(f"draw_line错误: {e}")

    def resizeEvent(self, event):
        """重写resize事件，保持固定尺寸"""
        super().resizeEvent(event)
        self.setFixedSize(self.canvas_width, self.canvas_height)

    def update_canvas_size(self):
        """更新画布尺寸以匹配显示区域"""
        try:
            if self.image.width() != self.width() or self.image.height() != self.height():
                new_pixmap = QPixmap(self.width(), self.height())
                new_pixmap.fill(Qt.white)

                if not self.image.isNull():
                    painter = QPainter(new_pixmap)
                    scaled_pixmap = self.image.scaled(self.width(), self.height(),
                                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    x_offset = (self.width() - scaled_pixmap.width()) // 2
                    y_offset = (self.height() - scaled_pixmap.height()) // 2
                    painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                    painter.end()

                self.image = new_pixmap
                self.update()
        except Exception as e:
            print(f"更新画布尺寸错误: {e}")

    def heightForWidth(self, width):
        """重写heightForWidth以保持固定比例"""
        return 300

    def paintEvent(self, event):
        """绘制事件"""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            if self.is_enabled:
                painter.fillRect(self.rect(), Qt.white)
            else:
                painter.fillRect(self.rect(), QColor(245, 245, 245))

            if not self.image.isNull():
                painter.drawPixmap(0, 0, self.image)

            if self.show_grid:
                if self.is_enabled:
                    painter.setPen(QPen(QColor(200, 200, 200, 100), 1))
                else:
                    painter.setPen(QPen(QColor(180, 180, 180, 80), 1))

                for y in range(0, self.height(), self.grid_size):
                    painter.drawLine(0, y, self.width(), y)
                for x in range(0, self.width(), self.grid_size):
                    painter.drawLine(x, 0, x, self.height())

            if self.is_enabled:
                painter.setPen(QPen(QColor(52, 152, 219), 2))
            else:
                painter.setPen(QPen(QColor(204, 204, 204), 2))

            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

            if not self.is_enabled:
                painter.setPen(QPen(QColor(153, 153, 153), 1))
                painter.drawText(self.rect(), Qt.AlignCenter, "手动画板已禁用\n（仅在表面划痕模式下可用）")

        except Exception as e:
            print(f"绘制事件错误: {e}")

    def to_canvas_coordinates(self, widget_point):
        """将窗口坐标转换为画布坐标"""
        x = int(widget_point.x() * self.image.width() / self.width())
        y = int(widget_point.y() * self.image.height() / self.height())

        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            return QPoint(x, y)
        return QPoint()

    def to_window_coordinates(self, canvas_point):
        """将画布坐标转换为窗口坐标"""
        x = int(canvas_point.x() * self.width() / self.image.width())
        y = int(canvas_point.y() * self.height() / self.image.height())
        return QPoint(x, y)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.is_enabled:
            self.drawing = True
            canvas_pos = self.to_canvas_coordinates(event.pos())
            self.last_point = canvas_pos
            self.current_point = event.pos()
            self.draw_point(canvas_pos)
            self.update()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        self.current_point = event.pos()

        if self.drawing and (event.buttons() & Qt.LeftButton) and self.is_enabled:
            current_canvas_pos = self.to_canvas_coordinates(event.pos())

            if (0 <= current_canvas_pos.x() < self.image.width() and
                    0 <= current_canvas_pos.y() < self.image.height()):
                self.draw_line(self.last_point, current_canvas_pos)
                self.last_point = current_canvas_pos

        self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.is_enabled:
            self.drawing = False
            self.update()

    def clear_canvas(self):
        """清除画布"""
        if not self.image.isNull():
            self.image.fill(Qt.white)
            self.brush_profile_cache = {}  # 清除缓存
            self.update()

    def set_brush_size(self, size):
        """设置画笔粗细"""
        self.brush_size = size
        # 清除缓存，因为画笔大小改变了
        self.brush_profile_cache = {}
        self.update()

    def set_brush_depth(self, depth):
        """设置画笔深度 - 修改为0-1.0范围"""
        # 深度范围变为：0.0-1.0
        self.brush_depth = max(0.0, min(1.0, depth))
        self.brush_profile_cache = {}
        self.update()

    def get_brush_depth(self):
        """获取当前画笔深度"""
        return self.brush_depth

    def set_eraser_mode(self, enabled):
        """设置橡皮擦模式"""
        self.eraser_mode = enabled
        self.update()

    def setEnabled(self, enabled):
        """设置画板的启用状态"""
        super().setEnabled(enabled)
        self.is_enabled = enabled

        if not enabled:
            self.drawing = False

        if enabled:
            self.setStyleSheet("""
                DrawingCanvas {
                    background-color: white; 
                    border: 2px solid #3498db; 
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                DrawingCanvas {
                    background-color: #f5f5f5; 
                    border: 2px solid #cccccc; 
                    border-radius: 5px;
                    color: #999999;
                }
            """)

    def get_scratches_mask(self, target_size=(200, 200)):
        """获取划痕掩码"""
        if self.image.isNull() or not self.is_enabled:
            return np.zeros(target_size, dtype=np.float32)

        try:
            qimage = self.image.toImage()
            mask = np.zeros(target_size, dtype=np.float32)

            scale_x = qimage.width() / target_size[0]
            scale_y = qimage.height() / target_size[1]

            for y in range(target_size[1]):
                for x in range(target_size[0]):
                    src_x = int(x * scale_x)
                    src_y = int(y * scale_y)

                    if 0 <= src_x < qimage.width() and 0 <= src_y < qimage.height():
                        pixel_color = QColor(qimage.pixel(src_x, src_y))
                        gray = 0.299 * pixel_color.red() + 0.587 * pixel_color.green() + 0.114 * pixel_color.blue()
                        depth = (255 - gray) / 255.0
                        mask[y, x] = depth

            return mask

        except Exception as e:
            print(f"获取划痕掩码时出错: {e}")
            return np.zeros(target_size, dtype=np.float32)

    def save_canvas_state(self):
        """保存画板状态"""
        try:
            if not self.image.isNull():
                return self.image.copy()
            return None
        except Exception as e:
            print(f"保存画板状态错误: {e}")
            return None

    def restore_canvas_state(self, image_data):
        """恢复画板状态"""
        try:
            if image_data and not image_data.isNull():
                self.image = image_data
                self.update()
        except Exception as e:
            print(f"恢复画板状态错误: {e}")

    def get_canvas_image(self):
        """获取画布图像"""
        return self.image

    def set_canvas_image(self, pixmap):
        """设置画布图像"""
        if not pixmap.isNull():
            self.image = pixmap
            self.update()

    def has_drawing(self):
        """检查画布上是否有绘制内容"""
        if self.image.isNull():
            return False

        image = self.image.toImage()
        for x in range(image.width()):
            for y in range(image.height()):
                pixel_color = QColor(image.pixel(x, y))
                if pixel_color != Qt.white:
                    return True
        return False

    def reset_canvas(self):
        """重置画布到初始状态"""
        self.clear_canvas()
        self.brush_size = 15
        self.brush_depth = 0.15
        self.eraser_mode = False
        self.drawing = False
        self.brush_profile_cache = {}
        self.update()

    def get_canvas_size(self):
        """获取画布尺寸"""
        return self.image.width(), self.image.height()

    def set_canvas_background(self, color):
        """设置画布背景颜色"""
        if not self.image.isNull():
            self.image.fill(color)
            self.update()

    def enable_grid(self, enabled):
        """启用/禁用网格显示"""
        self.show_grid = enabled
        self.update()

    def set_grid_size(self, size):
        """设置网格大小"""
        self.grid_size = max(5, min(100, size))
        self.update()

    def get_drawing_info(self):
        """获取绘制信息"""
        info = {
            'brush_size': self.brush_size,
            'brush_depth': self.brush_depth,
            'eraser_mode': self.eraser_mode,
            'is_enabled': self.is_enabled,
            'canvas_size': (self.image.width(), self.image.height()),
            'has_drawing': self.has_drawing(),
            'show_grid': self.show_grid,
            'grid_size': self.grid_size
        }
        return info

    def export_canvas_image(self, filename, format="PNG"):
        """导出画布图像到文件"""
        try:
            if not self.image.isNull():
                self.image.save(filename, format)
                return True
            return False
        except Exception as e:
            print(f"导出画布图像错误: {e}")
            return False

    def import_canvas_image(self, filename):
        """从文件导入画布图像"""
        try:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.canvas_width, self.canvas_height,
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image = scaled_pixmap
                self.update()
                return True
            return False
        except Exception as e:
            print(f"导入画布图像错误: {e}")
            return False

    def set_canvas_size(self, width, height):
        """设置画布尺寸（内部使用，外部通过固定尺寸控制）"""
        if width > 0 and height > 0:
            self.canvas_width = width
            self.canvas_height = height

            new_pixmap = QPixmap(width, height)
            new_pixmap.fill(Qt.white)

            if not self.image.isNull():
                painter = QPainter(new_pixmap)
                scaled_pixmap = self.image.scaled(width, height,
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x_offset = (width - scaled_pixmap.width()) // 2
                y_offset = (height - scaled_pixmap.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                painter.end()

            self.image = new_pixmap
            self.update()

    def get_drawing_statistics(self):
        """获取绘制统计信息"""
        if self.image.isNull() or not self.has_drawing():
            return {
                'total_pixels': 0,
                'drawn_pixels': 0,
                'coverage_percentage': 0.0,
                'average_intensity': 0.0
            }

        image = self.image.toImage()
        total_pixels = image.width() * image.height()
        drawn_pixels = 0
        total_intensity = 0.0

        for x in range(image.width()):
            for y in range(image.height()):
                pixel_color = QColor(image.pixel(x, y))
                if pixel_color != Qt.white:
                    drawn_pixels += 1
                    gray_value = (pixel_color.red() + pixel_color.green() + pixel_color.blue()) / 3.0 / 255.0
                    total_intensity += gray_value

        coverage_percentage = (drawn_pixels / total_pixels) * 100
        average_intensity = total_intensity / drawn_pixels if drawn_pixels > 0 else 0.0

        return {
            'total_pixels': total_pixels,
            'drawn_pixels': drawn_pixels,
            'coverage_percentage': coverage_percentage,
            'average_intensity': average_intensity
        }