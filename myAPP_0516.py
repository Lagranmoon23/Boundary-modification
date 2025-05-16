import sys
import numpy as np
import SimpleITK as sitk
import os
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QDesktopWidget,
                             QHBoxLayout, QVBoxLayout, QMenu, QAction, QFileDialog,
                             QLabel, QSizePolicy, QCheckBox, QScrollArea, QSlider,
                             QDoubleSpinBox,QSplitter) # Import QDoubleSpinBox
from PyQt5.QtGui import QIcon, QImage, QPixmap, QKeyEvent, QPainter, QPen, QBrush
from PyQt5.QtCore import QPoint, Qt, QEvent, pyqtSignal, QRect, QSize, QPointF

from skimage import measure
import warnings
warnings.filterwarnings("ignore", module="skimage")

# Import OpenCV for contour filling
try:
    import cv2
except ImportError:
    print("Warning: OpenCV is not installed. Mask filling functionality will not be available.")
    print("Please install it using: pip install opencv-python")
    cv2 = None

# --- Helper function to extract 3D surface mesh (using skimage) ---
# This function can be outside the MainWindow class (place it at the top level of the script)
import SimpleITK as sitk
import numpy as np
from skimage import measure # Ensure skimage is imported

def extract_surface_mesh(sitk_image: sitk.Image, level: float = 0.5):
    """
    从 SimpleITK 图像中提取等值面网格。

    Args:
        sitk_image: 输入的 SimpleITK 图像（期望是 3D）。
        level: 用于提取等值面的阈值。对于二值 mask，通常设置为 0.5。

    Returns:
        一个包含顶点 (vertices) 和面 (faces) 的元组，如果提取失败则返回 None。
    """
    if sitk_image is None or sitk_image.GetDimension() < 3:
        print("Error: Input image must be 3D for surface extraction.")
        return None

    try:
        # SimpleITK 使用 ZYX 顺序，skimage marchin_cubes 期望 (depth, height, width)
        # GetArrayFromImage 转换为 NumPy 的 Z, Y, X (depth, height, width)
        image_array = sitk.GetArrayFromImage(sitk_image)

        # 执行 Marching Cubes 算法
        # 返回的 vertices 是体素坐标，顺序通常是 (y, x, z) -> (row, column, slice) 对应 (height, width, depth)
        # VTK 通常使用 (x, y, z)
        # 我们在 convert_skimage_mesh_to_vtk 中处理坐标转换
        vertices, faces, normals, values = measure.marching_cubes(
            image_array,
            level=level,
            step_size=1, # 控制采样密度
            allow_degenerate=True
        )

        print(f"Marching Cubes extracted {len(vertices)} vertices and {len(faces)} faces.")

        return vertices, faces

    except Exception as e:
        print(f"Error during Marching Cubes extraction: {e}")
        return None

# --- Helper function for pixel type check ---
# --- Helper function for pixel type check ---
def is_integer_pixel_type(pixel_id):
    """ Checks if the SimpleITK pixel ID corresponds to an integer type. """
    try:
        # SimpleITK 的像素 ID 枚举通常将整数类型排在前面。
        # 我们可以检查 ID 是否小于第一个浮点类型的 ID。
        # 这是一个启发式方法，通常有效。更严谨的方法是与所有已知的整数类型 ID 进行比较，但这会比较冗长。
        # 获取已知浮点类型的像素 ID
        sitk_float32_id = sitk.sitkFloat32
        # 如果 pixel_id 小于 float32 ID，它很可能是整数类型。
        # 这假设了枚举顺序。
        is_integer = pixel_id < sitk_float32_id
        # 您可以取消下面这行的注释，用于调试查看像素 ID
        # print(f"Pixel ID: {pixel_id}, sitkFloat32 ID: {sitk_float32_id}, Is Integer: {is_integer}")
        return is_integer
    except AttributeError:
        # 如果 sitkFloat32 不可用或枚举顺序不同时的备用方案
        print("Warning: Could not reliably check pixel type using sitkFloat32. Assuming not integer.")
        return False # 如果无法可靠检查，则假设不是整数
    except Exception as e:
        print(f"Error checking pixel type: {e}")
        return False

# --- Custom Image Display Label (Multi-layer composition, interaction, contour display, dragging) ---
class SliceDisplayLabel(QLabel):
    composition_changed = pyqtSignal(list)

    # Default Constants for local deformation
    DEFAULT_INFLUENCE_RADIUS_IMAGE = 30.0 # Default influence radius in image pixels
    DEFAULT_FALLOFF_POWER = 2.0         # Default power for the fall-off function


    def __init__(self, parent=None):
        super().__init__(parent)
        self._composited_layers = []
        self._visible_images_data_refs = [] # References to image_data dicts from MainWindow

        self._pan_offset = QPoint(0, 0)
        self._zoom_factor = 1.0
        self._last_mouse_pos = QPoint() # Last mouse position for pan/drag delta
        self._is_panning = False
        self._is_zooming = False
        self._is_dragging = False # Flag for dragging a contour point

        self._scaled_pixmap_size = QSize(0, 0)

        # Stores (layer_index_in_composited_layers, contour_index, point_index) of the selected point
        self._selected_point = None

        # Instance variables for editable parameters, initialized from defaults
        # These will be updated by MainWindow via set methods
        self._influence_radius = self.DEFAULT_INFLUENCE_RADIUS_IMAGE
        self._falloff_power = self.DEFAULT_FALLOFF_POWER # Falloff power is not user editable yet

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid green;")
        self.setText("Axial (Z) View")

        self.setMouseTracking(True)
        self.setEnabled(True)

    # --- Public methods to set parameters from outside ---
    def set_influence_radius(self, radius: float):
        """Sets the influence radius for point dragging."""
        if radius > 0:
            self._influence_radius = radius
            # print(f"Influence radius set to {self._influence_radius}")

    def set_falloff_power(self, power: float):
        """Sets the fall-off power for point dragging."""
        if power >= 0: # Power can be 0 (no falloff within radius)
            self._falloff_power = power
            # print(f"Falloff power set to {self._falloff_power}")

    # --- New method to clear selection and dragging state ---
    def clear_selection_and_dragging(self):
        """Clears the currently selected point and dragging state."""
        # Check if a selection or dragging was active to avoid unnecessary updates
        if self._is_dragging or self._selected_point is not None:
            self._is_dragging = False
            self._selected_point = None
            # Trigger a repaint to remove the visual highlight of the selected point
            self.update()


    # --- Keep existing methods below ---

    # --- Public method to set composition data from MainWindow ---
    def set_composition_data(self, visible_images_data_list):
        """
        设置当前需要显示的可见图像数据列表。
        由 MainWindow 调用，用于更新 SliceDisplayLabel 的显示内容。

        Args:
            visible_images_data_list: 一个列表，包含当前所有可见图像的 image_data 字典引用。
        """
        # 存储旧的复合图层列表，用于判断是否是第一次加载
        old_composited_layers = self._composited_layers

        # 更新内部存储的可见图像数据引用列表
        self._visible_images_data_refs = visible_images_data_list

        # 在合成数据变化时，清除当前选中的点和拖动状态
        # 这是重要的，因为图层顺序或内容可能改变，旧的选中点索引可能无效
        self.clear_selection_and_dragging() # 确保您有 clear_selection_and_dragging 方法


        if not self._visible_images_data_refs:
            # 如果没有可见图像数据，清空内部状态和显示
            self._composited_layers = []
            self._scaled_pixmap_size = QSize(0, 0)
            self._pan_offset = QPoint(0, 0) # 没有图像时重置平移
            self._zoom_factor = 1.0       # 没有图像时重置缩放
            self.setPixmap(QPixmap()) # 清空显示的 Pixmap
            self.setText("Axial (Z) View") # 显示默认文本
            print("SliceDisplayLabel: No visible images to compose.")

        else:
             # 如果有可见图像数据，但之前没有复合图层 (第一次加载有效图像)
             if not old_composited_layers:
                  # 可以在这里选择重置平移和缩放，使新加载的图像居中并以默认缩放显示
                  # 如果您希望在加载新图像时视图不跳动，可以注释掉下面两行
                  self._pan_offset = QPoint(0, 0) # 第一次加载时重置平移
                  self._zoom_factor = 1.0       # 第一次加载时重置缩放
                  print("SliceDisplayLabel: First effective image loaded, resetting pan/zoom.")


        # 调用内部方法 _update_composition 来根据新的数据列表重新合成图像并绘制
        self._update_composition()

        # self.composition_changed.emit(self._visible_images_data_refs) # _update_composition 会调用这个信号，避免重复

    def _update_composition(self):
        """
        更新复合图像层，根据可见图像的切片、透明度、窗宽/窗位创建 QPixmap。
        并绘制轮廓。现在窗宽/窗位应用于所有图像。
        """
        self._composited_layers = []
        current_composite_original_size = QSize(0, 0)

        if not self._visible_images_data_refs:
            # 如果没有可见图像，清空显示
            self._scaled_pixmap_size = QSize(0, 0)
            self.setPixmap(QPixmap())
            self.setText("Axial (Z) View") # 显示默认文本
            self.update() # 触发重绘
            # 发出信号，通知外部（例如 MainWindow）合成图层已变化
            self.composition_changed.emit(self._visible_images_data_refs)
            return

        # 遍历所有可见的图像数据引用
        for img_data_index_in_visible_list, img_data in enumerate(self._visible_images_data_refs):
             sitk_image = img_data.get('sitk_image')
             # 获取当前显示的切片索引，如果未设置则默认为 0
             slice_index = img_data.get('current_slice_index', 0)
             # 获取透明度，如果未设置则默认为 1.0
             opacity = img_data.get('opacity', 1.0)
             # 直接获取 window_level 和 window_width，不再根据 is_mask 判断是否为 None
             window_level = img_data.get('window_level', None)
             window_width = img_data.get('window_width', None)
             is_mask = img_data.get('is_mask', False) # 保留 is_mask 状态，可能用于轮廓颜色等


             if not sitk_image:
                 # 如果 SimpleITK 图像对象不存在，跳过此层
                 continue

             try:
                 # 将 SimpleITK 图像转换为 NumPy 数组
                 image_array = sitk.GetArrayFromImage(sitk_image)
             except Exception as e:
                 print(f"Error getting array from SITK image for composition: {e}")
                 continue # 发生错误时跳过此层

             img_dimension = image_array.ndim

             slice_2d_original_data = None
             contours_for_slice = []

             # 根据图像维度获取当前切片的 2D 数据
             if img_dimension == 2:
                  # 如果是 2D 图像，直接使用整个数组
                  slice_2d_original_data = image_array
                  # 对于 2D 图像，我们只处理 slice_index = 0
                  if slice_index != 0:
                       print(f"Warning: 2D image {img_data.get('path', 'Unknown')} has slice_index {slice_index}, but only slice 0 is used.")
                       # Optional: you might want to handle this differently or skip
                       # continue # Or continue to next image if slice_index is non-zero


             elif img_dimension >= 3:
                  z_size = image_array.shape[0] # Z 维度通常是第一个维度 (深度)
                  # 确保切片索引在有效范围内
                  clamped_slice_index = max(0, min(slice_index, z_size - 1))
                  # 获取指定切片的 2D NumPy 数组
                  slice_2d_original_data = image_array[clamped_slice_index, :, :]


             # 如果成功获取了 2D 切片数据
             if slice_2d_original_data is not None:
                 # 调用 get_qpixmap_from_slice_array 将 NumPy 数组转换为 QPixmap
                 # 直接传递 window_level 和 window_width，无论是否为 mask
                 pixmap = self.get_qpixmap_from_slice_array(
                     slice_2d_original_data,
                     window_level, # <-- 直接传递 window_level 值
                     window_width  # <-- 直接传递 window_width 值
                 )

                 # 获取当前切片的 2D 轮廓数据，如果不存在则为空列表
                 contours_for_slice = img_data.get('contours', {}).get(slice_index, [])

                 # 如果成功创建了 QPixmap
                 if not pixmap.isNull():
                     # 将当前图层信息添加到复合图层列表中
                     # 存储：QPixmap, 透明度, 是否为 Mask, 轮廓, 原始 image_data 在 visible_images_data_refs 中的索引
                     self._composited_layers.append((pixmap, opacity, is_mask, contours_for_slice, img_data_index_in_visible_list))

                     # 如果这是第一个有效的图层，记录其原始尺寸
                     if current_composite_original_size.isEmpty():
                         current_composite_original_size = pixmap.size()

             else:
                 print(f"Warning: Could not get 2D slice data for {img_data.get('path', 'Unknown')} at slice {slice_index}.")


        # 更新缩放后 Pixmap 的尺寸 (基于第一个有效图层的原始尺寸和当前的缩放因子)
        self._scaled_pixmap_size = current_composite_original_size

        # 根据是否有复合图层来设置 Label 的文本
        if self._composited_layers:
             self.setText("") # 有图层时清空文本
        else:
             self.setText("No visible images") # 没有可见图层时显示默认文本

        # 在复合图层更新后，检查当前选中的点是否仍然有效
        # 如果选中的点所属的图层索引超出了新的复合图层列表范围，则清除选中状态
        if self._selected_point is not None:
             layer_index_in_composited = self._selected_point[0]
             if layer_index_in_composited >= len(self._composited_layers):
                  self._selected_point = None # 清除选中状态


        # 触发 Label 的 paintEvent 进行重绘
        self.update()
        # 发出信号，通知外部合成图层已更新
        self.composition_changed.emit(self._visible_images_data_refs)


    def get_qpixmap_from_slice_array(self, slice_array, window_level=None, window_width=None):
        if slice_array is None or slice_array.size == 0:
            return QPixmap()
        try:
            processed_slice = slice_array.astype(np.float64)

            if window_level is not None and window_width is not None:
                 if window_width <= 0:
                     level_val = float(window_level)
                     mean_val = np.mean(processed_slice) if processed_slice.size > 0 else 0
                     processed_slice = np.full(processed_slice.shape, 255.0 if level_val <= mean_val else 0.0)
                 else:
                     min_val = float(window_level) - float(window_width) / 2.0
                     max_val = float(window_level) + float(window_width) / 2.0
                     processed_slice = (processed_slice - min_val) / float(window_width)
                     processed_slice = np.clip(processed_slice * 255.0, 0, 255)
            else:
                 img_min = np.min(processed_slice) if processed_slice.size > 0 else 0
                 img_max = np.max(processed_slice) if processed_slice.size > 0 else 0
                 if img_max - img_min == 0:
                      processed_slice = np.full(processed_slice.shape, 128.0)
                 else:
                      processed_slice = 255.0 * (processed_slice - img_min) / (img_max - img_min)

            scaled_slice = processed_slice.astype(np.uint8)

            height, width = scaled_slice.shape
            bytesPerLine = width * 1
            if not scaled_slice.flags['C_CONTIGUOUS']:
                 scaled_slice = np.ascontiguousarray(scaled_slice)

            q_image = QImage(scaled_slice.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            return pixmap
        except Exception as e:
            print(f"SliceDisplayLabel: Error converting slice to QPixmap: {e}")
            if slice_array is not None:
                 print(f"  Slice shape: {slice_array.shape}, dtype: {slice_array.dtype}")
                 print(f"  Window Level: {window_level}, Window Width: {window_width}")
            return QPixmap()


    def image_to_screen(self, point_image: QPointF) -> QPointF:
        if self._scaled_pixmap_size.isEmpty() or self._zoom_factor <= 0:
            return QPointF()

        scaled_width = self._scaled_pixmap_size.width() * self._zoom_factor
        scaled_height = self._scaled_pixmap_size.height() * self._zoom_factor

        target_x = self.contentsRect().left() + self._pan_offset.x() + (self.contentsRect().width() - scaled_width) // 2
        target_y = self.contentsRect().top() + self._pan_offset.y() + (self.contentsRect().height() - scaled_height) // 2
        target_origin = QPointF(target_x, target_y)

        scale_x = scaled_width / self._scaled_pixmap_size.width() if self._scaled_pixmap_size.width() > 0 else 1.0
        scale_y = scaled_height / self._scaled_pixmap_size.height() if self._scaled_pixmap_size.height() > 0 else 1.0

        screen_x = target_origin.x() + point_image.x() * scale_x
        screen_y = target_origin.y() + point_image.y() * scale_y

        return QPointF(screen_x, screen_y)


    def screen_to_image(self, point_screen: QPointF) -> QPointF:
        if self._scaled_pixmap_size.isEmpty() or self._zoom_factor <= 0:
            return QPointF()

        scaled_width = self._scaled_pixmap_size.width() * self._zoom_factor
        scaled_height = self._scaled_pixmap_size.height() * self._zoom_factor

        target_x = self.contentsRect().left() + self._pan_offset.x() + (self.contentsRect().width() - scaled_width) // 2
        target_y = self.contentsRect().top() + self._pan_offset.y() + (self.contentsRect().height() - scaled_height) // 2
        target_origin = QPointF(target_x, target_y)

        scale_x = scaled_width / self._scaled_pixmap_size.width() if self._scaled_pixmap_size.width() > 0 else 0.0
        scale_y = scaled_height / self._scaled_pixmap_size.height() if self._scaled_pixmap_size.height() > 0 else 0.0

        image_x = (point_screen.x() - target_origin.x()) / scale_x if scale_x != 0 else 0.0
        image_y = (point_screen.y() - target_origin.y()) / scale_y if scale_y != 0 else 0.0

        return QPointF(image_x, image_y)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.contentsRect(), self.palette().window())

        if self._composited_layers and not self._scaled_pixmap_size.isEmpty():
            scaled_width = self._scaled_pixmap_size.width() * self._zoom_factor
            scaled_height = self._scaled_pixmap_size.height() * self._zoom_factor
            scaled_pixmap_size_int = QSize(int(scaled_width), int(scaled_height))

            target_x = self.contentsRect().left() + self._pan_offset.x() + (self.contentsRect().width() - scaled_width) // 2
            target_y = self.contentsRect().top() + self._pan_offset.y() + (self.contentsRect().height() - scaled_height) // 2
            target_origin = QPoint(int(target_x), int(target_y))


            for layer_index_in_composited, (pixmap, opacity, is_mask, contours, img_data_original_index) in enumerate(self._composited_layers):
                if not pixmap.isNull():
                    painter.setOpacity(opacity)
                    painter.drawPixmap(target_origin, pixmap.scaled(scaled_pixmap_size_int, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                    # --- 绘制边界 ---
                    if contours:
                        contour_color = Qt.red
                        point_color = Qt.blue

                        painter.setOpacity(1.0)
                        painter.setPen(QPen(contour_color, 1.5))

                        for contour_index, contour in enumerate(contours):
                             if contour.shape[0] > 1:
                                 contour_points_screen = [self.image_to_screen(QPointF(pt[0], pt[1])) for pt in contour]
                                 if contour_points_screen:
                                     painter.drawPolyline(*contour_points_screen)

                                 # Draw points
                                 painter.setPen(QPen(point_color, 1.0))
                                 painter.setBrush(QBrush(point_color))
                                 point_size = 3

                                 for point_index, point_screen in enumerate(contour_points_screen):
                                     is_this_the_selected_point = False
                                     if self._selected_point is not None and \
                                        self._selected_point[0] == layer_index_in_composited and \
                                        self._selected_point[1] == contour_index and \
                                        self._selected_point[2] == point_index:
                                          is_this_the_selected_point = True

                                     painter.setPen(QPen(Qt.yellow if is_this_the_selected_point else point_color, 1.0))
                                     painter.setBrush(QBrush(Qt.yellow if is_this_the_selected_point else point_color))
                                     current_point_size = 5 if is_this_the_selected_point else point_size
                                     painter.drawEllipse(point_screen, current_point_size, current_point_size)


        else:
            painter.setOpacity(1.0)
            painter.drawText(self.contentsRect(), Qt.AlignCenter, self.text())

        painter.end()


    def mousePressEvent(self, event):
        if event.modifiers() == Qt.ShiftModifier:
            if self._composited_layers:
                if event.button() == Qt.LeftButton:
                    self._is_panning = True
                    self._last_mouse_pos = event.pos()
                    event.accept()
                elif event.button() == Qt.RightButton:
                    self._is_zooming = True
                    self._last_mouse_pos = event.pos()
                    event.accept()
                else:
                     super().mousePressEvent(event)
            else:
                 super().mousePressEvent(event)
        elif event.button() == Qt.LeftButton:
             self._selected_point = None
             self._is_dragging = False
             click_pos_screen = event.pos()

             if self._composited_layers and not self._scaled_pixmap_size.isEmpty():
                  hit_tolerance_screen = 8

                  for layer_index_in_composited, (pixmap, opacity, is_mask, contours, img_data_original_index) in enumerate(self._composited_layers):
                       if contours:
                            for contour_index, contour in enumerate(contours):
                                 for point_index, point_image_coords in enumerate(contour):
                                     point_screen = self.image_to_screen(QPointF(point_image_coords[0], point_image_coords[1]))

                                     delta_vector = point_screen - QPointF(click_pos_screen)
                                     distance = (delta_vector.x()**2 + delta_vector.y()**2)**0.5

                                     if distance <= hit_tolerance_screen:
                                         print(f"Hit point layer={img_data_original_index} (composited index={layer_index_in_composited}), contour={contour_index}, point={point_index}")
                                         self._selected_point = (layer_index_in_composited, contour_index, point_index)
                                         self._is_dragging = True
                                         self._last_mouse_pos = event.pos()
                                         self.update()
                                         event.accept()
                                         return

             super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if self._is_panning:
            if event.buttons() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier:
                 delta = event.pos() - self._last_mouse_pos
                 self._pan_offset += delta
                 self._last_mouse_pos = event.pos()
                 self.update()
                 event.accept()
        # --- Point Dragging Logic with Local Deformation (No Mask Pixel Update Here) ---
        elif self._is_dragging and event.buttons() == Qt.LeftButton:
            layer_index_in_composited, contour_index, point_index = self._selected_point

            if layer_index_in_composited < len(self._composited_layers):
                img_data_ref = self._visible_images_data_refs[layer_index_in_composited]
                current_slice_index = img_data_ref.get('current_slice_index', 0)

                if current_slice_index in img_data_ref.get('contours', {}):
                     contours_for_slice = img_data_ref['contours'][current_slice_index]

                     if contour_index < len(contours_for_slice) and point_index < len(contours_for_slice[contour_index]):

                         start_pos_image = self.screen_to_image(QPointF(self._last_mouse_pos))
                         current_pos_image = self.screen_to_image(QPointF(event.pos()))
                         delta_image_qpointf = current_pos_image - start_pos_image
                         delta_image_np = np.array([delta_image_qpointf.x(), delta_image_qpointf.y()])

                         dragged_point_image_coords = np.copy(contours_for_slice[contour_index][point_index])

                         # --- Apply Local Elastic Deformation to points in the same contour ---
                         # Use the instance variable for influence radius
                         influence_radius = self._influence_radius
                         # Use the instance variable for falloff power (even if not user editable yet)
                         falloff_power = self._falloff_power

                         contour_array_to_modify = contours_for_slice[contour_index]

                         for p_idx in range(len(contour_array_to_modify)):
                             p_coords = contour_array_to_modify[p_idx]

                             dist = np.linalg.norm(p_coords - dragged_point_image_coords)

                             influence_factor = 0.0
                             if dist < influence_radius:
                                 influence_factor = (1.0 - (dist / influence_radius)) ** falloff_power

                             movement_vector = delta_image_np * influence_factor

                             contour_array_to_modify[p_idx] += movement_vector

                         # --- End Local Elastic Deformation ---

                         self._last_mouse_pos = event.pos()

                         self.update() # Update contour drawing
                         event.accept()


        elif self._is_zooming:
            if event.buttons() == Qt.RightButton and event.modifiers() == Qt.ShiftModifier:
                 delta_y_screen = event.pos().y() - self._last_mouse_pos.y()
                 zoom_step = delta_y_screen * 0.005
                 self._zoom_factor += zoom_step
                 self._zoom_factor = max(0.1, min(self._zoom_factor, 20.0))
                 self._last_mouse_pos = event.pos()
                 self.update()
                 event.accept()
        else:
            super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        was_dragging = self._is_dragging
        selected_point_info_at_release = self._selected_point

        if self._is_panning and event.button() == Qt.LeftButton:
            self._is_panning = False
            event.accept()
        elif self._is_zooming and event.button() == Qt.RightButton:
             self._is_zooming = False
             event.accept()
        # --- End Point Dragging ---
        elif was_dragging and event.button() == Qt.LeftButton:
            self._is_dragging = False
            self._selected_point = None # Deselect point visually

            # --- Trigger Mask Data Update AFTER Drag Ends ---
            if selected_point_info_at_release is not None:
                 layer_index_in_composited, contour_index, point_index = selected_point_info_at_release

                 if layer_index_in_composited < len(self._composited_layers):
                      img_data_ref = self._visible_images_data_refs[layer_index_in_composited]
                      current_slice_index = img_data_ref.get('current_slice_index', 0)

                      self._update_mask_slice_pixels(img_data_ref, current_slice_index)

            self._update_composition() # Update display to show potentially updated pixels and clear selection
            event.accept()

        else:
            super().mouseReleaseEvent(event)

    # --- Method to Update Mask Pixel Data ---
    def _update_mask_slice_pixels(self, img_data_ref, slice_index_to_update):
        print(f"Updating mask pixel data for slice {slice_index_to_update}...")
        if cv2 is None:
            print("Mask pixel update requires OpenCV, but it is not available.")
            print("Please install it using: pip install opencv-python")
            return

        sitk_image = img_data_ref.get('sitk_image')
        if sitk_image is None:
            print("Cannot update mask pixels: SimpleITK image object is missing.")
            return

        try:
            image_array_3d = sitk.GetArrayFromImage(sitk_image) # Get a COPY
        except Exception as e:
            print(f"Error getting image array copy for mask update: {e}")
            return

        img_dimension = image_array_3d.ndim

        if img_dimension >= 3:
            if slice_index_to_update < 0 or slice_index_to_update >= image_array_3d.shape[0]:
                 print(f"Error: Invalid slice index {slice_index_to_update} for mask update (dimension {img_dimension}).")
                 return
            slice_shape = image_array_3d.shape[1:]
            slice_dtype = image_array_3d.dtype
            current_slice_array_in_copy = image_array_3d[slice_index_to_update, :, :]
        elif img_dimension == 2:
            if slice_index_to_update != 0:
                 print(f"Error: Invalid slice index {slice_index_to_update} for 2D mask update.")
                 return
            slice_shape = image_array_3d.shape
            slice_dtype = image_array_3d.dtype
            slice_index_to_update = 0
            current_slice_array_in_copy = image_array_3d


        temp_mask_slice = np.zeros(slice_shape, dtype=slice_dtype)

        all_contours_on_slice = img_data_ref.get('contours', {}).get(slice_index_to_update, [])

        contours_cv2_format = [c.reshape(-1, 1, 2).astype(np.int32) for c in all_contours_on_slice if c.shape[0] > 1]

        if contours_cv2_format:
            try:
                fill_color = 1 # Assuming binary mask foreground is 1
                cv2.fillPoly(temp_mask_slice, contours_cv2_format, color=fill_color)

                if temp_mask_slice.shape == current_slice_array_in_copy.shape:
                     current_slice_array_in_copy[:, :] = temp_mask_slice[:, :]
                else:
                     print(f"Shape mismatch during mask update: temp_mask_slice {temp_mask_slice.shape} vs current_slice_array_in_copy {current_slice_array_in_copy.shape}. Mask update failed.")

            except Exception as fill_e:
                print(f"Error filling contours for mask pixel update (slice {slice_index_to_update}): {fill_e}")

        try:
             new_sitk_image = sitk.GetImageFromArray(image_array_3d)
             new_sitk_image.CopyInformation(sitk_image)

             img_data_ref['sitk_image'] = new_sitk_image
             print(f"Mask pixel data updated successfully for slice {slice_index_to_update}.")

        except Exception as create_image_e:
             print(f"Error creating or updating SITK image from modified array: {create_image_e}")
             print("Mask pixel data update failed.")


    # SliceDisplayLabel's own wheelEvent - passes event up
    # This method itself doesn't contain the slicing logic
    def wheelEvent(self, event):
        # Existing wheel event for slicing - just passes the event up to the parent (MainWindow)
        # Clearing selection and dragging state on wheel is handled by MainWindow's wheelEvent now
        super().wheelEvent(event)


# --- MainWindow Class ---
class MainWindow(QWidget):
    IMAGE_FILE_FILTERS = "医学图像文件 (*.nrrd *.mha);;所有文件 (*)"

    def __init__(self):
        super().__init__()
        self.loaded_images = []
        self.initUI()

    def initUI(self):
        print("MainWindow.initUI: Starting UI initialization...")

        # --- 设置窗口标题、位置和图标 ---
        self.setWindowTitle("边界修改与 3D 显示") # 修改窗口标题
        self.setGeometry(100, 100, 1500, 800)
        # 请检查 'icon/icon1.png' 文件是否存在，路径是否正确，以及是否是有效的图像文件
        # 如果 icon1.png 不存在，可以注释掉这行
        self.setWindowIcon(QIcon("icon1.png"))
        print("MainWindow.initUI: Window title, geometry, icon set.")

        # --- 居中窗口 ---
        print("MainWindow.initUI: Setting window position...")
        cnter_pos = QDesktopWidget().screenGeometry().center()
        self.move(int(cnter_pos.x() - self.width()/2), int(cnter_pos.y() - self.height()/2))
        print("MainWindow.initUI: Window position set.")

        # --- 设置主布局 (创建但不设置) ---
        # 只创建主布局对象，稍后在所有子布局组装好后设置
        print("MainWindow.initUI: Creating main layout object...")
        main_h_layout = QHBoxLayout(self)
        print("MainWindow.initUI: Main layout object created.")


        # --- 设置左侧侧边栏布局 (创建并填充) ---
        # 创建左侧布局和其内部的所有组件 (按钮、衰减半径、图像列表滚动区域)
        print("MainWindow.initUI: Setting up left sidebar layout...")
        left_v_layout = QVBoxLayout()

        # 按钮布局
        button_layout = QHBoxLayout()
        button_import = QPushButton("导入文件") # 修改按钮文本更通用
        button_import.setFixedSize(80, 25) # 调整按钮大小
        button_extract = QPushButton("提取边缘")
        button_extract.setFixedSize(80, 25)
        button_layout.addWidget(button_import)
        button_layout.addWidget(button_extract)

        button_layout.addStretch(1) # 添加伸缩项使按钮靠左

        left_v_layout.addLayout(button_layout) # 将按钮布局添加到左侧垂直布局


        # 衰减半径控制布局
        print("MainWindow.initUI: Setting up radius control...")
        radius_layout = QHBoxLayout()
        radius_label = QLabel("衰减半径(像素):") # 修改标签文本
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setRange(1.0, 500.0) # 设置一个合理的范围
        self.radius_spinbox.setSingleStep(1.0)  # 步长
        self.radius_spinbox.setDecimals(1)      # 小数位数
        # 从 SliceDisplayLabel 的默认常量设置初始值
        # 确保 SliceDisplayLabel.DEFAULT_INFLUENCE_RADIUS_IMAGE 在这个地方可以访问
        self.radius_spinbox.setValue(SliceDisplayLabel.DEFAULT_INFLUENCE_RADIUS_IMAGE)
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_spinbox)
        left_v_layout.addLayout(radius_layout) # 将半径控制布局添加到左侧垂直布局
        print("MainWindow.initUI: Radius control setup complete.")

        # 图像列表滚动区域
        print("MainWindow.initUI: Setting up image list scroll area...")
        self.image_list_scroll_area = QScrollArea()
        self.image_list_scroll_area.setWidgetResizable(True)
        self.image_list_widget = QWidget() # 用于容纳图像列表项的 widget
        self.image_list_layout = QVBoxLayout(self.image_list_widget) # 列表项的布局
        self.image_list_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft) # 顶部左侧对齐
        self.image_list_layout.addStretch(1) # 添加 stretch 将内容顶到顶部
        # 将 image_list_widget 设置为滚动区域的 widget
        self.image_list_scroll_area.setWidget(self.image_list_widget)

        left_v_layout.addWidget(self.image_list_scroll_area, 1) # 将滚动区域添加到左侧垂直布局，并设置伸缩因子为 1
        print("MainWindow.initUI: Left sidebar layout setup complete.")


        # --- 设置右侧显示区域布局 (创建并填充) ---
        # 创建右侧主垂直布局，以及其内部的 QSplitter 和视图控制布局
        print("MainWindow.initUI: Setting up right display area with splitter...")
        right_v_layout = QVBoxLayout()

        # --- 创建一个 QSplitter 用于 2D 和 3D 视图 ---
        # 创建一个水平方向的 Splitter 用于并排显示
        self.view_splitter = QSplitter(Qt.Horizontal)
        print("MainWindow.initUI: QSplitter created.")

        # 创建 2D 视图的容器 Widget 和布局
        print("MainWindow.initUI: Creating 2D view container...")
        self.axial_view_container = QWidget()
        axial_v_layout = QVBoxLayout(self.axial_view_container)
        axial_v_layout.setContentsMargins(0, 0, 0, 0) # 移除布局的边距

        # 创建 SliceDisplayLabel 实例 (原有的 2D 切片显示)
        self.axial_view_label = SliceDisplayLabel()
        self.axial_slice_info_label = QLabel("切片: -- / --")
        self.axial_slice_info_label.setAlignment(Qt.AlignCenter)

        # 将 2D 视图相关的控件添加到 2D 容器布局
        axial_v_layout.addWidget(self.axial_view_label, 1) # 2D 视图 Label 占据主要空间
        axial_v_layout.addWidget(self.axial_slice_info_label) # 信息标签在下方

        # 将 2D 视图容器添加到 QSplitter 中
        self.view_splitter.addWidget(self.axial_view_container)
        print("MainWindow.initUI: 2D view container added to splitter.")


        # --- 添加 VTK 3D 渲染组件 ---
        print("MainWindow.initUI: Setting up VTK 3D rendering component...")
        if vtk is not None and QVTKRenderWindowInteractor is not None:
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.vtk_renderer = vtk.vtkRenderer()
            self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)

            # 设置背景颜色 (可选)
            self.vtk_renderer.SetBackground(0.1, 0.2, 0.3) # 稍深的蓝色背景

            # 初始化 VTK 管线组件
            self.vtk_polydata = None
            self.vtk_mapper = vtk.vtkPolyDataMapper()
            self.vtk_actor = vtk.vtkActor()
            self.vtk_actor.SetMapper(self.vtk_mapper)
            self.vtk_renderer.AddActor(self.vtk_actor)
            self.vtk_actor.VisibilityOff() # 初始时不显示 actor

            # 将 VTK 渲染窗口添加到 QSplitter 中
            self.view_splitter.addWidget(self.vtk_widget)
            # 初始时隐藏 3D 视图，直到有模型被提取
            self.vtk_widget.hide()
            # --- 为 VTK widget 安装事件过滤器 ---
            # 这允许 MainWindow 在 VTK widget 处理事件之前截获它们
            if self.vtk_widget is not None: # 确保 vtk_widget 存在
                 self.vtk_widget.installEventFilter(self)
                 print("MainWindow.initUI: Installed event filter on vtk_widget.")
            print("MainWindow.initUI: VTK 3D rendering component added to splitter.")
        else:
             print("MainWindow.initUI: VTK or QVTKRenderWindowInteractor not available. 3D rendering disabled.")
             self.vtk_widget = None # 如果 VTK 不可用，将 vtk_widget 设置为 None，以便后续检查

        # 可选：设置 QSplitter 初始分割比例
        # 例如，将 2D 和 3D 视图的初始宽度比例设置为 1:1
        # self.view_splitter.setSizes([self.width() // 2, self.width() // 2])


        # 将 QSplitter 添加到右侧的主垂直布局 (占据主要空间)
        right_v_layout.addWidget(self.view_splitter, 1)
        print("MainWindow.initUI: QSplitter added to right layout.")


        # 创建并填充视图切换按钮布局 (功能：控制 3D 视图显示/隐藏)
        print("MainWindow.initUI: Creating view control buttons layout...")
        view_control_layout = QHBoxLayout()
        # 这些按钮现在用于控制 3D 视图的显示/隐藏
        self.btn_show_2d = QPushButton("隐藏 3D 视图") # 修改按钮文本以反映功能
        self.btn_show_3d = QPushButton("显示 3D 视图")
        view_control_layout.addWidget(self.btn_show_2d)
        view_control_layout.addWidget(self.btn_show_3d)
        print("MainWindow.initUI: View control buttons layout created.")

        # 将视图切换按钮布局添加到右侧主垂直布局 (位于 Splitter 下方)
        right_v_layout.addLayout(view_control_layout)

        print("MainWindow.initUI: Right display area setup complete.")


        # --- 将左右布局添加到主布局 ---
        print("MainWindow.initUI: Assembling main layout...")
        # 将左侧垂直布局添加到主水平布局的左侧
        main_h_layout.addLayout(left_v_layout, 1) # 左侧布局占据 1 份空间
        # 将右侧垂直布局添加到主水平布局的右侧
        main_h_layout.addLayout(right_v_layout, 2) # 右侧布局占据 2 份空间
        print("MainWindow.initUI: Main layout assembled.")


        # --- 设置主窗口的布局 ---
        # 在所有子布局和 Widget 都添加到主布局后，设置主布局
        print("MainWindow.initUI: Setting main window layout...")
        self.setLayout(main_h_layout) # <-- 这行代码在这里设置主布局
        print("MainWindow.initUI: Main window layout set successfully.")


        # --- 初始状态：2D 视图可见，3D 视图隐藏 ---
        # 在布局设置完成后，设置 Widget 的初始可见性
        # 在 Splitter 布局中，2D 视图容器默认是可见的
        if self.axial_view_container is not None:
             self.axial_view_container.show() # 确保 2D 视图容器可见

        if self.vtk_widget is not None:
             self.vtk_widget.hide() # 默认 3D 视图隐藏


        # --- 连接信号和槽 ---
        # 在所有 Widget 都创建好并添加到布局后，连接信号和槽
        print("MainWindow.initUI: Connecting signals and slots...")
        # 连接导入、提取边缘、堆叠按钮 (如果保留了)
        button_import.clicked.connect(self.import_file)
        button_extract.clicked.connect(self.extract_edges_action)
        

        # 连接 2D 视图 Label 的信号 (例如 composition_changed)
        self.axial_view_label.composition_changed.connect(self.update_info_labels)

        # 连接衰减半径 SpinBox 的信号
        self.radius_spinbox.valueChanged.connect(self.axial_view_label.set_influence_radius)

        # 连接视图切换按钮的信号到新的方法 (控制 3D 视图显示/隐藏)
        self.btn_show_2d.clicked.connect(self.show_2d_view_elements) # 此方法现在隐藏 3D 视图
        self.btn_show_3d.clicked.connect(self.show_3d_view_elements) # 此方法现在显示 3D 视图 (如果模型已加载)

        # ... 其他列表项中的滑块等的连接，这些是在 import_file 方法中动态创建和连接的 ...


        print("MainWindow.initUI: Signals and slots connected.")


        # --- 设置窗口最小尺寸 ---
        # 在布局设置完成后进行
        print("MainWindow.initUI: Setting minimum size...")
        self.setMinimumSize(self.sizeHint())
        print("MainWindow.initUI: Minimum size set.")


        print("MainWindow.initUI: UI initialization complete.")

    
    def wheelEvent(self, event):
        """
        处理鼠标滚轮事件。
        如果滚轮事件来自 VTK widget，阻止其触发 2D 切片切换。
        否则，如果存在可见的 3D 图像，根据滚轮方向切换切片。
        """


    # --- 实现事件过滤器方法 ---
    def eventFilter(self, watched, event):
        """
        事件过滤器方法，用于截获发送给其他对象的事件。
        这里用来阻止来自 VTK widget 的滚轮事件传播到 MainWindow 的切片逻辑。

        Args:
            watched: 事件的目标对象 (被观察的对象)。
            event: 被截获的事件。

        Returns:
            bool: True 表示事件已被处理并停止传播，False 表示事件继续正常处理和传播。
        """
        # 检查被观察的对象是否是 VTK widget，并且事件类型是否是滚轮事件 (QEvent.Wheel)
        if watched == self.vtk_widget and event.type() == QEvent.Wheel:
            # 如果是来自 VTK widget 的滚轮事件，我们截获它
            # 我们假设 VTK widget 自己会处理缩放，我们在这里阻止它影响 MainWindow 的切片逻辑
            # 接受事件，阻止其进一步传播 (例如传播到 MainWindow 的 wheelEvent 方法)
            # print(f"Event filter: Intercepted Wheel event from VTK widget. Event type: {event.type()}") # 可选的调试打印
            event.accept() # 明确标记事件已被接受和处理
            return True # 返回 True 表示事件已被处理，停止传播

        # 对于其他事件或不是我们关注的 widget 的事件，让它们继续正常处理和传播
        return super().eventFilter(watched, event)

    # --- 控制视图元素显示/隐藏的新方法 (Modified with prints) ---
    def show_2d_view_elements(self):
        """确保 2D 视图元素可见 (它们是 splitter 的一部分)。
           隐藏 3D 视图。
           添加了调试打印。
           """
        print("\n--- show_2d_view_elements Called ---")
        print(f"VTK widget 当前可见性: {self.vtk_widget.isVisible() if self.vtk_widget else 'N/A'}")
        print(f"VTK widget 的父级 Widget: {self.vtk_widget.parent() if self.vtk_widget else 'N/A'}")
        print(f"VTK widget 的父级是否是 QSplitter: {isinstance(self.vtk_widget.parent(), QSplitter) if self.vtk_widget and self.vtk_widget.parent() else 'N/A'}")


        print("确保 2D 视图元素可见。")
        # 在 splitter 布局中，2D 容器通常是始终可见的
        if self.axial_view_container is not None:
            self.axial_view_container.show()
        if self.axial_slice_info_label is not None:
            self.axial_slice_info_label.show()


        if self.vtk_widget is not None:
             print("正在隐藏 VTK widget...")
             self.vtk_widget.hide() # Hide 3D when showing 2D explicitly
             # --- 添加打印 ---
             print(f"在 show_2d_view_elements 中调用 hide() 后: VTK widget 可见性: {self.vtk_widget.isVisible()}")
             # --- 打印结束 ---
        self.axial_view_label.update() # 强制重新绘制 2D 视图

        print("--- show_2d_view_elements Finished ---")


    # --- 控制视图元素显示/隐藏的新方法 (Modified with prints) ---
    def show_3d_view_elements(self):
        """
        确保 3D 视图元素可见 (VTK 渲染窗口)。
        如果 VTK 可用且已经加载了模型，这个方法会使 VTK 窗口可见。
        添加了调试打印。
        """
        print("\n--- show_3d_view_elements Called ---")
        if self.vtk_widget is not None:
             print(f"VTK widget 当前可见性: {self.vtk_widget.isVisible()}")
             print(f"VTK actor 当前可见性: {self.vtk_actor.GetVisibility() if self.vtk_actor else 'N/A'}")
             print(f"VTK widget 的父级 Widget: {self.vtk_widget.parent()}")
             print(f"VTK widget 的父级是否是 QSplitter: {isinstance(self.vtk_widget.parent(), QSplitter) if self.vtk_widget.parent() else 'N/A'}")


             if self.vtk_actor is not None and self.vtk_actor.GetVisibility():
                 print("VTK actor 可见，正在显示 VTK widget...")
                 self.vtk_widget.show()
                 # --- 添加打印 ---
                 print(f"在 show_3d_view_elements 中调用 show() 后: VTK widget 可见性: {self.vtk_widget.isVisible()}")
                 # --- 打印结束 ---

                 # 确保 2D 视图也保持可见 (在 splitter 布局中)
                 if self.axial_view_container is not None:
                      self.axial_view_container.show()
                 if self.axial_slice_info_label is not None:
                      self.axial_slice_info_label.show()


                 self.vtk_widget.GetRenderWindow().Render() # 强制渲染
             else:
                 print("没有可显示的 3D 模型。")
                 # --- 添加打印 ---
                 print(f"没有模型时，VTK widget 可见性: {self.vtk_widget.isVisible()}")
                 # --- 打印结束 ---
                 # 如果没有模型，确保隐藏 3D 视图
                 self.vtk_widget.hide()
                 # 确保 2D 视图可见
                 if self.axial_view_container is not None:
                     self.axial_view_container.show()
                 if self.axial_slice_info_label is not None:
                     self.axial_slice_info_label.show()


             print("--- show_3d_view_elements Finished ---")

        else:
             print("3D view is not available (VTK not installed).")
             print("--- show_3d_view_elements Finished ---")

    # 步骤 3：添加将 skimage 网格数据转换为 VTK PolyData 的方法

    def convert_skimage_mesh_to_vtk(self, vertices, faces):
        """
        将 skimage 的 Marching Cubes 输出 (顶点和面) 转换为 VTK vtkPolyData。

        Args:
            vertices: NumPy array, 形状 (N, 3)，顶点的 (x, y, z) 坐标。
            faces: NumPy array, 形状 (M, 3)，每个三角形面的顶点索引。

        Returns:
            vtkPolyData 对象，如果输入无效则返回 None。
        """
        if vertices is None or faces is None or vertices.shape[0] == 0 or faces.shape[0] == 0:
            print("Warning: No mesh data to convert to VTK.")
            return None

        try:
            # 创建 vtkPoints 对象并添加顶点
            vtk_points = vtk.vtkPoints()
            # 重要的是要注意 SimpleITK 和 NumPy 使用 Z, Y, X 顺序，
            # 而 skimage.marching_cubes 返回的是 (z, y, x) 索引对应的 (y, x, z) 坐标 (columns, rows, slices)
            # VTK 通常使用 (x, y, z) 笛卡尔坐标。
            # 这里的 vertices 是 (y, x, z) 顺序，我们需要将其转换为 (x, y, z)
            # 物理空间坐标转换（如果需要）：
            # physical_coords = origin + vertex_coords * spacing @ direction.T
            # 对于简单的可视化，先使用体素坐标 (x, y, z) 对应 skimage 的 (columns, rows, slices)
            # 将 skimage 的 (y, x, z) -> VTK 的 (x, y, z) 转换为 (vertices[:, 1], vertices[:, 0], vertices[:, 2])
            for i in range(vertices.shape[0]):
                vtk_points.InsertNextPoint(vertices[i, 1], vertices[i, 0], vertices[i, 2])


            # 创建 vtkCellArray 对象并添加三角形面
            vtk_triangles = vtk.vtkCellArray()
            for i in range(faces.shape[0]):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, faces[i, 0])
                triangle.GetPointIds().SetId(1, faces[i, 1])
                triangle.GetPointIds().SetId(2, faces[i, 2])
                vtk_triangles.InsertNextCell(triangle)

            # 创建 vtkPolyData 对象
            vtk_polydata = vtk.vtkPolyData()
            vtk_polydata.SetPoints(vtk_points)
            vtk_polydata.SetPolys(vtk_triangles)

            # 可选：生成法线以便正确的光照渲染
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(vtk_polydata)
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.Update()
            vtk_polydata = normals.GetOutput()


            print("Successfully converted skimage mesh to VTK PolyData.")
            return vtk_polydata

        except Exception as e:
            print(f"Error converting skimage mesh to VTK PolyData: {e}")
            return None

    # --- Helper method to update item control states ---
    # --- Helper method to update item control states (Modified) ---
    def _update_item_control_states(self, image_data, controls_container_widget):
        """
        根据可见性和 mask 状态启用/禁用图像列表项的控制项 (滑块) 及其容器。

        Args:
            image_data: 图像数据字典。
            controls_container_widget: 包含滑块等控制项的 QWidget 容器。
        """
        is_visible = image_data['is_visible']
        is_mask = image_data.get('is_mask', False)

        # 直接启用/禁用整个控制项容器 Widget
        # 禁用容器会自动禁用其内的所有子控件 (包括滑块)
        controls_container_widget.setEnabled(is_visible)

        # 如果需要更细粒度的控制，可以在这里根据 is_mask 状态
        # 再次设置单个滑块的可用性，但通常禁用容器已经足够。
        # 这里不再调用 .parent().setEnabled()，避免潜在问题。

        # image_data['slider_opacity'].setEnabled(is_visible) # 可选，容器已禁用
        # if image_data.get('slider_ww') is not None:
        #      image_data['slider_ww'].setEnabled(is_visible and not is_mask) # 可选，容器已禁用
        # if image_data.get('slider_wl') is not None:
        #      image_data['slider_wl'].setEnabled(is_visible and not is_mask) # 可选，容器已禁用


    def import_file(self, checked=False):
        """
        打开文件对话框，选择并导入医学图像文件 (.nrrd, .mha, .png, .jpg, .bmp)。
        尝试识别图像是否为 mask，并为图像在列表中创建控制项，包括窗宽窗位滑块。
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filters = self.IMAGE_FILE_FILTERS
        # 允许选择多个文件
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择要导入的医学图像",
            "",
            filters,
            options=options
        )

        if file_paths:
            print(f"Selected {len(file_paths)} files for import.")
            for file_path in file_paths:
                print(f"Attempting to load: {os.path.basename(file_path)}")
                try:
                    # 使用 SimpleITK 读取图像
                    image = sitk.ReadImage(file_path)

                    is_mask = False
                    pixel_id = image.GetPixelID()
                    # 使用已经修改过的 is_integer_pixel_type 辅助函数
                    if is_integer_pixel_type(pixel_id):
                         try:
                             # 将 SimpleITK 图像转换为 NumPy 数组进行分析
                             image_array_for_check = sitk.GetArrayFromImage(image)
                             # 检查唯一值数量，作为判断是否为 mask 的启发式方法
                             unique_values = np.unique(image_array_for_check)

                             # 如果唯一值数量较少 (例如 <= 256)，或者符合二值 mask 特征 (0 和一个其他值)
                             if len(unique_values) <= 256:
                                  is_mask = True
                                  # 修改这里的打印语句，移除对 GetPixelIDTypeAsString 的调用
                                  print(f"Identified {os.path.basename(file_path)} as a potential mask (Unique values: {len(unique_values)}).")
                             elif len(unique_values) <= 2 and unique_values[0] == 0:
                                 is_mask = True
                                 # 修改这里的打印语句，移除对 GetPixelIDTypeAsString 的调用
                                 print(f"Identified {os.path.basename(file_path)} as a potential binary mask.")
                             else:
                                 # 虽然是整数类型，但唯一值数量太多，不像 typical mask
                                 print(f"Did not identify {os.path.basename(file_path)} as a mask (Too many unique values or not binary).")


                         except Exception as unique_e:
                             # 捕获在获取 numpy 数组或检查唯一值时可能发生的错误
                             print(f"Error checking unique values for mask detection: {unique_e}")
                             # 如果唯一值检查失败，则假设不是 mask
                             is_mask = False


                    # 如果 is_integer_pixel_type 返回 False，或者唯一值检查后 is_mask 仍为 False
                    if not is_mask:
                         # 修改这里的打印语句，移除对 GetPixelIDTypeAsString 的调用
                         print(f"Did not identify {os.path.basename(file_path)} as a mask.")


                    # --- 计算强度统计信息用于窗宽/窗位 (对所有图像) ---
                    # 这个代码块现在对所有图像执行，无论是否为 mask
                    min_val, max_val, mean_val = (0.0, 255.0, 128.0) # 默认值，使用浮点数
                    try:
                        stats = sitk.StatisticsImageFilter()
                        stats.Execute(image)
                        min_val = float(stats.GetMinimum()) # 转换为浮点数
                        max_val = float(stats.GetMaximum()) # 转换为浮点数
                        mean_val = float(stats.GetMean())   # 转换为浮点数
                    except Exception as stats_e:
                        print(f"Error getting intensity statistics for {os.path.basename(file_path)}: {stats_e}")
                        # 如果获取统计信息失败，回退到基于像素类型的估算或默认值
                        try:
                            pixel_id = image.GetPixelID()
                            if hasattr(sitk, 'GetPixelIDValueMaximum'):
                                # 尝试获取像素类型的最大最小值
                                max_val_sitk = sitk.GetPixelIDValueMaximum(pixel_id)
                                min_val_sitk = sitk.GetPixelIDValueMinimum(pixel_id)
                                # 确保处理非数字的情况 (如 bool 类型)
                                if np.isscalar(min_val_sitk) and np.isscalar(max_val_sitk):
                                    min_val = float(min_val_sitk)
                                    max_val = float(max_val_sitk)
                                    mean_val = (min_val + max_val) / 2.0
                                else:
                                     # 如果获取的不是标量，回退到硬编码默认值
                                     min_val, max_val, mean_val = (0.0, 255.0, 128.0)
                            else:
                                 # Fallback to hardcoded defaults if getting pixel range also fails
                                 min_val, max_val, mean_val = (0.0, 255.0, 128.0)
                        except Exception as pixel_range_e:
                            print(f"Error getting pixel range for {os.path.basename(file_path)}: {pixel_range_e}")
                            min_val, max_val, mean_val = (0.0, 255.0, 128.0)


                    initial_level = mean_val
                    initial_width = max_val - min_val
                    if initial_width <= 0:
                         initial_width = 1.0 # 避免除以零或非正数窗宽


                    # 获取图像基本信息
                    file_name = os.path.basename(file_path)
                    img_size = image.GetSize()
                    img_dimension = image.GetDimension()
                    # SimpleITK Size 是 (width, height, depth, ...), GetDimension 是实际维度
                    # z_size 对应深度 (SimpleITK Size 的第三个元素，如果存在)
                    z_size = img_size[2] if img_dimension > 2 else 1


                    # 创建显示在列表中的信息文本
                    info_text = (f"名称: {file_name}\n"
                                 f"尺寸: {img_size[0]}x{img_size[1]}{'x' + str(z_size) if img_dimension > 2 else ''}\n" # 根据维度显示 Z 尺寸
                                 f"维度: {img_dimension}D")
                    if is_mask:
                         info_text += "\n类型: Mask" # 仍然显示是否为 Mask


                    info_label = QLabel(info_text)
                    info_label.setWordWrap(True) # 允许文本换行

                    visible_checkbox = QCheckBox("显示")
                    visible_checkbox.setChecked(True) # 默认显示

                    # 创建包含控制项的容器
                    item_controls_container = QWidget()
                    controls_v_layout = QVBoxLayout(item_controls_container)
                    controls_v_layout.setContentsMargins(0, 0, 0, 0) # 移除布局的边距

                    # 透明度滑块 (对所有图像都适用)
                    opacity_h_layout = QHBoxLayout()
                    opacity_label = QLabel("透明度:")
                    opacity_slider = QSlider(Qt.Horizontal)
                    opacity_slider.setRange(0, 100) # 0% 到 100%
                    opacity_slider.setValue(100) # 默认 100% 不透明
                    opacity_slider.setToolTip("调节图像透明度")
                    opacity_h_layout.addWidget(opacity_label)
                    opacity_h_layout.addWidget(opacity_slider)
                    controls_v_layout.addLayout(opacity_h_layout)

                    # --- 创建窗宽/窗位滑块 (对所有图像) ---
                    # 移除 if not is_mask: 的判断
                    ww_slider = None # 保持变量声明
                    wl_slider = None # 保持变量声明

                    ww_h_layout = QHBoxLayout() # <-- 这部分现在是无条件的
                    ww_label = QLabel("窗宽:")
                    ww_slider = QSlider(Qt.Horizontal)
                    # 计算滑块初始值和设置范围 (使用上面计算的 min_val, max_val, initial_width)
                    max_width_range = max_val - min_val
                    if max_width_range <= 0: max_width_range = 1000.0 # 使用浮点数
                    mapped_initial_width = max(1.0, initial_width) # 确保窗宽至少为 1.0
                    ww_slider_initial_value = int(((mapped_initial_width - 1.0) / max_width_range) * 100) if max_width_range > 0 else 50
                    ww_slider_initial_value = max(0, min(100, ww_slider_initial_value)) # 夹紧到 [0, 100]
                    ww_slider.setRange(0, 100)
                    ww_slider.setValue(ww_slider_initial_value)
                    ww_slider.setToolTip(f"调节窗宽 ({initial_width:.2f})")
                    ww_h_layout.addWidget(ww_label)
                    ww_h_layout.addWidget(ww_slider)
                    controls_v_layout.addLayout(ww_h_layout) # <-- 无条件添加到布局


                    wl_h_layout = QHBoxLayout() # <-- 这部分现在是无条件的
                    wl_label = QLabel("窗位:")
                    wl_slider = QSlider(Qt.Horizontal)
                    # 计算滑块初始值和设置范围 (使用上面计算的 min_val, max_val, initial_level)
                    range_for_slider = max_val - min_val
                    if range_for_slider <= 0: range_for_slider = 100.0 # 使用浮点数
                    wl_slider_initial_value = int(((initial_level - min_val) / range_for_slider) * 100) if range_for_slider > 0 else 50
                    wl_slider_initial_value = max(0, min(100, wl_slider_initial_value)) # 夹紧到 [0, 100]
                    wl_slider.setRange(0, 100)
                    wl_slider.setValue(wl_slider_initial_value)
                    wl_slider.setToolTip(f"调节窗位 ({initial_level:.2f})")
                    wl_h_layout.addWidget(wl_label)
                    wl_h_layout.addWidget(wl_slider)
                    controls_v_layout.addLayout(wl_h_layout) # <-- 无条件添加到布局


                    # 构建图像数据字典，存储所有相关信息
                    image_data = {
                        'path': file_path,
                        'sitk_image': image,
                        'info_label': info_label,
                        'checkbox': visible_checkbox,
                        'slider_opacity': opacity_slider,
                        'slider_ww': ww_slider, # <-- 现在总是存储创建的滑块对象
                        'slider_wl': wl_slider, # <-- 现在总是存储创建的滑块对象
                        'is_visible': True,
                        'opacity': 1.0,
                        'current_slice_index': z_size // 2 if img_dimension > 2 else 0, # 默认显示中间切片或第一切片
                        'window_level': initial_level, # <-- 存储所有图像的初始窗位
                        'window_width': initial_width, # <-- 存储所有图像的初始窗宽
                        'min_intensity': min_val, # <-- 存储最小值 (用于后续窗宽窗位计算)
                        'max_intensity': max_val, # <-- 存储最大值 (用于后续窗宽窗位计算)
                        'is_mask': is_mask, # 仍然保留 Mask 标记，用于其他逻辑 (如 3D 提取)
                        'contours': {}, # 存储 2D 轮廓 {slice_index: [contour1, contour2, ...]}
                        'edited_contours': {}, # 可选：存储修改后的轮廓
                        'mesh_data': None, # 添加字段来存储 3D 网格数据 (VTKPolyData 或其他格式)
                        'controls_container_widget': item_controls_container # <-- 添加控制项容器的引用
                    }
                    # 将新的图像数据字典添加到已加载图像列表
                    self.loaded_images.append(image_data)

                    # 创建列表项的主布局和 widget
                    item_main_h_layout = QHBoxLayout()
                    item_main_h_layout.addWidget(info_label, 1) # 信息标签占据更多空间
                    item_main_h_layout.addWidget(visible_checkbox)
                    item_main_h_layout.addWidget(item_controls_container) # 添加控制项容器

                    item_widget = QWidget()
                    item_widget.setLayout(item_main_h_layout)
                    item_widget.setStyleSheet("border: 1px solid lightgray; padding: 5px;")

                    # 设置自定义上下文菜单策略，用于右键菜单
                    item_widget.setContextMenuPolicy(Qt.CustomContextMenu)
                    # 连接自定义上下文菜单请求信号到 show_item_context_menu 方法
                    # 使用 lambda 传递当前的 image_data
                    item_widget.customContextMenuRequested.connect(
                        lambda pos, data=image_data: self.show_item_context_menu(pos, data)
                    )

                    # 将新的图像列表项 widget 插入到列表布局的末尾 (在 stretch 之前)
                    self.image_list_layout.insertWidget(self.image_list_layout.count() - 1, item_widget)

                    # 连接控件的信号到相应的槽函数
                    # 使用 lambda 传递当前的 image_data 字典，确保信号触发时处理的是正确的图像
                    visible_checkbox.stateChanged.connect(lambda state, data=image_data: self.toggle_image_visibility(state, data))
                    opacity_slider.valueChanged.connect(lambda value, data=image_data: self.update_image_opacity(value, data))

                    # 连接 WW/WL 滑块的信号 (现在滑块总是被创建)
                    # if ww_slider is not None: # <-- 这个检查现在 technically redundant but harmless
                    ww_slider.valueChanged.connect(lambda value, data=image_data: self.update_window_width(value, data))
                    # if wl_slider is not None: # <-- 这个检查现在 technically redundant but harmless
                    wl_slider.valueChanged.connect(lambda value, data=image_data: self.update_window_level(value, data))


                    # Call helper to set initial slider enable/disable state, Pass the container widget
                    # _update_item_control_states 将只基于可见性启用/禁用控制项容器
                    self._update_item_control_states(image_data, item_controls_container)

                    # 删除自动提取 2D 轮廓的代码 (如果之前保留了)
                    # if is_mask and cv2 is not None:
                    #      self._extract_2d_contours_for_image(image_data) # 已在之前的步骤中删除或注释掉


                except Exception as e:
                    # 捕获读取文件或处理过程中发生的任何其他错误
                    print(f"读取文件 {file_path} 时发生错误: {e}")
                    # 在列表中显示加载失败信息
                    error_label = QLabel(f"加载失败: {os.path.basename(file_path)}\n错误: {e}")
                    error_label.setStyleSheet("color: red;")
                    # 将错误信息标签插入到列表布局的末尾 (在 stretch 之前)
                    self.image_list_layout.insertWidget(self.image_list_layout.count() - 1, error_label)

            # 处理完所有选定的文件后，更新右侧显示区域
            self.update_right_display()

        else:
            # 如果用户取消了文件选择对话框
            print("文件导入被取消。")

    


    # 步骤 4 (可选但推荐): 添加一个内部方法来提取 2D 轮廓
    def _extract_2d_contours_for_image(self, image_data):
        """Internal method to extract 2D contours for a given image_data item."""
        print(f"Extracting 2D contours for {image_data.get('path', 'Stacked 3D Mask')}...")
        if cv2 is None:
            print("Contours extraction requires OpenCV, but it is not available.")
            return

        sitk_image = image_data.get('sitk_image')
        if sitk_image is None or sitk_image.GetDimension() < 2:
            print("Cannot extract contours: SimpleITK image object is missing or not 2D/3D.")
            return

        try:
            image_array = sitk.GetArrayFromImage(sitk_image)
            img_dimension = image_array.ndim
            img_shape = image_array.shape
            z_size = img_shape[0] if img_dimension > 2 else 1

            image_data['contours'] = {}

            for slice_index in range(z_size):
                try:
                    slice_2d_original_data = None
                    if img_dimension >= 3:
                         slice_2d_original_data = image_array[slice_index, :, :]
                    elif img_dimension == 2:
                         slice_2d_original_data = image_array
                         if slice_index > 0: break # For 2D image, only process slice 0

                    if slice_2d_original_data is not None:
                         binary_slice = slice_2d_original_data > 0.5
                         contours_for_slice = []
                         # 只有当切片中有前景像素时才提取轮廓
                         if np.max(binary_slice) > 0:
                             # find_contours 接受 (height, width) 数组
                             # 返回的轮廓是 (points, 2) 数组，点坐标是 (row, column) 即 (y, x)
                             contours_skimage = measure.find_contours(binary_slice, 0.5)
                             # 我们需要将轮廓点转换为 (x, y) 坐标用于绘图，所以交换列
                             contours_for_slice = [np.c_[c[:, 1], c[:, 0]] for c in contours_skimage]


                         image_data['contours'][slice_index] = contours_for_slice

                except Exception as contour_e:
                    print(f"提取切片 {slice_index} 的 2D 边缘时发生错误: {contour_e}")
                    image_data['contours'][slice_index] = []

            print(f"完成 2D 边缘提取：{image_data.get('path', 'Stacked 3D Mask')}")

        except Exception as array_e:
             print(f"Error accessing image array for contour extraction: {array_e}")

    # 步骤 4：修改 extract_3d_model_action 方法

    def extract_3d_model_action(self, image_data):
        print(f"Attempting to extract 3D model for: {os.path.basename(image_data.get('path', 'Unknown'))}")

        sitk_image = image_data.get('sitk_image')
        if not sitk_image or sitk_image.GetDimension() < 3 or not image_data.get('is_mask', False):
            print("Selected image is not a 3D mask.")
            return

        # 假设二值 mask 的等值面在 0.5
        mesh_data = extract_surface_mesh(sitk_image, level=0.5)

        if mesh_data:
            vertices, faces = mesh_data
            print("3D mesh extracted successfully from skimage.")

            # 将 skimage 网格数据转换为 VTK PolyData
            vtk_polydata = self.convert_skimage_mesh_to_vtk(vertices, faces)

            if vtk_polydata:
                # 更新 VTK 渲染管线
                self.update_vtk_display(vtk_polydata)

                # 存储 mesh_data (可选，但保留与之前逻辑一致)
                image_data['mesh_data'] = mesh_data
                print(f"Mesh data stored in image_data for {os.path.basename(image_data.get('path', 'Unknown'))}.")
            else:
                 print("Failed to convert mesh data to VTK format.")

        else:
            print("Failed to extract 3D mesh using skimage.")

    # 步骤 5：添加更新 VTK 显示的方法
    # --- 更新 VTK 显示的方法 (Modified for Splitter layout and added prints) ---
    def update_vtk_display(self, vtk_polydata):
        """
        使用新的 VTK PolyData 更新 3D 渲染显示，并使 3D 视图可见。
        添加了调试打印。
        """
        print("\n--- update_vtk_display Called ---")
        # 检查 VTK Widget 是否可用
        if self.vtk_widget is None:
             print("VTK widget not available. Cannot update 3D display.")
             print("--- update_vtk_display Finished ---")
             return

        if vtk_polydata is None:
             print("No VTK PolyData to display, hiding 3D view elements.")
             self.vtk_actor.VisibilityOff() # 隐藏 actor
             self.vtk_widget.hide()       # 隐藏 3D widget
             # 确保 2D 视图是可见的 (在 splitter 布局中)
             if self.axial_view_container is not None:
                 self.axial_view_container.show()
             if self.axial_slice_info_label is not None:
                  self.axial_slice_info_label.show()
             print("--- update_vtk_display Finished ---")
             return

        try:
            # 更新 mapper 的输入数据
            self.vtk_mapper.SetInputData(vtk_polydata)

            # 使 actor 可见
            self.vtk_actor.VisibilityOn()

            # 重置相机以适应新的模型
            self.vtk_renderer.ResetCamera()

            # 使 3D widget 可见 (它在 splitter 中)
            print("Calling self.vtk_widget.show()...")
            self.vtk_widget.show()
            # --- 添加打印 ---
            print(f"在 update_vtk_display 中调用 show() 后: VTK widget 可见性: {self.vtk_widget.isVisible()}")
            print(f"VTK widget 的父级 Widget: {self.vtk_widget.parent()}")
            print(f"VTK widget 的父级是否是 QSplitter: {isinstance(self.vtk_widget.parent(), QSplitter) if self.vtk_widget.parent() else 'N/A'}") # 增加对 parent() 返回 None 的检查
            # --- 打印结束 ---

            # 确保 2D 视图也保持可见 (在 splitter 布局中它们并排)
            if self.axial_view_container is not None:
                self.axial_view_container.show()
            if self.axial_slice_info_label is not None:
                 self.axial_slice_info_label.show()


            # 强制 VTK 窗口渲染
            self.vtk_widget.GetRenderWindow().Render()


            print("VTK display updated with new mesh.")
            print("--- update_vtk_display Finished ---")


        except Exception as e:
            print(f"Error updating VTK display: {e}")
            print("--- update_vtk_display Finished ---")       

    # 步骤 6：处理窗口关闭时的 VTK 清理

    def closeEvent(self, event):
        """
        在窗口关闭时执行 VTK 渲染窗口的清理。
        """
        print("Closing MainWindow, cleaning up VTK...")
        if self.vtk_widget:
            self.vtk_widget.close() # 这会调用 vtkRenderWindowInteractor 的 UnRegister 方法
        print("VTK cleanup attempted.")
        super().closeEvent(event)

    def extract_edges_action(self, checked=False):
        print("用户点击了 '提取边缘' 按钮")

        if cv2 is None:
            print("边缘填充功能需要 OpenCV 库，但它未安装。请运行 'pip install opencv-python' 进行安装。")

        images_to_process = []
        for image_data in self.loaded_images:
            if image_data.get('is_visible', False):
                images_to_process.append(image_data)

        if not images_to_process:
            print("没有可见的图像需要提取边缘。")
            return

        print(f"正在为 {len(images_to_process)} 个可见图像提取边缘...")

        for image_data in images_to_process:
            sitk_image = image_data['sitk_image']
            image_array = sitk.GetArrayFromImage(sitk_image)
            img_dimension = image_array.ndim
            img_shape = image_array.shape
            z_size = img_shape[0] if img_dimension > 2 else 1

            image_data['contours'] = {}
            print(f"提取 {os.path.basename(image_data.get('path', 'Unknown'))} 的边缘 ({z_size} 切片)...")

            for slice_index in range(z_size):
                try:
                    slice_2d_original_data = None
                    if img_dimension >= 3:
                         slice_2d_original_data = image_array[slice_index, :, :]
                    elif img_dimension == 2:
                         slice_2d_original_data = image_array
                         if slice_index > 0: break

                    if slice_2d_original_data is not None:
                         binary_slice = slice_2d_original_data > 0.5
                         contours_for_slice = []
                         if np.max(binary_slice) > 0:
                             contours_skimage = measure.find_contours(binary_slice, 0.5)
                             contours_for_slice = [np.c_[c[:, 1], c[:, 0]] for c in contours_skimage]

                         image_data['contours'][slice_index] = contours_for_slice

                except Exception as contour_e:
                    print(f"提取 {os.path.basename(image_data.get('path', 'Unknown'))} 切片 {slice_index} 的边缘时发生错误: {contour_e}")
                    image_data['contours'][slice_index] = []

            print(f"完成边缘提取：{os.path.basename(image_data.get('path', 'Unknown'))}")

        self.update_right_display()
        print("边缘提取完成，更新显示。")


    def show_item_context_menu(self, pos, image_data):
        print("\n--- show_item_context_menu called ---")
        # 打印当前右键点击的图像名称
        print(f"Image data received for: {os.path.basename(image_data.get('path', 'Unknown'))}")

        menu = QMenu(self)
        save_action = QAction("保存图像", self)
        menu.addAction(save_action)
        print("Added '保存图像' action.")

        # --- 检查添加 3D 提取动作的条件 ---
        # 检查是否存在 sitk_image 对象
        has_sitk_image = image_data.get('sitk_image') is not None
        # 检查图像维度是否大于 2
        is_3d_or_more = False
        if has_sitk_image:
            try:
                is_3d_or_more = image_data['sitk_image'].GetDimension() > 2
            except Exception as e:
                print(f"Error checking image dimension: {e}")
                is_3d_or_more = False # 如果检查维度出错，认为不是 3D
        # 检查图像是否被标记为 mask
        is_mask = image_data.get('is_mask', False)

        # 打印出每个条件的结果
        print(f"Condition checks: has_sitk_image={has_sitk_image}, is_3d_or_more={is_3d_or_more}, is_mask={is_mask}")

        # 添加提取 3D 模型动作
        # 只有当图像是 3D 且被识别为 mask 时才显示此选项
        if has_sitk_image and is_3d_or_more and is_mask:
            print("All conditions met. Adding '提取 3D 模型' action.")
            extract_3d_action = QAction("提取 3D 模型", self)
            menu.addAction(extract_3d_action)
            extract_3d_action.triggered.connect(lambda: self.extract_3d_model_action(image_data))
        else:
            print("Conditions not met. '提取 3D 模型' action not added.")


        # 将保存动作连接到槽函数
        save_action.triggered.connect(lambda: self.save_specific_image(image_data))

        print("Executing context menu...")
        # 显示右键菜单
        menu.exec_(self.sender().mapToGlobal(pos))
        print("--- show_item_context_menu finished ---")


    def save_specific_image(self, image_data):
        print(f"尝试保存图像: {os.path.basename(image_data.get('path', 'Unknown'))}")
        if image_data and image_data.get('sitk_image'):
            sitk_image = image_data['sitk_image']
            original_path = image_data['path']
            default_file_name = os.path.basename(original_path)
            file_name_base, file_extension = os.path.splitext(default_file_name)

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filters = self.IMAGE_FILE_FILTERS
            default_filter = f"{file_extension.upper().replace('.', '')} Files (*{file_extension})" if file_extension and file_extension.lower() in ['.mha', '.nrrd'] else "MHA Files (*.mha)"


            save_path, selected_filter = QFileDialog.getSaveFileName(
                self, f"保存图像: {os.path.basename(original_path)}", default_file_name, filters, default_filter, options=options)

            if save_path:
                if '.' not in os.path.basename(save_path):
                    if "(*.mha)" in selected_filter: save_path += ".mha"
                    elif "(*.nrrd)" in selected_filter: save_path += ".nrrd"
                    elif os.path.splitext(save_path)[1] == "":
                         save_path += ".mha"

                try:
                    sitk.WriteImage(sitk_image, save_path)
                    print(f"图像保存成功到: {save_path}")
                except Exception as e:
                    print(f"保存图像时发生错误: {e}")
            else:
                print("取消了保存。")

        else:
            print("图像数据无效，无法保存")


    def toggle_image_visibility(self, state, image_data):
        """
        根据复选框状态切换图像的可见性，并更新控制项状态。
        """
        image_data['is_visible'] = (state == Qt.Checked)
        print(f"Toggling visibility for {os.path.basename(image_data.get('path', 'Unknown'))}: {image_data['is_visible']}")


        # 从 image_data 字典中获取控制项容器 Widget 的引用
        controls_container_widget = image_data.get('controls_container_widget')

        # 使用辅助方法更新控制项状态，并传递获取到的容器 Widget
        if controls_container_widget is not None:
            self._update_item_control_states(image_data, controls_container_widget)
        else:
             print("Warning: Controls container widget not found in image_data for this item.")
             # 如果没有找到容器，虽然不应该发生，但为了避免崩溃，这里可以打印警告并跳过更新控制项状态


        # 更新右侧主显示区域以反映可见性变化
        # 这会导致 SliceDisplayLabel 重新构建 composited_layers
        self.update_right_display()

    def update_image_opacity(self, value, image_data):
        image_data['opacity'] = value / 100.0
        self.update_right_display()

    def update_window_width(self, value, image_data):
        min_intensity = image_data.get('min_intensity', 0)
        max_intensity = image_data.get('max_intensity', 255)

        max_width_range = max_intensity - min_intensity
        if max_width_range <= 0:
             max_width_range = 1000

        mapped_width = 1.0 + (max_width_range) * (value / 100.0)
        mapped_width = max(1.0, mapped_width)

        image_data['window_width'] = mapped_width
        image_data['slider_ww'].setToolTip(f"调节窗宽 ({mapped_width:.2f})")
        self.update_right_display()

    def update_window_level(self, value, image_data):
        min_intensity = image_data.get('min_intensity', 0)
        max_intensity = image_data.get('max_intensity', 255)

        range_for_slider = max_intensity - min_intensity
        if range_for_slider <= 0: range_for_slider = 100

        mapped_level = min_intensity + (range_for_slider) * (value / 100.0)

        image_data['window_level'] = mapped_level
        image_data['slider_wl'].setToolTip(f"调节窗位 ({mapped_level:.2f})")
        self.update_right_display()


    def update_right_display(self, checked=False):
        visible_images_data_list = []
        for image_data in self.loaded_images:
            if image_data['is_visible']:
                visible_images_data_list.append(image_data)

        self.axial_view_label.set_composition_data(visible_images_data_list)

    def update_info_labels(self, visible_images_data_list):
        info_source_image = None
        for img_data in visible_images_data_list:
             if img_data.get('sitk_image') and img_data['sitk_image'].GetDimension() > 2:
                  info_source_image = img_data
                  break
        if info_source_image is None:
            for img_data in visible_images_data_list:
                 if img_data.get('sitk_image'):
                     info_source_image = img_data
                     break

        if info_source_image:
             current_slice = info_source_image.get('current_slice_index', 0)
             total_slices = info_source_image['sitk_image'].GetSize()[2] if info_source_image['sitk_image'].GetDimension() > 2 else 1
             self.axial_slice_info_label.setText(f"切片: {current_slice + 1} / {total_slices}")

        else:
             self.axial_slice_info_label.setText("切片: -- / --")

    # Removed empty keyPressEvent method

    # --- MainWindow's wheelEvent ---
    def wheelEvent(self, event):
        delta_y = event.angleDelta().y()

        # Check if there's any loaded AND visible 3D image to slice through
        has_visible_3d = any(img_data.get('is_visible', False) and img_data.get('sitk_image') and img_data['sitk_image'].GetDimension() > 2 for img_data in self.loaded_images)

        if delta_y != 0 and has_visible_3d:
            event.accept() # Accept the event as handled for slicing

            slice_direction = 1 if delta_y > 0 else -1

            # Update the current slice index for all loaded AND visible 3D images
            for image_data in self.loaded_images:
                if image_data.get('is_visible', False) and image_data.get('sitk_image') and image_data['sitk_image'].GetDimension() > 2:
                    current_slice = image_data.get('current_slice_index', 0)
                    z_size = image_data['sitk_image'].GetSize()[2]
                    new_slice = max(0, min(current_slice + slice_direction, z_size - 1))
                    image_data['current_slice_index'] = new_slice

            # --- Call public method on SliceDisplayLabel to clear state ---
            # Instead of directly accessing private attributes
            self.axial_view_label.clear_selection_and_dragging() # Use the new method


            # Update the display on the right panel based on the new slice indices
            self.update_right_display()

        else:
            # Pass the wheel event up if not handled for slicing (e.g., for scrolling)
            super().wheelEvent(event)


# --- Application Launch Code ---
if __name__ == "__main__":
    print("--- Program Starting ---")
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    print("QApplication created successfully.")

    print("Creating MainWindow instance...")
    try:
        main_window = MainWindow()
        print("MainWindow instance created successfully.")
    except Exception as e:
        print(f"Error during MainWindow creation: {e}")
        sys.exit(1) # Exit if MainWindow creation fails

    print("Calling main_window.show()...")
    main_window.show()
    print("main_window.show() called.")

    print("Starting application event loop (app.exec_())...")
    try:
        exit_code = app.exec_()
        print(f"Application event loop finished with exit code {exit_code}.")
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error during application event loop: {e}")
        sys.exit(1)
