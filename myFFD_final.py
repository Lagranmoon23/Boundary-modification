import sys
import numpy as np
import SimpleITK as sitk
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QDesktopWidget,
                             QHBoxLayout, QVBoxLayout, QMenu, QAction, QFileDialog,
                             QLabel, QSizePolicy, QCheckBox, QScrollArea, QSlider)
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


# --- Helper function for pixel type check ---
def is_integer_pixel_type(pixel_id):
    """ Checks if the SimpleITK pixel ID corresponds to an integer type using string representation. """
    try:
        pixel_type_string = sitk.GetPixelIDTypeAsString(pixel_id)
        return 'int' in pixel_type_string.lower()
    except Exception as e:
        print(f"Error getting pixel type string: {e}")
        return False

# --- Custom Image Display Label (Multi-layer composition, interaction, contour display, dragging) ---
class SliceDisplayLabel(QLabel):
    composition_changed = pyqtSignal(list)

    # Constants for local deformation
    DEFAULT_INFLUENCE_RADIUS_IMAGE = 30.0 # Default influence radius in image pixels
    DEFAULT_FALLOFF_POWER = 2.0         # Default power for the fall-off function (e.g., 1.0 for linear, 2.0 for quadratic)


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

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid green;")
        self.setText("Axial (Z) View")

        self.setMouseTracking(True)
        self.setEnabled(True)

    def set_composition_data(self, visible_images_data_list):
        self._visible_images_data_refs = visible_images_data_list
        # Clear dragging and selection states as the layers might have changed or reordered
        # Also clear them if the previously selected layer is no longer visible.
        if self._selected_point is not None:
             layer_index_in_composited = self._selected_point[0]
             if layer_index_in_composited >= len(self._composited_layers): # Check against OLD composited layers first
                  self._selected_point = None
                  self._is_dragging = False

             # Re-find the layer index in the NEW list if needed to preserve selection?
             # This is complex, skipping for now. Clear selection on any composition change.
             self._selected_point = None
             self._is_dragging = False


        if not self._visible_images_data_refs:
            self._composited_layers = []
            self._scaled_pixmap_size = QSize(0, 0)
            self._pan_offset = QPoint(0, 0)
            self._zoom_factor = 1.0
            self.setPixmap(QPixmap())
            self.setText("Axial (Z) View")

        else:
             if not self._composited_layers:
                  self._pan_offset = QPoint(0, 0)
                  self._zoom_factor = 1.0

        self._update_composition()


    def _update_composition(self):
        self._composited_layers = []
        current_composite_original_size = QSize(0, 0)

        if not self._visible_images_data_refs:
            self._scaled_pixmap_size = QSize(0, 0)
            self.setPixmap(QPixmap())
            self.setText("Axial (Z) View")
            self.update()
            self.composition_changed.emit(self._visible_images_data_refs)
            return

        for img_data_index_in_visible_list, img_data in enumerate(self._visible_images_data_refs):
             sitk_image = img_data.get('sitk_image')
             slice_index = img_data.get('current_slice_index', 0)
             opacity = img_data.get('opacity', 1.0)
             window_level = img_data.get('window_level', None)
             window_width = img_data.get('window_width', None)
             is_mask = img_data.get('is_mask', False)

             if not sitk_image:
                 continue

             image_array = sitk.GetArrayFromImage(sitk_image)
             img_dimension = image_array.ndim

             slice_2d_original_data = None
             contours_for_slice = []

             if img_dimension == 2:
                  slice_2d_original_data = image_array
             elif img_dimension >= 3:
                  z_size = image_array.shape[0]
                  clamped_slice_index = max(0, min(slice_index, z_size - 1))
                  slice_2d_original_data = image_array[clamped_slice_index, :, :]

             if slice_2d_original_data is not None:
                 pixmap = self.get_qpixmap_from_slice_array(
                     slice_2d_original_data,
                     window_level if not is_mask else None,
                     window_width if not is_mask else None
                 )

                 contours_for_slice = img_data.get('contours', {}).get(slice_index, [])


                 if not pixmap.isNull():
                     # Store pixmap, opacity, is_mask, contours, and *index in self._visible_images_data_refs*
                     # This index links back to the original image_data dictionary
                     self._composited_layers.append((pixmap, opacity, is_mask, contours_for_slice, img_data_index_in_visible_list))


                     if current_composite_original_size.isEmpty():
                         current_composite_original_size = pixmap.size()


        self._scaled_pixmap_size = current_composite_original_size

        if self._composited_layers:
             self.setText("")
        else:
             self.setText("No visible images")

        if self._selected_point is not None:
             layer_index_in_composited = self._selected_point[0]
             if layer_index_in_composited >= len(self._composited_layers):
                  self._selected_point = None


        self.update()
        self.composition_changed.emit(self._visible_images_data_refs)

    def get_qpixmap_from_slice_array(self, slice_array, window_level=None, window_width=None):
        if slice_array is None or slice_array.size == 0:
            return QPixmap()
        try:
            processed_slice = slice_array.astype(np.float64)

            if window_level is not None and window_width is not None:
                 if window_width <= 0:
                     level_val = float(window_level)
                     processed_slice = np.full(processed_slice.shape, 255.0 if level_val <= np.mean(processed_slice) else 0.0)
                 else:
                     min_val = float(window_level) - float(window_width) / 2.0
                     max_val = float(window_level) + float(window_width) / 2.0
                     processed_slice = (processed_slice - min_val) / float(window_width)
                     processed_slice = np.clip(processed_slice * 255.0, 0, 255)
            else:
                 img_min = np.min(processed_slice)
                 img_max = np.max(processed_slice)
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
        """ Transforms a point from image coordinates (x, y) to screen coordinates (x, y). """
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
        """ Transforms a point from screen coordinates (x, y) to image coordinates (x, y). """
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


    # --- 修正: mouseMoveEvent 只更新轮廓点位置 ---
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
                         influence_radius = self.DEFAULT_INFLUENCE_RADIUS_IMAGE
                         falloff_power = self.DEFAULT_FALLOFF_POWER

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

                         # Trigger a repaint (This updates the contour drawing ONLY)
                         self.update()
                         event.accept()


        elif self._is_zooming:
            if event.buttons() == Qt.RightButton and event.modifiers() == Qt.ShiftModifier:
                 delta_y = event.pos().y() - self._last_mouse_pos.y()
                 zoom_step = delta_y * 0.005
                 self._zoom_factor += zoom_step
                 self._zoom_factor = max(0.1, min(self._zoom_factor, 20.0))
                 self._last_mouse_pos = event.pos()
                 self.update()
                 event.accept()
        else:
            super().mouseMoveEvent(event)


    # --- 修正: mouseReleaseEvent 添加 Mask 数据更新逻辑 ---
    def mouseReleaseEvent(self, event):
        # Store dragging state before resetting
        was_dragging = self._is_dragging

        if self._is_panning and event.button() == Qt.LeftButton:
            self._is_panning = False
            event.accept()
        elif self._is_zooming and event.button() == Qt.RightButton:
             self._is_zooming = False
             event.accept()
        # --- End Point Dragging ---
        elif was_dragging and event.button() == Qt.LeftButton:
            self._is_dragging = False # End dragging mode
            # Optional: Keep point selected after drag ends? For now, deselect.
            selected_point_info = self._selected_point # Get info before clearing
            self._selected_point = None

            # --- Trigger Mask Data Update After Drag Ends ---
            if selected_point_info is not None:
                 layer_index_in_composited, contour_index, point_index = selected_point_info

                 # Ensure the layer index is still valid
                 if layer_index_in_composited < len(self._composited_layers):
                      img_data_ref = self._visible_images_data_refs[layer_index_in_composited]
                      current_slice_index = img_data_ref.get('current_slice_index', 0)

                      # Call method to update mask pixel data for this slice
                      self._update_mask_slice_pixels(img_data_ref, current_slice_index)

            # --- End Mask Data Update ---

            # Trigger a full display update to show potentially updated pixels AND clear selection highlight
            self._update_composition() # This will regenerate pixmap from updated sitk image
            event.accept()

        else:
            super().mouseReleaseEvent(event)

    # --- 新增: 更新 Mask 像素数据的方法 ---
    def _update_mask_slice_pixels(self, img_data_ref, slice_index_to_update):
        """
        Rasterizes contours for a specific slice onto a blank array,
        creates a new SITK image from the modified array copy, and updates the image_data reference.
        Assumes binary mask (0 or 1).
        """
        print(f"Updating mask pixel data for slice {slice_index_to_update}...")
        if cv2 is None:
            print("Mask pixel update requires OpenCV, but it is not available.")
            return

        sitk_image = img_data_ref.get('sitk_image')
        if sitk_image is None:
            print("Cannot update mask pixels: SimpleITK image object is missing.")
            return

        try:
            # Get a COPY of the image array (crucial if view is read-only)
            image_array_3d = sitk.GetArrayFromImage(sitk_image)
        except Exception as e:
            print(f"Error getting image array copy for mask update: {e}")
            return

        img_dimension = image_array_3d.ndim

        # Determine shape, dtype, and access the correct slice in the array COPY
        if img_dimension >= 3:
            slice_shape = image_array_3d.shape[1:] # (height, width)
            slice_dtype = image_array_3d.dtype
            # Get a mutable view of the slice within the COPY
            current_slice_array_in_copy = image_array_3d[slice_index_to_update, :, :]
        elif img_dimension == 2:
            slice_shape = image_array_3d.shape # (height, width)
            slice_dtype = image_array_3d.dtype
            slice_index_to_update = 0 # Ensure we process the only slice
            # Get a mutable view of the 2D array COPY itself
            current_slice_array_in_copy = image_array_3d


        # Create a temporary blank slice array with the same shape and dtype
        temp_mask_slice = np.zeros(slice_shape, dtype=slice_dtype)

        # Get all contours for the current slice from the cache
        # Use the potentially modified contours from img_data_ref['contours'][slice_index_to_update]
        all_contours_on_slice = img_data_ref.get('contours', {}).get(slice_index_to_update, [])

        # Prepare contours for OpenCV fillPoly
        # cv2.fillPoly expects a list of contours, where each contour is shape (N, 1, 2) and dtype np.int32
        # Filter contours with less than 2 points
        contours_cv2_format = [c.reshape(-1, 1, 2).astype(np.int32) for c in all_contours_on_slice if c.shape[0] > 1]

        if contours_cv2_format:
            try:
                # Assuming binary mask (0/1), fill with 1.
                # Fill color should be the foreground label. If multi-label, this is complex.
                fill_color = 1 # Assuming binary mask foreground is 1

                # Use the temporary blank slice for filling
                cv2.fillPoly(temp_mask_slice, contours_cv2_format, color=fill_color)

                # Copy the filled binary data back to the slice in the 3D array COPY
                # This modifies the image_array_3d COPY
                if temp_mask_slice.shape == current_slice_array_in_copy.shape:
                     current_slice_array_in_copy[:, :] = temp_mask_slice[:, :]
                else:
                     print(f"Shape mismatch during mask update: temp_mask_slice {temp_mask_slice.shape} vs current_slice_array_in_copy {current_slice_array_in_copy.shape}")


            except Exception as fill_e:
                print(f"Error filling contours for mask pixel update (slice {slice_index_to_update}): {fill_e}")
                # Continue even if filling fails, just don't update the pixel data


        # --- Create a NEW SimpleITK image from the modified 3D array COPY ---
        try:
             new_sitk_image = sitk.GetImageFromArray(image_array_3d)
             # Copy metadata (Origin, Spacing, Direction) from the old image to the new one
             new_sitk_image.CopyInformation(sitk_image)

             # Replace the old sitk image object reference with the new one
             img_data_ref['sitk_image'] = new_sitk_image
             print(f"Mask pixel data updated for slice {slice_index_to_update}.")

        except Exception as create_image_e:
             print(f"Error creating new SITK image from modified array: {create_image_e}")
             print("Mask pixel data update failed.")


    def wheelEvent(self, event):
        # Existing wheel event for slicing
        # Clear selection and dragging on slice change
        self._is_dragging = False
        self._selected_point = None
        super().wheelEvent(event)


# --- MainWindow Class ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.loaded_images = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("边界修改")
        self.setGeometry(100, 100, 1500, 800)
        self.setWindowIcon(QIcon("icon/icon1.png"))

        cnter_pos = QDesktopWidget().screenGeometry().center()
        self.move(int(cnter_pos.x() - self.width()/2), int(cnter_pos.y() - self.height()/2))

        main_h_layout = QHBoxLayout(self)

        left_v_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button1 = QPushButton("导入")
        button1.setFixedSize(75, 25)
        button2 = QPushButton("提取边缘")
        button2.setFixedSize(75, 25)
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addStretch(1)
        left_v_layout.addLayout(button_layout)

        self.image_list_scroll_area = QScrollArea()
        self.image_list_scroll_area.setWidgetResizable(True)
        self.image_list_widget = QWidget()
        self.image_list_layout = QVBoxLayout(self.image_list_widget)
        self.image_list_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_list_layout.addStretch(1)
        self.image_list_scroll_area.setWidget(self.image_list_widget)
        left_v_layout.addWidget(self.image_list_scroll_area, 1)

        right_v_layout = QVBoxLayout()
        self.axial_view_label = SliceDisplayLabel()

        self.axial_slice_info_label = QLabel("切片: -- / --")
        self.axial_slice_info_label.setAlignment(Qt.AlignCenter)

        right_v_layout.addWidget(self.axial_view_label, 1)
        right_v_layout.addWidget(self.axial_slice_info_label)

        main_h_layout.addLayout(left_v_layout, 1)
        main_h_layout.addLayout(right_v_layout, 2)

        self.setLayout(main_h_layout)

        self.setMinimumSize(self.sizeHint())

        file_menu = QMenu(self)
        import_action = QAction("导入", self)
        file_menu.addAction(import_action)

        import_action.triggered.connect(self.import_file)

        button1.clicked.connect(self.import_file)

        button2.clicked.connect(self.extract_edges_action)

        self.axial_view_label.composition_changed.connect(self.update_info_labels)


    def import_file(self, checked=False):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择要导入的医学图像",
            "",
            "医学图像文件 (*.nrrd *.mha);;所有文件 (*)",
            options=options
        )

        if file_paths:
            for file_path in file_paths:
                try:
                    image = sitk.ReadImage(file_path)

                    is_mask = False
                    pixel_id = image.GetPixelID()
                    if is_integer_pixel_type(pixel_id):
                         try:
                             image_array_for_check = sitk.GetArrayFromImage(image)
                             unique_values = np.unique(image_array_for_check)
                             if len(unique_values) <= 256:
                                  is_mask = True
                                  print(f"Identified {os.path.basename(file_path)} as a potential mask (Type: {sitk.GetPixelIDTypeAsString(pixel_id)}, Unique values: {len(unique_values)}).")
                             elif len(unique_values) <= 2 and unique_values[0] == 0:
                                 is_mask = True
                                 print(f"Identified {os.path.basename(file_path)} as a potential binary mask (Type: {sitk.GetPixelIDTypeAsString(pixel_id)}).")

                         except Exception as unique_e:
                             print(f"Error checking unique values for mask detection: {unique_e}")

                    if not is_mask:
                         try:
                             pixel_type_string = sitk.GetPixelIDTypeAsString(pixel_id)
                             print(f"Did not identify {os.path.basename(file_path)} as a mask (Type: {pixel_type_string}).")
                         except:
                             print(f"Did not identify {os.path.basename(file_path)} as a mask (Could not get pixel type string).")


                    stats = sitk.StatisticsImageFilter()
                    stats.Execute(image)
                    min_val = stats.GetMinimum()
                    max_val = stats.GetMaximum()
                    mean_val = stats.GetMean()

                    initial_level = mean_val
                    initial_width = max_val - min_val
                    if initial_width == 0:
                         initial_width = 1.0

                    file_name = os.path.basename(file_path)
                    img_size = image.GetSize()
                    img_dimension = image.GetDimension()
                    z_size = img_size[2] if img_dimension > 2 else 1

                    info_text = (f"名称: {file_name}\n"
                                 f"尺寸: {img_size[0]}x{img_size[1]}x{z_size}\n"
                                 f"维度: {img_dimension}D")
                    if is_mask:
                         info_text += "\n类型: Mask"

                    info_label = QLabel(info_text)
                    info_label.setWordWrap(True)

                    visible_checkbox = QCheckBox("显示")
                    visible_checkbox.setChecked(True)

                    item_controls_container = QWidget()
                    controls_v_layout = QVBoxLayout(item_controls_container)
                    controls_v_layout.setContentsMargins(0, 0, 0, 0)

                    opacity_h_layout = QHBoxLayout()
                    opacity_label = QLabel("透明度:")
                    opacity_slider = QSlider(Qt.Horizontal)
                    opacity_slider.setRange(0, 100)
                    opacity_slider.setValue(100)
                    opacity_slider.setToolTip("调节图像透明度")
                    opacity_h_layout.addWidget(opacity_label)
                    opacity_h_layout.addWidget(opacity_slider)
                    controls_v_layout.addLayout(opacity_h_layout)

                    ww_h_layout = QHBoxLayout()
                    ww_label = QLabel("窗宽:")
                    ww_slider = QSlider(Qt.Horizontal)
                    max_width_range = max_val - min_val
                    if max_width_range <= 0:
                         max_width_range = 1000
                    ww_slider_initial_value = int(((initial_width - 1.0) / max_width_range) * 100) if max_width_range > 0 else 50
                    ww_slider_initial_value = max(0, min(100, ww_slider_initial_value))
                    ww_slider.setRange(0, 100)
                    ww_slider.setValue(ww_slider_initial_value)
                    ww_slider.setToolTip(f"调节窗宽 ({initial_width:.2f})")
                    if is_mask:
                         ww_slider.setEnabled(False)
                         ww_slider.setToolTip("Mask图像无需窗宽调节")
                    ww_h_layout.addWidget(ww_label)
                    ww_h_layout.addWidget(ww_slider)
                    controls_v_layout.addLayout(ww_h_layout)

                    wl_h_layout = QHBoxLayout()
                    wl_label = QLabel("窗位:")
                    wl_slider = QSlider(Qt.Horizontal)
                    range_for_slider = max_val - min_val
                    if range_for_slider <= 0: range_for_slider = 100
                    wl_slider_initial_value = int(((initial_level - min_val) / range_for_slider) * 100) if range_for_slider > 0 else 50
                    wl_slider_initial_value = max(0, min(100, wl_slider_initial_value))
                    wl_slider.setRange(0, 100)
                    wl_slider.setValue(wl_slider_initial_value)
                    wl_slider.setToolTip(f"调节窗位 ({initial_level:.2f})")
                    if is_mask:
                         wl_slider.setEnabled(False)
                         wl_slider.setToolTip("Mask图像无需窗位调节")
                    wl_h_layout.addWidget(wl_label)
                    wl_h_layout.addWidget(wl_slider)
                    controls_v_layout.addLayout(wl_h_layout)

                    image_data = {
                        'path': file_path,
                        'sitk_image': image,
                        'info_label': info_label,
                        'checkbox': visible_checkbox,
                        'slider_opacity': opacity_slider,
                        'slider_ww': ww_slider,
                        'slider_wl': wl_slider,
                        'is_visible': True,
                        'opacity': 1.0,
                        'current_slice_index': z_size // 2 if img_dimension > 2 else 0,
                        'window_level': initial_level,
                        'window_width': initial_width,
                        'min_intensity': min_val,
                        'max_intensity': max_val,
                        'is_mask': is_mask,
                        'contours': {},
                        'edited_contours': {}
                    }
                    self.loaded_images.append(image_data)

                    item_main_h_layout = QHBoxLayout()
                    item_main_h_layout.addWidget(info_label, 1)
                    item_main_h_layout.addWidget(visible_checkbox)
                    item_main_h_layout.addWidget(item_controls_container)

                    item_widget = QWidget()
                    item_widget.setLayout(item_main_h_layout)
                    item_widget.setStyleSheet("border: 1px solid lightgray; padding: 5px;")

                    item_widget.setContextMenuPolicy(Qt.CustomContextMenu)
                    item_widget.customContextMenuRequested.connect(
                        lambda pos, data=image_data: self.show_item_context_menu(pos, data)
                    )

                    self.image_list_layout.insertWidget(self.image_list_layout.count() - 1, item_widget)

                    visible_checkbox.stateChanged.connect(lambda state, data=image_data: self.toggle_image_visibility(state, data))
                    if not is_mask:
                        opacity_slider.valueChanged.connect(lambda value, data=image_data: self.update_image_opacity(value, data))
                        ww_slider.valueChanged.connect(lambda value, data=image_data: self.update_window_width(value, data))
                        wl_slider.valueChanged.connect(lambda value, data=image_data: self.update_window_level(value, data))
                    else:
                         opacity_slider.valueChanged.connect(lambda value, data=image_data: self.update_image_opacity(value, data))


                except Exception as e:
                    print(f"读取文件 {file_path} 时发生错误: {e}")
                    error_label = QLabel(f"加载失败: {os.path.basename(file_path)}\n错误: {e}")
                    error_label.setStyleSheet("color: red;")
                    self.image_list_layout.insertWidget(self.image_list_layout.count() - 1, error_label)

            self.update_right_display()

        else:
            pass

    def extract_edges_action(self, checked=False):
        print("用户点击了 '提取边缘' 按钮")

        if cv2 is None:
            print("边缘填充功能需要 OpenCV 库，但它未安装。请运行 'pip install opencv-python' 进行安装。")
            # Still proceed with contour extraction if skimage is available, but filling won't work
            # The loop below uses skimage, so it's fine. The warning is printed.

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

            image_data['contours'] = {} # Clear existing cached contours for this image
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
        menu = QMenu(self)
        save_action = QAction("保存图像", self)
        menu.addAction(save_action)

        save_action.triggered.connect(lambda: self.save_specific_image(image_data))

        menu.exec_(self.sender().mapToGlobal(pos))

    def save_specific_image(self, image_data):
        print(f"尝试保存图像: {os.path.basename(image_data.get('path', 'Unknown'))}")
        if image_data and image_data.get('sitk_image'):
            sitk_image = image_data['sitk_image']
            original_path = image_data['path']
            default_file_name = os.path.basename(original_path)
            file_name_base, file_extension = os.path.splitext(default_file_name)

            options = QFileDialog.Options()
            filters = "MHA Files (*.mha);;NRRD Files (*.nrrd);;所有文件 (*)"
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
                    # The sitk_image object reference in image_data should point to the potentially modified image
                    sitk.WriteImage(sitk_image, save_path)
                    print(f"图像保存成功到: {save_path}")
                except Exception as e:
                    print(f"保存图像时发生错误: {e}")
            else:
                print("取消了保存。")

        else:
            print("图像数据无效，无法保存")

    def toggle_image_visibility(self, state, image_data):
        image_data['is_visible'] = (state == Qt.Checked)
        image_data['slider_opacity'].setEnabled(image_data['is_visible'])
        if not image_data.get('is_mask', False):
             image_data['slider_ww'].setEnabled(image_data['is_visible'])
             image_data['slider_wl'].setEnabled(image_data['is_visible'])

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

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        delta_y = event.angleDelta().y()

        has_visible_3d = any(img_data.get('is_visible', False) and img_data.get('sitk_image') and img_data['sitk_image'].GetDimension() > 2 for img_data in self.loaded_images)

        if delta_y != 0 and has_visible_3d:
            event.accept()

            slice_direction = 1 if delta_y > 0 else -1

            for image_data in self.loaded_images:
                if image_data.get('is_visible', False) and image_data.get('sitk_image') and image_data['sitk_image'].GetDimension() > 2:
                    current_slice = image_data.get('current_slice_index', 0)
                    z_size = image_data['sitk_image'].GetSize()[2]
                    new_slice = max(0, min(current_slice + slice_direction, z_size - 1))
                    image_data['current_slice_index'] = new_slice

            # Clear dragging/selection state on slice change
            self._is_dragging = False
            self._selected_point = None

            self.update_right_display()

        else:
            super().wheelEvent(event)


# --- Application Launch Code ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())