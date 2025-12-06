# main.py

import sys
import numpy as np

from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QToolBar,
    QFileDialog,
    QSizePolicy,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QPushButton,
)
from PyQt6.QtGui import QAction, QPixmap, QImage, QTransform
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize, QTimer

from core.adjustments import apply_all_adjustments, apply_detail_np
from core.fsc_tone import apply_fsc_tone

from engine import Engine

class ImageLabel(QtWidgets.QLabel):
    """
    Bildanzeige mit Crop-Overlay:
    - setCropMode(True/False)
    - setInitialCropRect(QRect)
    - getCropRect() -> QRect | None
    """
    clickedAt = pyqtSignal(QPoint)
    selectionMade = pyqtSignal(QRect)

    HANDLE_RADIUS = 6
    MIN_SIZE = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(QtGui.QPalette.ColorRole.Dark)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setScaledContents(False)

        # Crop-Status
        self._crop_enabled: bool = False
        self._crop_rect: QtCore.QRect | None = None

        # Drag-Status
        self._drag_mode: str | None = None  # "move" | "resize" | None
        self._drag_handle: int | None = None  # 0..7
        self._drag_start_pos: QtCore.QPoint | None = None
        self._drag_start_rect: QtCore.QRect | None = None

    # -------------------------------------------------- API

    def setCropMode(self, enabled: bool) -> None:
        self._crop_enabled = bool(enabled)
        if not enabled:
            self._drag_mode = None
            self._drag_handle = None
        self.update()

    def setInitialCropRect(self, rect: QtCore.QRect) -> None:
        if rect is None:
            self._crop_rect = None
        else:
            self._crop_rect = QtCore.QRect(rect)
        self.update()

    def getCropRect(self) -> QtCore.QRect | None:
        if self._crop_rect is None:
            return None
        return QtCore.QRect(self._crop_rect)

    # -------------------------------------------------- intern: Handle-Positionen

    def _handle_points(self) -> list[QtCore.QPoint]:
        """
        8 Handle-Punkte in Label-Koordinaten:
        0: TL, 1: TM, 2: TR, 3: MR, 4: BR, 5: BM, 6: BL, 7: ML
        """
        pts: list[QtCore.QPoint] = []
        if self._crop_rect is None:
            return pts

        r = self._crop_rect
        cx = r.center().x()
        cy = r.center().y()

        pts.append(QtCore.QPoint(r.left(), r.top()))          # TL
        pts.append(QtCore.QPoint(cx, r.top()))                # TM
        pts.append(QtCore.QPoint(r.right(), r.top()))         # TR
        pts.append(QtCore.QPoint(r.right(), cy))              # MR
        pts.append(QtCore.QPoint(r.right(), r.bottom()))      # BR
        pts.append(QtCore.QPoint(cx, r.bottom()))             # BM
        pts.append(QtCore.QPoint(r.left(), r.bottom()))       # BL
        pts.append(QtCore.QPoint(r.left(), cy))               # ML
        return pts

    def _hit_test_handle(self, pos: QtCore.QPoint) -> int | None:
        if self._crop_rect is None:
            return None
        for idx, pt in enumerate(self._handle_points()):
            if (pt - pos).manhattanLength() <= self.HANDLE_RADIUS + 2:
                return idx
        return None

    def _cursor_for_handle(self, idx: int) -> QtGui.QCursor:
        # einfache Version, keine perfekte Winkel-Logik
        if idx in (0, 4):   # TL / BR
            return QtGui.QCursor(Qt.CursorShape.SizeFDiagCursor)
        if idx in (2, 6):   # TR / BL
            return QtGui.QCursor(Qt.CursorShape.SizeBDiagCursor)
        if idx in (1, 5):   # TM / BM
            return QtGui.QCursor(Qt.CursorShape.SizeVerCursor)
        if idx in (3, 7):   # MR / ML
            return QtGui.QCursor(Qt.CursorShape.SizeHorCursor)
        return QtGui.QCursor(Qt.CursorShape.ArrowCursor)

    # -------------------------------------------------- Events

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Wenn kein Crop-Modus aktiv ist: Klick an MainWindow weitergeben
        if not self._crop_enabled:
            if ev.button() == Qt.MouseButton.LeftButton:
                self.clickedAt.emit(ev.position().toPoint())
            return super().mousePressEvent(ev)

        # Ab hier: Crop-Modus aktiv
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)

        if self._crop_rect is None:
            # Neues Rechteck beginnen
            pos = ev.position().toPoint()
            self._crop_rect = QtCore.QRect(pos, pos)
            self._drag_mode = "resize"
            self._drag_handle = 4  # BR
            self._drag_start_pos = pos
            self._drag_start_rect = QtCore.QRect(self._crop_rect)
            self.setCursor(self._cursor_for_handle(4))
            self.update()
            return

        pos = ev.position().toPoint()
        self._drag_start_pos = pos
        self._drag_start_rect = QtCore.QRect(self._crop_rect)

        handle = self._hit_test_handle(pos)
        if handle is not None:
            self._drag_mode = "resize"
            self._drag_handle = handle
            self.setCursor(self._cursor_for_handle(handle))
        elif self._crop_rect.contains(pos):
            self._drag_mode = "move"
            self._drag_handle = None
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            # Klick außerhalb -> neues Rechteck aufziehen
            self._drag_mode = "resize"
            self._drag_handle = 4  # BR
            self._crop_rect = QtCore.QRect(pos, pos)
            self._drag_start_rect = QtCore.QRect(self._crop_rect)
            self.setCursor(self._cursor_for_handle(4))

        self.update()


    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        pos = ev.position().toPoint()

        if self._crop_enabled and self._drag_mode is None and self._crop_rect is not None:
            # nur Hover: Cursor ändern
            handle = self._hit_test_handle(pos)
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif self._crop_rect.contains(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if not self._crop_enabled or self._crop_rect is None or self._drag_mode is None:
            return super().mouseMoveEvent(ev)

        assert self._drag_start_pos is not None
        assert self._drag_start_rect is not None

        dx = pos.x() - self._drag_start_pos.x()
        dy = pos.y() - self._drag_start_pos.y()
        r = QtCore.QRect(self._drag_start_rect)

        if self._drag_mode == "move":
            r.translate(dx, dy)
            # innerhalb des Labels begrenzen
            r.moveLeft(max(0, min(self.width() - r.width(), r.left())))
            r.moveTop(max(0, min(self.height() - r.height(), r.top())))
        elif self._drag_mode == "resize":
            idx = self._drag_handle
            if idx is None:
                return

            # je nach Handle Seiten verschieben
            if idx in (0, 6, 7):  # links betroffen
                new_left = r.left() + dx
                new_left = max(0, min(new_left, r.right() - self.MIN_SIZE))
                r.setLeft(new_left)
            if idx in (2, 3, 4):  # rechts betroffen
                new_right = r.right() + dx
                new_right = min(self.width() - 1, max(new_right, r.left() + self.MIN_SIZE))
                r.setRight(new_right)
            if idx in (0, 1, 2):  # oben betroffen
                new_top = r.top() + dy
                new_top = max(0, min(new_top, r.bottom() - self.MIN_SIZE))
                r.setTop(new_top)
            if idx in (4, 5, 6):  # unten betroffen
                new_bottom = r.bottom() + dy
                new_bottom = min(self.height() - 1, max(new_bottom, r.top() + self.MIN_SIZE))
                r.setBottom(new_bottom)

        self._crop_rect = r.normalized()
        self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._crop_enabled and ev.button() == Qt.MouseButton.LeftButton:
            # Bei loslassen im Crop-Modus das finale Rect melden
            if self._crop_rect is not None:
                self.selectionMade.emit(QtCore.QRect(self._crop_rect))
            self._drag_mode = None
            self._drag_handle = None
            self._drag_start_pos = None
            self._drag_start_rect = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

        return super().mouseReleaseEvent(ev)


    def leaveEvent(self, ev: QtCore.QEvent) -> None:
        if not self._crop_enabled or self._drag_mode is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        return super().leaveEvent(ev)

    # -------------------------------------------------- Zeichnen

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        super().paintEvent(ev)

        if not self._crop_enabled or self._crop_rect is None:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        r = self._crop_rect

        # dunkle Maske außen herum
        painter.save()
        painter.setBrush(QtGui.QColor(0, 0, 0, 120))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)

        outer = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRect(outer)
        inner = QtGui.QPainterPath()
        inner.addRect(QtCore.QRectF(r))
        path = path.subtracted(inner)
        painter.drawPath(path)

        painter.restore()

        # Rahmen
        pen = QtGui.QPen(QtGui.QColor(80, 160, 255), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(r)

        # 8 Handles
        painter.setBrush(QtGui.QColor(80, 160, 255))
        painter.setPen(QtGui.QPen(QtGui.QColor(20, 60, 120), 1))

        for pt in self._handle_points():
            painter.drawEllipse(
                pt,
                self.HANDLE_RADIUS,
                self.HANDLE_RADIUS,
            )

        painter.end()


class ResetSlider(QtWidgets.QSlider):
    def __init__(self, orientation, default_value=0, parent=None):
        super().__init__(orientation, parent)
        self._default_value = default_value

    def mouseDoubleClickEvent(self, event):
        # double click -> reset to default and force valueChanged
        self.setValue(self._default_value)
        self.valueChanged.emit(self._default_value)
        super().mouseDoubleClickEvent(event)

class TonePanel(QtWidgets.QWidget):
    gammaChanged = QtCore.pyqtSignal(float)
    contrastChanged = QtCore.pyqtSignal(float)
    exposureChanged = QtCore.pyqtSignal(float)
    highlightsChanged = QtCore.pyqtSignal(float)
    shadowsChanged = QtCore.pyqtSignal(float)
    blackpointChanged = QtCore.pyqtSignal(float)
    whitepointChanged = QtCore.pyqtSignal(float)
    cmyChanged = QtCore.pyqtSignal(float, float, float)
    saturationChanged = QtCore.pyqtSignal(float)
    noiseChanged = QtCore.pyqtSignal(float)
    sharpenChanged = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)

        # --- Tone group (main) ---
        tone_group = QtWidgets.QGroupBox("Tone")
        tone_lay = QtWidgets.QFormLayout(tone_group)

        self.s_exposure = self._make_slider(-50, 50, 0)   # ~ -2..+2 EV
        self.s_gamma = self._make_slider(-50, 50, 0)      # ~ 0.5..1.5
        self.s_contrast = self._make_slider(-50, 50, 0)
        self.s_saturation = self._make_slider(-100, 100, 0)

        # Exposure row
        self.lbl_exposure_val = QtWidgets.QLabel("0.0 EV")
        self.lbl_exposure_val.setFixedWidth(50)
        self.lbl_exposure_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_e = QtWidgets.QHBoxLayout()
        row_e.addWidget(self.s_exposure)
        row_e.addWidget(self.lbl_exposure_val)
        tone_lay.addRow("Exposure     ", row_e)

        # Gamma row
        self.lbl_gamma_val = QtWidgets.QLabel("1.00")
        self.lbl_gamma_val.setFixedWidth(50)
        self.lbl_gamma_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_g = QtWidgets.QHBoxLayout()
        row_g.addWidget(self.s_gamma)
        row_g.addWidget(self.lbl_gamma_val)
        tone_lay.addRow("Gamma     ", row_g)

        # Contrast row
        self.lbl_contrast_val = QtWidgets.QLabel("0.00")
        self.lbl_contrast_val.setFixedWidth(50)
        self.lbl_contrast_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_c = QtWidgets.QHBoxLayout()
        row_c.addWidget(self.s_contrast)
        row_c.addWidget(self.lbl_contrast_val)
        tone_lay.addRow("Contrast     ", row_c)

        # Saturation row
        self.lbl_sat_val = QtWidgets.QLabel("1.00")
        self.lbl_sat_val.setFixedWidth(50)
        self.lbl_sat_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_sat = QtWidgets.QHBoxLayout()
        row_sat.addWidget(self.s_saturation)
        row_sat.addWidget(self.lbl_sat_val)
        tone_lay.addRow("Saturation          ", row_sat)

        # --- CMY group ---
        cmy_group = QtWidgets.QGroupBox("CMY")
        cmy_lay = QtWidgets.QFormLayout(cmy_group)

        self.s_cyan = self._make_slider(-100, 100, 0)
        self.s_mag = self._make_slider(-100, 100, 0)
        self.s_yel = self._make_slider(-100, 100, 0)

        self.lbl_cyan_val = QtWidgets.QLabel("0.00")
        self.lbl_cyan_val.setFixedWidth(50)
        self.lbl_cyan_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_cy = QtWidgets.QHBoxLayout()
        row_cy.addWidget(self.s_cyan)
        row_cy.addWidget(self.lbl_cyan_val)
        cmy_lay.addRow("Cyan/Red", row_cy)

        self.lbl_mag_val = QtWidgets.QLabel("0.00")
        self.lbl_mag_val.setFixedWidth(50)
        self.lbl_mag_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_mg = QtWidgets.QHBoxLayout()
        row_mg.addWidget(self.s_mag)
        row_mg.addWidget(self.lbl_mag_val)
        cmy_lay.addRow("Magenta/Green", row_mg)

        self.lbl_yel_val = QtWidgets.QLabel("0.00")
        self.lbl_yel_val.setFixedWidth(50)
        self.lbl_yel_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_yl = QtWidgets.QHBoxLayout()
        row_yl.addWidget(self.s_yel)
        row_yl.addWidget(self.lbl_yel_val)
        cmy_lay.addRow("Yellow/Blue", row_yl)

        # CMY gradients (wie vorher)
        self.s_cyan.setStyleSheet("""
QSlider::groove:horizontal {
    height: 6px;
    border-radius: 3px;
    background: qlineargradient(
        x1:0, y1:0.5, x2:1, y2:0.5,
        stop:0 #00ffff, stop:1 #ff0000);
}
QSlider::handle:horizontal {
    width: 12px;
    background: #dddddd;
    border: 1px solid #555555;
    margin: -4px 0;
    border-radius: 6px;
}
""")

        self.s_mag.setStyleSheet("""
QSlider::groove:horizontal {
    height: 6px;
    border-radius: 3px;
    background: qlineargradient(
        x1:0, y1:0.5, x2:1, y2:0.5,
        stop:0 #ff00ff, stop:1 #00ff00);
}
QSlider::handle:horizontal {
    width: 12px;
    background: #dddddd;
    border: 1px solid #555555;
    margin: -4px 0;
    border-radius: 6px;
}
""")

        self.s_yel.setStyleSheet("""
QSlider::groove:horizontal {
    height: 6px;
    border-radius: 3px;
    background: qlineargradient(
        x1:0, y1:0.5, x2:1, y2:0.5,
        stop:0 #ffff00, stop:1 #0000ff);
}
QSlider::handle:horizontal {
    width: 12px;
    background: #dddddd;
    border: 1px solid #555555;
    margin: -4px 0;
    border-radius: 6px;
}
""")

        # --- Ranges group (Shadows/Highlights/Black/White) ---
        ranges_group = QtWidgets.QGroupBox("Ranges")
        ranges_lay = QtWidgets.QFormLayout(ranges_group)

        self.s_shadow = self._make_slider(-100, 100, 0)
        self.s_high = self._make_slider(-100, 100, 0)
        self.s_black = self._make_slider(-100, 100, 0)
        self.s_white = self._make_slider(-100, 100, 0)

        self.lbl_shadow_val = QtWidgets.QLabel("0.00")
        self.lbl_shadow_val.setFixedWidth(50)
        self.lbl_shadow_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_s = QtWidgets.QHBoxLayout()
        row_s.addWidget(self.s_shadow)
        row_s.addWidget(self.lbl_shadow_val)
        ranges_lay.addRow("Shadows", row_s)

        self.lbl_high_val = QtWidgets.QLabel("0.00")
        self.lbl_high_val.setFixedWidth(50)
        self.lbl_high_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_h = QtWidgets.QHBoxLayout()
        row_h.addWidget(self.s_high)
        row_h.addWidget(self.lbl_high_val)
        ranges_lay.addRow("Highlights", row_h)

        self.lbl_black_val = QtWidgets.QLabel("0.00")
        self.lbl_black_val.setFixedWidth(50)
        self.lbl_black_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_b = QtWidgets.QHBoxLayout()
        row_b.addWidget(self.s_black)
        row_b.addWidget(self.lbl_black_val)
        ranges_lay.addRow("Blackpoint         ", row_b)

        self.lbl_white_val = QtWidgets.QLabel("0.00")
        self.lbl_white_val.setFixedWidth(50)
        self.lbl_white_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_w = QtWidgets.QHBoxLayout()
        row_w.addWidget(self.s_white)
        row_w.addWidget(self.lbl_white_val)
        ranges_lay.addRow("Whitepoint", row_w)

        # --- Detail group (Noise / Sharpen) ---
        detail_group = QtWidgets.QGroupBox("Detail")
        detail_lay = QtWidgets.QFormLayout(detail_group)

        self.s_noise = self._make_slider(0, 100, 0)    # 0..1
        self.s_sharpen = self._make_slider(0, 100, 0)  # 0..1

        self.lbl_noise_val = QtWidgets.QLabel("0.00")
        self.lbl_noise_val.setFixedWidth(50)
        self.lbl_noise_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_n = QtWidgets.QHBoxLayout()
        row_n.addSpacing(32)
        row_n.addWidget(self.s_noise)
        row_n.addWidget(self.lbl_noise_val)
        detail_lay.addRow("Noise", row_n)

        self.lbl_sharp_val = QtWidgets.QLabel("0.00")
        self.lbl_sharp_val.setFixedWidth(50)
        self.lbl_sharp_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row_sh = QtWidgets.QHBoxLayout()
        row_sh.addSpacing(32)
        row_sh.addWidget(self.s_sharpen)
        row_sh.addWidget(self.lbl_sharp_val)
        detail_lay.addRow("Sharpen", row_sh)

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_reset_tone = QtWidgets.QPushButton("Reset tone")
        self.btn_reset_color = QtWidgets.QPushButton("Reset color")
        btn_row.addWidget(self.btn_reset_tone)
        btn_row.addWidget(self.btn_reset_color)

        lay.addWidget(tone_group)
        lay.addWidget(cmy_group)
        lay.addWidget(ranges_group)
        lay.addWidget(detail_group)
        lay.addLayout(btn_row)
        lay.addStretch(1)

        # connections -> emit normalized values
        self.s_exposure.valueChanged.connect(self._emit_exposure)
        self.s_gamma.valueChanged.connect(self._emit_gamma)
        self.s_contrast.valueChanged.connect(self._emit_contrast)
        self.s_saturation.valueChanged.connect(self._emit_saturation)

        self.s_cyan.valueChanged.connect(self._emit_cmy)
        self.s_mag.valueChanged.connect(self._emit_cmy)
        self.s_yel.valueChanged.connect(self._emit_cmy)

        self.s_shadow.valueChanged.connect(self._emit_shadow)
        self.s_high.valueChanged.connect(self._emit_high)
        self.s_black.valueChanged.connect(self._emit_black)
        self.s_white.valueChanged.connect(self._emit_white)

        self.s_noise.valueChanged.connect(self._emit_noise)
        self.s_sharpen.valueChanged.connect(self._emit_sharpen)

        self.btn_reset_tone.clicked.connect(self._on_reset_tone)
        self.btn_reset_color.clicked.connect(self._on_reset_color)

        # initial labels
        self._emit_exposure(self.s_exposure.value())
        self._emit_gamma(self.s_gamma.value())
        self._emit_contrast(self.s_contrast.value())
        self._emit_saturation(self.s_saturation.value())
        self._emit_cmy(0)
        self._emit_shadow(self.s_shadow.value())
        self._emit_high(self.s_high.value())
        self._emit_black(self.s_black.value())
        self._emit_white(self.s_white.value())
        self._emit_noise(self.s_noise.value())
        self._emit_sharpen(self.s_sharpen.value())

    # ---------- Helper: Slider mit fester Breite + Doppelklick-Reset ----------
    def _make_slider(self, min_val: int, max_val: int, default: int) -> QtWidgets.QSlider:
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setValue(default)
        s.setSingleStep(1)
        s.setPageStep(5)
        s.setMinimumWidth(160)
        s.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        s.default_value = default
        s.installEventFilter(self)
        return s

    def eventFilter(self, obj, ev):
        if isinstance(obj, QtWidgets.QSlider) and ev.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            if hasattr(obj, "default_value"):
                obj.setValue(obj.default_value)
            return True
        return super().eventFilter(obj, ev)

    # ---------- Emit-Handler ----------
    def _emit_exposure(self, v: int):
        ev = v / 25.0  # -2..+2
        self.lbl_exposure_val.setText(f"{ev:+.1f} EV")
        self.exposureChanged.emit(ev)

    def _emit_gamma(self, v: int):
        gamma = 1.0 + v / 100.0  # ~0.5..1.5
        self.lbl_gamma_val.setText(f"{gamma:.2f}")
        self.gammaChanged.emit(gamma)

    def _emit_contrast(self, v: int):
        c = v / 100.0  # -0.5..+0.5
        self.lbl_contrast_val.setText(f"{c:+.2f}")
        self.contrastChanged.emit(c)

    def _emit_saturation(self, v: int):
        # -100..100 -> 0.5..1.5
        sat = 1.0 + (v / 200.0)
        self.lbl_sat_val.setText(f"{sat:.2f}")
        self.saturationChanged.emit(sat)

    def _emit_cmy(self, _v: int):
        c = self.s_cyan.value() / 100.0
        m = self.s_mag.value() / 100.0
        y = self.s_yel.value() / 100.0
        self.lbl_cyan_val.setText(f"{c:+.2f}")
        self.lbl_mag_val.setText(f"{m:+.2f}")
        self.lbl_yel_val.setText(f"{y:+.2f}")
        self.cmyChanged.emit(c, m, y)

    def _emit_shadow(self, v: int):
        val = v / 100.0
        self.lbl_shadow_val.setText(f"{val:+.2f}")
        self.shadowsChanged.emit(val)

    def _emit_high(self, v: int):
        val = v / 100.0
        self.lbl_high_val.setText(f"{val:+.2f}")
        self.highlightsChanged.emit(val)

    def _emit_black(self, v: int):
        val = v / 100.0
        self.lbl_black_val.setText(f"{val:+.2f}")
        self.blackpointChanged.emit(val)

    def _emit_white(self, v: int):
        val = v / 100.0
        self.lbl_white_val.setText(f"{val:+.2f}")
        self.whitepointChanged.emit(val)

    def _emit_noise(self, v: int):
        val = v / 100.0
        self.lbl_noise_val.setText(f"{val:.2f}")
        self.noiseChanged.emit(val)

    def _emit_sharpen(self, v: int):
        val = v / 100.0
        self.lbl_sharp_val.setText(f"{val:.2f}")
        self.sharpenChanged.emit(val)

    # ---------- Reset-Buttons ----------
    def _on_reset_tone(self):
        self.s_exposure.setValue(0)
        self.s_gamma.setValue(0)
        self.s_contrast.setValue(0)
        self.s_saturation.setValue(0)
        self.s_shadow.setValue(0)
        self.s_high.setValue(0)
        self.s_black.setValue(0)
        self.s_white.setValue(0)

    def _on_reset_color(self):
        self.s_cyan.setValue(0)
        self.s_mag.setValue(0)
        self.s_yel.setValue(0)
        self.s_noise.setValue(0)
        self.s_sharpen.setValue(0)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Converter Clean – Claude Negativ Preview")

        # Engine
        self.engine = Engine()
        # Preview-Größe an UI anpassen (kleinere Previews für Hochformat)
        self.engine.preview_max_dim = 1200  # statt fix 1200
        # Basis-Preview (skalierte Positiv-Vorschau), auf die Tone-Slider wirken
        self.base_preview_np: np.ndarray | None = None
        self._crop_preview_scale: float = 1.0

        # Mapping von Bild -> Label (für Crop)
        self._view_scale: float | None = None
        self._view_offset_x: int = 0
        self._view_offset_y: int = 0
        self._view_img_size: tuple[int, int] | None = None  # (W, H)
        self.crop_mode: bool = False


        # Geometrie-Status für das Arbeitsbild (Preview)
        self.geom_orientation = 0  # 0,1,2,3 -> 0°, 90°, 180°, 270°
        self.geom_mirror_h = False
        self.geom_mirror_v = False


        # Tone / Color UI state (für Preview-Slider)
        self.ui_exposure = 0.0
        self.ui_gamma = 1.0
        self.ui_contrast = 0.0
        self.ui_saturation = 1.0

        self.ui_cyan = 0.0
        self.ui_magenta = 0.0
        self.ui_yellow = 0.0

        self.ui_shadows = 0.0
        self.ui_highlights = 0.0
        self.ui_blackpoint = 0.0
        self.ui_whitepoint = 0.0

        self.ui_noise = 0.0
        self.ui_sharpen = 0.0

        #Zentrales Bild-Label
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Wichtig: Größe darf nicht durch die Pixmap erzwungen werden
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.image_label.setMinimumSize(0, 0)


        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 40, 40, 40)  # Rahmen: links/oben/rechts/unten
        layout.addWidget(self.image_label)

        self.setCentralWidget(central)
        # Timer für "refit nach Resize", damit das Bild nicht bei jedem Pixel verschmiert
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._refit_image_to_view)

        # Merker für RAW-Anzeige vor dem Convert
        self._last_raw_qimage: QImage | None = None

        self.image_label.clickedAt.connect(self.on_image_clicked)

        self.pixmap_before_crop: QPixmap | None = None

        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)


        act_open = QAction("Open RAW…", self)
        act_open.triggered.connect(self.on_open_raw)
        toolbar.addAction(act_open)

        self.act_wb = QAction("WB from border", self)
        self.act_wb.setCheckable(True)
        toolbar.addAction(self.act_wb)

        self.act_convert = QAction("Convert", self)
        self.act_convert.triggered.connect(self.on_convert)
        toolbar.addAction(self.act_convert)

        # Crop & Rotate
        self.act_crop = QAction("Crop", self)
        self.act_crop.setCheckable(True)
        self.act_crop.toggled.connect(self.on_crop_toggled)
        toolbar.addAction(self.act_crop)

        self.act_rot_left = QAction("Rotate ⟲ 90°", self)
        self.act_rot_left.triggered.connect(self.on_rotate_left)
        toolbar.addAction(self.act_rot_left)

        self.act_rot_right = QAction("Rotate ⟳ 90°", self)
        self.act_rot_right.triggered.connect(self.on_rotate_right)
        toolbar.addAction(self.act_rot_right)

        self.act_recalc = QAction("Recalculate", self)
        self.act_recalc.triggered.connect(self.on_recalculate)
        self.act_recalc.setEnabled(False)
        toolbar.addAction(self.act_recalc)

        # Export
        self.act_export = QAction("Export JPEG…", self)
        self.act_export.triggered.connect(self.on_export)
        self.act_export.setEnabled(False)
        toolbar.addAction(self.act_export)




        # Tone / Color Dock rechts andocken
        self.tone_panel = TonePanel()

        self.tone_dock = QDockWidget("Tone / Color", self)
        self.tone_dock.setWidget(self.tone_panel)
        self.tone_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tone_dock)

        # --- Thumbnails-Dock links (Platzhalter) ---
        self.thumb_panel = QWidget()
        thumb_layout = QVBoxLayout(self.thumb_panel)
        thumb_layout.setContentsMargins(8, 8, 8, 8)
        thumb_layout.setSpacing(8)

        thumb_placeholder = QLabel("Thumbnails\n(noch nicht implementiert)")
        thumb_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_placeholder.setStyleSheet("color: #aaaaaa;")
        thumb_layout.addWidget(thumb_placeholder)

        self.thumb_dock = QDockWidget("Thumbnails", self)
        self.thumb_dock.setWidget(self.thumb_panel)
        self.thumb_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        # gleiche (maximale) Breite wie das Tone/Color-Dock rechts
        self.thumb_dock.setMaximumWidth(self.tone_dock.maximumWidth())
        self.thumb_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.thumb_dock)

        # erstmal deaktiviert, bis ein Positiv vorhanden ist
        self.tone_panel.setEnabled(False)


        # Nur maximale Breite begrenzen, damit es im Vollbild nicht riesig wird
        self.tone_dock.setMaximumWidth(420)  # Wert nach Gefühl anpassen (z.B. 400–450)

        # Abdocken/Schließen deaktivieren (optional, wie gewünscht)
        self.tone_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tone_dock)



        # erstmal deaktiviert, bis ein Positiv vorhanden ist
        self.tone_panel.setEnabled(False)


        self.tone_panel.exposureChanged.connect(self.on_exposure_changed)
        self.tone_panel.gammaChanged.connect(self.on_gamma_changed)
        self.tone_panel.contrastChanged.connect(self.on_contrast_changed)
        self.tone_panel.highlightsChanged.connect(self.on_highlights_changed)
        self.tone_panel.shadowsChanged.connect(self.on_shadows_changed)
        self.tone_panel.blackpointChanged.connect(self.on_blackpoint_changed)
        self.tone_panel.whitepointChanged.connect(self.on_whitepoint_changed)
        self.tone_panel.cmyChanged.connect(self.on_cmy_changed)
        self.tone_panel.saturationChanged.connect(self.on_saturation_changed)
        self.tone_panel.noiseChanged.connect(self.on_noise_changed)
        self.tone_panel.sharpenChanged.connect(self.on_sharpen_changed)


        # Startzustand
        self.wb_mode = False
        self.act_convert.setEnabled(False)

        self.act_wb.toggled.connect(self.on_wb_toggled)

        # Startgröße
        self.resize(1200, 800)

    # ---------- Slots ----------


    def on_open_raw(self) -> None:
        # Dateidialog – zunächst einfach alle Dateien anzeigen
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RAW file",
            "",
            "RAW files (*.*)",
        )
        if not path:
            return

        try:
            qimg = self.engine.load_raw(path)
        except Exception as e:
            # Ganz pragmatisch: Fehler im Fenster-Titel anzeigen
            self.setWindowTitle(f"Error: {type(e).__name__}: {e}")
            return

        # <<< HIER NEU: RAW-Preview für spätere Resize-Anpassung merken >>>
        self._last_raw_qimage = qimg

        # ❗ Statt direkt setPixmap: immer über _show_qimage einpassen
        self._show_qimage(qimg)

        self.setWindowTitle(f"Converter Clean – {Path(path).name}")
        self.act_convert.setEnabled(True)

        self.statusBar().showMessage("RAW geladen. Optional WB vom Rand (WB-Button)", 5000)

    def on_wb_toggled(self, checked: bool) -> None:
        self.wb_mode = checked
        if checked:
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.statusBar().showMessage("WB-Modus: klick auf den Filmrand.")
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().clearMessage()

    def on_image_clicked(self, pos: QPoint) -> None:
        """
        Wird aufgerufen, wenn im WB-Modus auf das Bild geklickt wird.
        Setzt WB vom Rand und zeigt das aktualisierte Negativ.
        """
        if not self.wb_mode:
            return

        pix = self.image_label.pixmap()
        if pix is None:
            return

        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pix_w = pix.width()
        pix_h = pix.height()

        offset_x = max(0, (label_w - pix_w) // 2)
        offset_y = max(0, (label_h - pix_h) // 2)

        x = pos.x() - offset_x
        y = pos.y() - offset_y
        if x < 0 or y < 0 or x >= pix_w or y >= pix_h:
            return

        try:
            qimg = self.engine.set_wb_from_border_preview_point(x, y)
        except Exception as e:
            self.statusBar().showMessage(f"WB error: {type(e).__name__}: {e}", 5000)
            return

        # ❗ Wieder: nicht direkt setPixmap, sondern unser zentrales _show_qimage
        self._show_qimage(qimg)

        self.wb_mode = False
        self.act_wb.setChecked(False)
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.statusBar().showMessage("WB vom Rand gesetzt.", 3000)

    def on_tone_values_changed(self, values: dict) -> None:
        """
        Wird aufgerufen, wenn Slider im TonePanel bewegt werden.
        values enthält z.B.:
        {
            "exposure": 0,
            "gamma": 10,
            "contrast": -20,
            "cyan": 5,
            "magenta": 0,
            "yellow": -10,
        }
        """
        if self.engine.pos_full_orig is None:
            return

        # Nur die aktiven Slider an die Engine durchreichen
        self.engine.gamma_slider = values.get("gamma", 0)
        self.engine.contrast_slider = values.get("contrast", 0)
        self.engine.cyan_slider = values.get("cyan", 0)
        self.engine.magenta_slider = values.get("magenta", 0)
        self.engine.yellow_slider = values.get("yellow", 0)

        try:
            qimg = self.engine._make_cropped_preview()
        except Exception as e:
            self.statusBar().showMessage(
                f"Tone error: {type(e).__name__}: {e}", 5000
            )
            return

        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def on_convert(self) -> None:
        try:
            qimg = self.engine.convert_to_positive(use_analysis_crop=False)
        except Exception as e:
            self.statusBar().showMessage(
                f"Convert error: {type(e).__name__}: {e}", 5000
            )
            return

        if qimg is None:
            self.statusBar().showMessage("Convert error: no image returned.", 5000)
            return

        # Basis-Preview für Tone-Slider merken UND Anzeige einpassen
        self._set_base_preview_from_qimage(qimg)

        # Export & Recalculate ab jetzt möglich
        if hasattr(self, "act_export"):
            self.act_export.setEnabled(True)
        if hasattr(self, "act_recalc"):
            self.act_recalc.setEnabled(True)
        if hasattr(self, "tone_panel"):
            self.tone_panel.setEnabled(True)

        self.statusBar().showMessage("Converted.", 3000)

    def on_recalculate(self) -> None:
        try:
            qimg = self.engine.recalculate()
        except Exception as e:
            self.statusBar().showMessage(f"Recalculate error: {type(e).__name__}: {e}", 5000)
            return

        # Basis-Preview anpassen (neue Konvertierung)
        self._set_base_preview_from_qimage(qimg)

        if hasattr(self, "act_export"):
            self.act_export.setEnabled(True)

        # Tone-Panel aktiv lassen, neu auf Basis-Preview arbeiten
        if hasattr(self, "tone_panel"):
            self.tone_panel.setEnabled(True)

        self.statusBar().showMessage("Recalculated.", 3000)


    # ------- Preview-Helfer (QImage <-> numpy) -------

    def _qimage_to_np(self, qimg: QImage) -> np.ndarray:
        """QImage (RGB888) -> float32-Array 0..1."""
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(3 * w * h)
        arr = np.frombuffer(ptr, np.uint8).reshape(h, w, 3)
        return arr.astype(np.float32) / 255.0

    def _np_to_qimage(self, arr: np.ndarray) -> QImage:
        """float32-Array 0..1 -> QImage (RGB888)."""
        arr_u8 = np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        # WICHTIG: dafür sorgen, dass das Array wirklich C-contiguous ist
        arr_u8 = np.ascontiguousarray(arr_u8)

        h, w, _ = arr_u8.shape
        qimg = QImage(arr_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return qimg.copy()

    def _show_qimage(self, qimg):
        # Zielgröße = aktuelle sichtbare Größe des ImageLabels
        target_w = self.image_label.width()
        target_h = self.image_label.height()

        if target_w <= 0 or target_h <= 0:
            pix = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pix)
            return

        # Bild proportional in target_w x target_h einpassen
        scaled = qimg.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

    def _set_base_preview_from_qimage(self, qimg: QImage) -> None:
        """
        Speichert die aktuelle Positiv-Vorschau als Basis-Preview (np-Array)
        und zeigt sie im ImageLabel an (skaliert auf den verfügbaren Platz).
        """
        self.base_preview_np = self._qimage_to_np(qimg)
        self._show_qimage(qimg)

    def _refit_image_to_view(self) -> None:
        """Bild passend in die aktuelle Bildmitte einpassen (nach Resize)."""

        # Wenn das Label (z.B. beim Start) noch keine sinnvolle Größe hat, abbrechen
        if self.image_label.width() <= 0 or self.image_label.height() <= 0:
            return

        # 1) Nach Convert/Recalculate: aus der Arbeits-Preview neu rendern
        if self.base_preview_np is not None:
            arr = self._apply_tone_to_preview(self.base_preview_np)
            qimg = self._np_to_qimage(arr)
            self._show_qimage(qimg)
            return

        # 2) Vor Convert: RAW-Preview, falls vorhanden, neu einpassen
        if self._last_raw_qimage is not None:
            self._show_qimage(self._last_raw_qimage)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)

        # Bei jedem Resize den Timer neu starten – nur wenn der Nutzer
        # kurz aufhört zu ziehen, wird wirklich neu gerendert.
        if hasattr(self, "_resize_timer") and self._resize_timer is not None:
            self._resize_timer.start(150)  # 150 ms kannst du nach Geschmack anpassen



    # --- Crop
    def _enter_crop_mode(self) -> None:
        """Switch ImageLabel into crop mode with full positive preview."""
        if self.engine.pos_full_orig is None:
            return

        try:
            qfull = self.engine.get_full_preview_for_crop()
        except Exception as e:
            print(f"Crop preview error: {e}")
            return

        # aktuelles Bild merken (für Cancel / Fehlerfall)
        self.pixmap_before_crop = self.image_label.pixmap()

        # Full-Preview auf die Größe des ImageLabels skalieren
        target_w = self.image_label.width()
        target_h = self.image_label.height()
        if target_w > 0 and target_h > 0:
            qdisp = qfull.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            qdisp = qfull

        self.image_label.setPixmap(QPixmap.fromImage(qdisp))

        # Scale-Faktor von Full-Preview -> Label merken
        if qfull.width() > 0 and qfull.height() > 0:
            self._crop_preview_scale = qdisp.width() / qfull.width()
        else:
            self._crop_preview_scale = 1.0

        # Initiales Crop-Rect: vorhandenes pos_crop_rect benutzen, falls vorhanden
        init_rect = None
        if (
            self.engine.pos_crop_rect is not None
            and self.engine.full_scale_x is not None
            and self.engine.full_scale_y is not None
        ):
            x, y, w, h = self.engine.pos_crop_rect
            # Original -> Full-Preview
            x0_p = int(x * self.engine.full_scale_x)
            y0_p = int(y * self.engine.full_scale_y)
            x1_p = int((x + w) * self.engine.full_scale_x)
            y1_p = int((y + h) * self.engine.full_scale_y)

            # Full-Preview -> Label
            s = self._crop_preview_scale
            x0_l = int(x0_p * s + 0.5)
            y0_l = int(y0_p * s + 0.5)
            x1_l = int(x1_p * s + 0.5)
            y1_l = int(y1_p * s + 0.5)

            init_rect = QRect(
                min(x0_l, x1_l),
                min(y0_l, y1_l),
                abs(x1_l - x0_l),
                abs(y1_l - y0_l),
            )

        # Wenn kein vorheriger Crop: fast das ganze Bild nehmen
        if init_rect is None:
            r = self.image_label.rect()
            margin = 20
            init_rect = r.adjusted(margin, margin, -margin, -margin)

        self.image_label.setInitialCropRect(init_rect)
        self.image_label.setCropMode(True)

    def _apply_crop_from_label(self) -> None:
        """Read crop rect from ImageLabel and apply it to Engine."""
        rect = self.image_label.getCropRect()
        self.image_label.setCropMode(False)

        # Kein Crop-Rechteck -> Vorschau zurücksetzen
        if rect is None:
            if self.pixmap_before_crop is not None:
                self.image_label.setPixmap(self.pixmap_before_crop)
            return

        # Label-Koordinaten -> Full-Preview-Koordinaten
        s = getattr(self, "_crop_preview_scale", 1.0)
        x_p = int(rect.x() / s)
        y_p = int(rect.y() / s)
        w_p = int(rect.width() / s)
        h_p = int(rect.height() / s)

        try:
            qimg = self.engine.apply_crop_from_full_preview_rect(x_p, y_p, w_p, h_p)
        except Exception as e:
            print(f"Apply crop error: {e}")
            if self.pixmap_before_crop is not None:
                self.image_label.setPixmap(self.pixmap_before_crop)
            return

        # Ab hier ist der neue Crop aktiv: Basis-Preview aktualisieren
        self._set_base_preview_from_qimage(qimg)

    @QtCore.pyqtSlot(bool)
    def on_crop_toggled(self, checked: bool) -> None:
        """Crop-Modus ein/aus."""
        if checked:
            if self.base_preview_np is None:
                self.act_crop.setChecked(False)
                return

            self.crop_mode = True
            # Einfach: fast das ganze Bild als Start-Rechteck
            r = self.image_label.rect()
            margin = 20
            init_rect = r.adjusted(margin, margin, -margin, -margin)
            self.image_label.setInitialCropRect(init_rect)
            self.image_label.setCropMode(True)
        else:
            self._apply_crop_from_label()

    def _apply_crop_from_label(self) -> None:
        """Liest das Crop-Rechteck aus dem ImageLabel und cropt base_preview_np."""
        self.crop_mode = False
        self.image_label.setCropMode(False)

        if self.base_preview_np is None:
            self._refresh_tone_preview()
            return

        rect = self.image_label.getCropRect()
        pix = self.image_label.pixmap()
        if rect is None or pix is None:
            self._refresh_tone_preview()
            return

        # Größen von Label und angezeigtem Pixmap
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pix_w = pix.width()
        pix_h = pix.height()

        # Pixmap ist zentriert im Label
        offset_x = max(0, (label_w - pix_w) // 2)
        offset_y = max(0, (label_h - pix_h) // 2)

        # Crop-Rechteck in "Pixmap-Koordinaten" umrechnen
        x0_l = rect.x() - offset_x
        y0_l = rect.y() - offset_y
        x1_l = x0_l + rect.width()
        y1_l = y0_l + rect.height()

        # Auf den sichtbaren Bereich der Pixmap begrenzen
        x0_l = max(0, min(pix_w - 1, x0_l))
        y0_l = max(0, min(pix_h - 1, y0_l))
        x1_l = max(0, min(pix_w, x1_l))
        y1_l = max(0, min(pix_h, y1_l))

        if x1_l <= x0_l + 2 or y1_l <= y0_l + 2:
            self._refresh_tone_preview()
            return

        # Pixmap -> Arbeitsbild (base_preview_np)
        H, W = self.base_preview_np.shape[:2]
        s = min(pix_w / float(W), pix_h / float(H))
        if s <= 0:
            self._refresh_tone_preview()
            return

        x0 = int(round(x0_l / s))
        y0 = int(round(y0_l / s))
        x1 = int(round(x1_l / s))
        y1 = int(round(y1_l / s))

        # Auf Bildgrenzen begrenzen
        x0 = max(0, min(W - 1, x0))
        y0 = max(0, min(H - 1, y0))
        x1 = max(x0 + 1, min(W, x1))
        y1 = max(y0 + 1, min(H, y1))

        # 👉 destruktiver Crop auf das Arbeitsbild
        self.base_preview_np = self.base_preview_np[y0:y1, x0:x1].copy()
        self._refresh_tone_preview()


    # ------ Slider pipeline

    def _apply_tone_to_preview(self, base_srgb: np.ndarray) -> np.ndarray:
        """
        Nimmt die Basis-Preview (sRGB 0..1),
        wendet alle Tone/Color-Adjustments an
        und gibt wieder sRGB zurück.

        Nutzt jetzt die core.adjustments Funktionen direkt.
        """
        if base_srgb is None:
            return None

        from core.adjustments import _srgb_to_linear, _linear_to_srgb

        # sRGB → linear
        img_lin = _srgb_to_linear(base_srgb)

        # Tone & Color Adjustments
        img_lin_adj = self.apply_tone_and_color(img_lin)

        # linear → sRGB
        img_srgb = _linear_to_srgb(img_lin_adj)

        return np.clip(img_srgb, 0.0, 1.0)

    def apply_tone_and_color(self, img_linear: np.ndarray) -> np.ndarray:
        """
        Wendet alle Tone/Color-Adjustments mit der neuen unified Funktion an.
        Arbeitet auf linearem RGB (0..1, float32).
        """

        if img_linear is None:
            return None

        # ----- UI-Werte lesen -----
        exposure = float(getattr(self, "ui_exposure", 0.0))      # -4..+4 EV
        gamma = float(getattr(self, "ui_gamma", 1.0))            # 0.2..3.0
        contrast = float(getattr(self, "ui_contrast", 0.0))      # -1..+1
        saturation_ui = float(getattr(self, "ui_saturation", 1.0))  # 0..2

        # Direkt durchreichen: 0..2 → 0..200 % in apply_fsc_tone
        # 0   → 0   (komplett entsättigt)
        # 1   → 100 (neutral, wie FSC)
        # 2   → 200 (max, wie FSC)
        saturation = max(0.0, min(2.0, saturation_ui))


        cyan = float(getattr(self, "ui_cyan", 0.0))              # -1..+1
        magenta = float(getattr(self, "ui_magenta", 0.0))        # -1..+1
        yellow = float(getattr(self, "ui_yellow", 0.0))          # -1..+1

        shadows = float(getattr(self, "ui_shadows", 0.0))        # -1..+1
        highlights = float(getattr(self, "ui_highlights", 0.0))  # -1..+1
        blackpoint = float(getattr(self, "ui_blackpoint", 0.0))  # -1..+1
        whitepoint = float(getattr(self, "ui_whitepoint", 0.0))  # -1..+1

        nr_amount = float(getattr(self, "ui_noise", 0.0))        # 0..1
        sharpen_amount = float(getattr(self, "ui_sharpen", 0.0)) # 0..1

        # ----- Tone & Color Adjustments (ALLES in einem Schritt) -----
        img_adjusted = apply_all_adjustments(
            img_linear,
            exposure=exposure,
            gamma=gamma,
            contrast=contrast,
            saturation=saturation,
            cyan=cyan,
            magenta=magenta,
            yellow=yellow,
            shadows=shadows,
            highlights=highlights,
            blackpoint=blackpoint,
            whitepoint=whitepoint,
        )

        # ----- Detail (NR + Sharpening) -----
        img_final = apply_detail_np(img_adjusted, nr_amount, sharpen_amount)

        return img_final

    def _refresh_tone_preview(self) -> None:
            # Während des Crop-Modus nichts rendern,
            # sonst übermalen wir die Full-Crop-Preview mit dem Arbeitsbild
            if getattr(self, "crop_mode", False):
                return

            if self.base_preview_np is None:
                return

            arr = self._apply_tone_to_preview(self.base_preview_np)
            qimg = self._np_to_qimage(arr)
            self._show_qimage(qimg)

    def _apply_shadows_highlights_srgb(
        self,
        img_srgb: np.ndarray,
        sh: float,
        hi: float,
    ) -> np.ndarray:
        """
        Vereinfachte, stabile Shadows/Highlights:
        - sh, hi in etwa -1..+1
        - wirkt nur selektiv auf dunkle bzw. helle Bereiche
        - vermeidet Blocking und krasse Invertierungen
        """
        x = np.clip(img_srgb, 0.0, 1.0).astype(np.float32)
        if abs(sh) < 1e-3 and abs(hi) < 1e-3:
            return x

        eps = 1e-6

        # Luminanz nach Rec.709
        R = x[..., 0]
        G = x[..., 1]
        B = x[..., 2]
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        Y = np.clip(Y, 0.0, 1.0)

        Y_new = Y.copy()

        # -------- Shadows --------
        if abs(sh) > 1e-3:
            # Gewicht: 1 bei Y=0, 0 bei Y>=0.5
            w_sh = np.clip((0.5 - Y) / 0.5, 0.0, 1.0)

            # Max. Änderung in Luma: ca. +/-0.35
            amt_sh = 0.35 * sh  # sh>0: heller, sh<0: dunkler
            Y_new += w_sh * amt_sh

        # Luma nach Shadow-Anpassung clampen
        Y_new = np.clip(Y_new, 0.0, 1.0)

        # RGB an neue Luma anpassen
        ratio_sh = (Y_new + eps) / (Y + eps)
        x = x * ratio_sh[..., None]

        # Neue Luma für Highlights aus aktuellem x
        R2 = x[..., 0]
        G2 = x[..., 1]
        B2 = x[..., 2]
        Y2 = 0.2126 * R2 + 0.7152 * G2 + 0.0722 * B2
        Y2 = np.clip(Y2, 0.0, 1.0)
        Y2_new = Y2.copy()

        # -------- Highlights --------
        if abs(hi) > 1e-3:
            # Gewicht: 0 bei Y<=0.5, 1 bei Y>=1.0
            w_hi = np.clip((Y2 - 0.5) / 0.5, 0.0, 1.0)

            # Max. Änderung in Luma: ca. +/-0.35
            # hi>0: Highlights abdunkeln, hi<0: Highlights heller
            amt_hi = -0.35 * hi
            Y2_new += w_hi * amt_hi

        Y2_new = np.clip(Y2_new, 0.0, 1.0)

        ratio_hi = (Y2_new + eps) / (Y2 + eps)
        x = x * ratio_hi[..., None]

        return np.clip(x, 0.0, 1.0)

    def _apply_saturation_hsv(self, img_srgb: np.ndarray, sat_ui: float) -> np.ndarray:
        """
        Sättigung in HSV:
        - H bleibt gleich
        - V bleibt gleich
        - S wird so geändert, dass:
          * low-sat Farben stärker profitieren
          * high-sat Farben weniger stark angefasst werden
        sat_ui kommt aus dem UI: 0..2, Mitte 1.0 = neutral
        """
        import numpy as np

        x = np.clip(img_srgb, 0.0, 1.0).astype(np.float32)

        # sat_ui: 0..2 -> amount: -1..+1
        amount = float(sat_ui) - 1.0
        if abs(amount) < 1e-3:
            return x

        R = x[..., 0]
        G = x[..., 1]
        B = x[..., 2]

        maxc = np.max(x, axis=-1)
        minc = np.min(x, axis=-1)
        delta = maxc - minc

        # Hue
        H = np.zeros_like(maxc)
        mask = delta > 1e-6

        mask_r = mask & (maxc == R)
        mask_g = mask & (maxc == G)
        mask_b = mask & (maxc == B)

        H[mask_r] = ((G[mask_r] - B[mask_r]) / (delta[mask_r] + 1e-6)) % 6.0
        H[mask_g] = ((B[mask_g] - R[mask_g]) / (delta[mask_g] + 1e-6)) + 2.0
        H[mask_b] = ((R[mask_b] - G[mask_b]) / (delta[mask_b] + 1e-6)) + 4.0

        H = (H / 6.0) % 1.0

        V = maxc
        S = delta / (maxc + 1e-6)

        # Neue Saturation:
        # - amount > 0: flache Farben (kleines S) bekommen mehr Boost als schon knallige
        # - amount < 0: alles wird gleichmäßig entsättigt
        S_new = np.empty_like(S)

        if amount >= 0.0:
            # Richtung 1.0 in Abhängigkeit von (1-S): je flacher, desto stärker
            S_new = S + amount * (1.0 - S)
        else:
            # Richtung 0.0
            S_new = S * (1.0 + amount)

        S_new = np.clip(S_new, 0.0, 1.0)

        # Zurück nach RGB
        C = S_new * V
        Hp = H * 6.0
        X = C * (1.0 - np.abs((Hp % 2.0) - 1.0))

        zeros = np.zeros_like(C)

        R1 = np.zeros_like(C)
        G1 = np.zeros_like(C)
        B1 = np.zeros_like(C)

        mask0 = (0.0 <= Hp) & (Hp < 1.0)
        R1[mask0], G1[mask0], B1[mask0] = C[mask0], X[mask0], zeros[mask0]

        mask1 = (1.0 <= Hp) & (Hp < 2.0)
        R1[mask1], G1[mask1], B1[mask1] = X[mask1], C[mask1], zeros[mask1]

        mask2 = (2.0 <= Hp) & (Hp < 3.0)
        R1[mask2], G1[mask2], B1[mask2] = zeros[mask2], C[mask2], X[mask2]

        mask3 = (3.0 <= Hp) & (Hp < 4.0)
        R1[mask3], G1[mask3], B1[mask3] = zeros[mask3], X[mask3], C[mask3]

        mask4 = (4.0 <= Hp) & (Hp < 5.0)
        R1[mask4], G1[mask4], B1[mask4] = X[mask4], zeros[mask4], C[mask4]

        mask5 = (5.0 <= Hp) & (Hp < 6.0)
        R1[mask5], G1[mask5], B1[mask5] = C[mask5], zeros[mask5], X[mask5]

        m = V - C
        R2 = R1 + m
        G2 = G1 + m
        B2 = B1 + m

        out = np.stack([R2, G2, B2], axis=-1)
        return np.clip(out, 0.0, 1.0)


    # ------- Tone / Color callbacks -------

    def on_exposure_changed(self, ev: float) -> None:
        # EV (z.B. -2..+2) aus TonePanel
        self.ui_exposure = max(-4.0, min(4.0, float(ev)))
        self._refresh_tone_preview()

    def on_gamma_changed(self, v: float) -> None:
        self.ui_gamma = max(0.2, min(3.0, float(v)))
        self._refresh_tone_preview()

    def on_contrast_changed(self, v: float) -> None:
        # v ca. -0.5 .. +0.5
        self.ui_contrast = max(-1.0, min(1.0, float(v)))
        self._refresh_tone_preview()

    def on_shadows_changed(self, v: float) -> None:
        # v ca. -1..+1
        self.ui_shadows = max(-1.0, min(1.0, float(v)))
        self._refresh_tone_preview()

    def on_highlights_changed(self, v: float) -> None:
        # v ca. -1..+1
        self.ui_highlights = max(-1.0, min(1.0, float(v)))
        self._refresh_tone_preview()

    def on_blackpoint_changed(self, v: float) -> None:
        # v ca. -1..+1
        self.ui_blackpoint = max(-1.0, min(1.0, float(v)))
        self._refresh_tone_preview()

    def on_whitepoint_changed(self, v: float) -> None:
        # v ca. -1..+1
        self.ui_whitepoint = max(-1.0, min(1.0, float(v)))
        self._refresh_tone_preview()

    def on_cmy_changed(self, c: float, m: float, y: float) -> None:
        self.ui_cyan = max(-1.0, min(1.0, float(c)))
        self.ui_magenta = max(-1.0, min(1.0, float(m)))
        self.ui_yellow = max(-1.0, min(1.0, float(y)))
        self._refresh_tone_preview()

    def on_saturation_changed(self, v: float) -> None:
        # v kommt als Faktor (z.B. 0.0..2.0)
        self.ui_saturation = max(0.0, float(v))
        self._refresh_tone_preview()

    def on_noise_changed(self, v: float) -> None:
        self.ui_noise = float(v)
        self._refresh_tone_preview()

    def on_sharpen_changed(self, v: float) -> None:
        self.ui_sharpen = float(v)
        self._refresh_tone_preview()



    # ------- Rotate/Mirror ------
    def on_rotate_left(self) -> None:
        """Arbeitsbild 90° gegen den Uhrzeigersinn drehen."""
        if self.base_preview_np is None:
            return
        self.base_preview_np = np.rot90(self.base_preview_np, k=1)
        self._refresh_tone_preview()

    def on_rotate_right(self) -> None:
        """Arbeitsbild 90° im Uhrzeigersinn drehen."""
        if self.base_preview_np is None:
            return
        self.base_preview_np = np.rot90(self.base_preview_np, k=-1)
        self._refresh_tone_preview()




    #-----
    def _show_negative_preview(self):
        """Show negative with WB applied as preview."""
        try:
            qimg = self.engine._make_negative_preview()
            self.image_label.setPixmapScaled(QPixmap.fromImage(qimg))
        except Exception as e:
            print(f"Warning: {e}")



    # ----- Export ----- #
    def on_export(self) -> None:
        """
        Exportiert genau das, was aktuell im Hauptbild zu sehen ist
        (image_label-Pixmap) als JPEG.
        Damit sind Crop + alle Tone/Color-Slider im Export enthalten.
        """
        pix = self.image_label.pixmap()
        if pix is None:
            QtWidgets.QMessageBox.information(
                self, "Export", "Kein Bild zum Exportieren vorhanden."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JPEG",
            "export.jpg",
            "JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return

        try:
            qimg = pix.toImage().convertToFormat(QImage.Format.Format_RGB888)
            # Qualität 95 wie in 4.0
            ok = qimg.save(path, "JPEG", 95)
            if not ok:
                raise RuntimeError("Speichern ist fehlgeschlagen.")

            self.statusBar().showMessage(f"Exportiert: {path}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export-Fehler", f"{type(e).__name__}: {e}"
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
