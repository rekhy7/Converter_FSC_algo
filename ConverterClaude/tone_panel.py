"""
Tone adjustment UI panel.
"""

from PyQt6 import QtWidgets, QtCore


class TonePanel(QtWidgets.QWidget):
    """Panel with tone adjustment sliders."""

    adjustmentsChanged = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QtWidgets.QLabel("<b>Tone Adjustments</b>")
        layout.addWidget(title)

        # Exposure: -2 to +2 EV
        self.slider_exposure = self._create_slider_row(
            "Exposure", -20, 20, 0, layout
        )

        # Gamma: 0.5 to 2.0
        self.slider_gamma = self._create_slider_row(
            "Gamma", 10, 40, 20, layout
        )

        # Contrast: -0.5 to +0.5
        self.slider_contrast = self._create_slider_row(
            "Contrast", -10, 10, 0, layout
        )

        # Saturation: 0.0 to 2.0
        self.slider_saturation = self._create_slider_row(
            "Saturation", 0, 20, 10, layout
        )

        btn_reset = QtWidgets.QPushButton("Reset All")
        btn_reset.clicked.connect(self.reset_all)
        layout.addWidget(btn_reset)

        layout.addStretch()

    def _create_slider_row(self, label, min_val, max_val, default_val, parent_layout):
        lbl = QtWidgets.QLabel(label)
        parent_layout.addWidget(lbl)

        row = QtWidgets.QHBoxLayout()

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 4)

        value_label = QtWidgets.QLabel()
        value_label.setMinimumWidth(60)
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        row.addWidget(slider)
        row.addWidget(value_label)
        parent_layout.addLayout(row)

        return slider, value_label

    def _connect_signals(self):
        self.slider_exposure[0].valueChanged.connect(self._on_slider_changed)
        self.slider_gamma[0].valueChanged.connect(self._on_slider_changed)
        self.slider_contrast[0].valueChanged.connect(self._on_slider_changed)
        self.slider_saturation[0].valueChanged.connect(self._on_slider_changed)
        self._update_labels()

    def _on_slider_changed(self):
        self._update_labels()

        exposure = self.slider_exposure[0].value() / 10.0
        gamma = self.slider_gamma[0].value() / 20.0
        contrast = self.slider_contrast[0].value() / 20.0
        saturation = self.slider_saturation[0].value() / 10.0

        self.adjustmentsChanged.emit(exposure, gamma, contrast, saturation)

    def _update_labels(self):
        exp_val = self.slider_exposure[0].value() / 10.0
        self.slider_exposure[1].setText(f"{exp_val:+.1f} EV")

        gamma_val = self.slider_gamma[0].value() / 20.0
        self.slider_gamma[1].setText(f"{gamma_val:.2f}")

        contrast_val = self.slider_contrast[0].value() / 20.0
        self.slider_contrast[1].setText(f"{contrast_val:+.2f}")

        sat_val = self.slider_saturation[0].value() / 10.0
        self.slider_saturation[1].setText(f"{sat_val:.2f}")

    def reset_all(self):
        self.slider_exposure[0].setValue(0)
        self.slider_gamma[0].setValue(20)
        self.slider_contrast[0].setValue(0)
        self.slider_saturation[0].setValue(10)
