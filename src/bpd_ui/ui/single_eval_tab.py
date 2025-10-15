"""Single image evaluation tab for BPDneo-CXR application."""

from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QGroupBox,
    QTextEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from bpd_ui.models.model_util import list_available_models
from bpd_ui.core.model_manager import ModelManager
from bpd_ui.core.image_io import load_image_for_display, load_image_for_model
from bpd_ui.core.tasks import submit


class SingleEvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.mm = ModelManager("cpu")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Single Image Evaluation")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        layout.addWidget(self._create_image_selection_group())
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background: #f5f5f5;")
        self.image_label.setText("No image loaded")
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self._create_controls_group())
        layout.addWidget(self._create_results_group())

    def _create_image_selection_group(self):
        g = QGroupBox("Image Selection")
        h = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #666;")
        h.addWidget(self.file_path_label, stretch=1)
        b = QPushButton("Select Image...")
        b.clicked.connect(self._select_image)
        h.addWidget(b)
        g.setLayout(h)
        return g

    def _create_controls_group(self):
        g = QGroupBox("Model Selection and Inference")
        h = QHBoxLayout()
        h.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        models = list_available_models()
        for model_name, info in models.items():
            t = (
                f"{model_name} (AUROC: {info['auroc']:.3f})"
                if "auroc" in info
                else model_name
            )
            self.model_combo.addItem(t, userData=model_name)
        idx = self.model_combo.findData("bpd_xrv_progfreeze_lp_cutmix")
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self._run_inference)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;padding:10px;border-radius:5px;} QPushButton:disabled{background:#ccc;} QPushButton:hover:!disabled{background:#45a049;}"
        )
        h.addWidget(self.model_combo, stretch=1)
        h.addWidget(self.run_btn)
        g.setLayout(h)
        return g

    def _create_results_group(self):
        g = QGroupBox("Results")
        v = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setText("No results yet. Select an image and run inference.")
        v.addWidget(self.results_text)
        g.setLayout(v)
        return g

    def _select_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select X-ray Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.dcm);;All Files (*)",
        )
        if not fp:
            return
        self.image_path = Path(fp)
        self.file_path_label.setText(str(self.image_path))
        self.run_btn.setEnabled(True)
        self._display_image(self.image_path)

    def _display_image(self, image_path):
        pil_image = load_image_for_display(image_path)
        arr = np.array(pil_image)
        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            512,
            512,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)

    def _run_inference(self):
        if self.image_path is None:
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.results_text.setText("Running inference...")
        model_name = self.model_combo.currentData()
        img = load_image_for_model(self.image_path)
        sigs = submit(self.mm.predict, model_name, img)
        sigs.finished.connect(self._on_done)
        sigs.error.connect(self._on_err)

    def _on_done(self, res):
        prob, logit = res
        label = "Moderate/Severe BPD" if prob >= 0.5 else "No/Mild BPD"
        s = f"Inference Complete\n\nModel: {self.model_combo.currentData()}\nImage: {self.image_path.name}\n\nBinary Label: {label}\n\nP(Moderate/Severe BPD): {prob:.4f}\nP(No/Mild BPD): {1-prob:.4f}\n\nLogits: {logit:.4f}"
        self.results_text.setText(s)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Inference")

    def _on_err(self, msg):
        self.results_text.setText(f"Error: {msg}")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Inference")
