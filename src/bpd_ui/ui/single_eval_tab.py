"""Single image evaluation tab for BPDneo-CXR application."""

import torch
from pathlib import Path
from PIL import Image
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
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage

from bpd_ui.models.model_util import (
    load_pretrained_model,
    get_preprocessing_transforms,
    list_available_models,
)


class InferenceWorker(QThread):
    """
    Worker thread for running model inference without blocking the UI.

    Parameters
    ----------
    model_name : str
        Name of the model to use for inference
    image_path : str
        Path to the image file
    device : torch.device
        Device to run inference on

    Signals
    -------
    finished : Signal(dict)
        Emitted when inference completes with results dictionary
    error : Signal(str)
        Emitted when an error occurs during inference
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model_name, image_path, device):
        super().__init__()
        self.model_name = model_name
        self.image_path = image_path
        self.device = device

    def run(self):
        """Run the inference pipeline in a separate thread."""
        try:
            # Load model
            model = load_pretrained_model(self.model_name, device=self.device)

            # Load and preprocess image
            img = Image.open(self.image_path)
            transform = get_preprocessing_transforms(self.model_name)
            img_tensor = transform(img)

            # Run inference
            img_batch = img_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = model(img_batch)
                prob = torch.sigmoid(logits).item()
                binary_label = "Moderate/Severe BPD" if prob >= 0.5 else "No/Mild BPD"

            # Emit results
            results = {
                "probability": prob,
                "binary_label": binary_label,
                "logits": logits.item(),
                "model_name": self.model_name,
            }
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class SingleEvalTab(QWidget):
    """
    Tab for evaluating a single chest X-ray image.

    Provides interface for:
    - Loading an X-ray image
    - Selecting a model
    - Running inference
    - Displaying results (probability and binary label)

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget, by default None

    Attributes
    ----------
    image_path : Path or None
        Currently loaded image path
    device : torch.device
        Device for running inference (CPU only)
    worker : InferenceWorker or None
        Current inference worker thread
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.device = torch.device("cpu")  # CPU-only for deployment
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Initialize the user interface components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Single Image Evaluation")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Image selection group
        image_group = self._create_image_selection_group()
        layout.addWidget(image_group)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background: #f5f5f5;")
        self.image_label.setText("No image loaded")
        layout.addWidget(self.image_label, stretch=1)

        # Model selection and inference controls
        controls_group = self._create_controls_group()
        layout.addWidget(controls_group)

        # Results display
        results_group = self._create_results_group()
        layout.addWidget(results_group)

    def _create_image_selection_group(self):
        """
        Create the image selection group box.

        Returns
        -------
        QGroupBox
            Group box containing image selection controls
        """
        group = QGroupBox("Image Selection")
        layout = QHBoxLayout()

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #666;")
        layout.addWidget(self.file_path_label, stretch=1)

        select_btn = QPushButton("Select Image...")
        select_btn.clicked.connect(self._select_image)
        layout.addWidget(select_btn)

        group.setLayout(layout)
        return group

    def _create_controls_group(self):
        """
        Create the model selection and inference controls group.

        Returns
        -------
        QGroupBox
            Group box containing model selection and run button
        """
        group = QGroupBox("Model Selection and Inference")
        layout = QHBoxLayout()

        # Model selection
        layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        models = list_available_models()
        for model_name, info in models.items():
            display_text = f"{model_name} (AUROC: {info['auroc']:.3f})"
            self.model_combo.addItem(display_text, userData=model_name)
        # Set default to best model
        default_idx = self.model_combo.findData("bpd_xrv_progfreeze_lp_cutmix")
        if default_idx >= 0:
            self.model_combo.setCurrentIndex(default_idx)
        layout.addWidget(self.model_combo, stretch=1)

        # Run inference button
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self._run_inference)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            QPushButton:hover:!disabled {
                background-color: #45a049;
            }
        """
        )
        layout.addWidget(self.run_btn)

        group.setLayout(layout)
        return group

    def _create_results_group(self):
        """
        Create the results display group.

        Returns
        -------
        QGroupBox
            Group box containing results text area
        """
        group = QGroupBox("Results")
        layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setText("No results yet. Select an image and run inference.")
        layout.addWidget(self.results_text)

        group.setLayout(layout)
        return group

    def _select_image(self):
        """
        Open file dialog to select an X-ray image.

        Supported formats: PNG, JPG, JPEG, DICOM
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select X-ray Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.dcm);;All Files (*)",
        )

        if file_path:
            self.image_path = Path(file_path)
            self.file_path_label.setText(str(self.image_path))
            self.run_btn.setEnabled(True)

            # Display image
            self._display_image(self.image_path)

    def _display_image(self, image_path):
        """
        Display the selected image in the UI.

        Parameters
        ----------
        image_path : Path
            Path to the image file
        """
        try:
            # Load image using PIL
            pil_image = Image.open(image_path)

            # Convert to grayscale if needed for display
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert PIL Image to QPixmap
            img_array = np.array(pil_image)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                img_array.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_image)

            # Scale to fit display while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                512,
                512,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{str(e)}")

    def _run_inference(self):
        """Start inference on the selected image using the selected model."""
        if self.image_path is None:
            return

        # Disable button during inference
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.results_text.setText(
            "Running inference, please wait...\n(Loading model and processing image)"
        )

        # Get selected model
        model_name = self.model_combo.currentData()

        # Create and start worker thread
        self.worker = InferenceWorker(model_name, str(self.image_path), self.device)
        self.worker.finished.connect(self._on_inference_complete)
        self.worker.error.connect(self._on_inference_error)
        self.worker.start()

    def _on_inference_complete(self, results):
        """
        Handle completion of inference.

        Parameters
        ----------
        results : dict
            Dictionary containing inference results with keys:
            - probability: float
            - binary_label: str
            - logits: float
            - model_name: str
        """
        # Format results for display
        prob = results["probability"]
        label = results["binary_label"]
        model_name = results["model_name"]

        results_text = f"""
Inference Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: {model_name}
Image: {self.image_path.name}

Prediction
──────────
Binary Label: {label}

Probabilities
──────────
P(Moderate/Severe BPD): {prob:.4f}
P(No/Mild BPD): {1-prob:.4f}

Raw Output
──────────
Logits: {results['logits']:.4f}
        """.strip()

        self.results_text.setText(results_text)

        # Re-enable button
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Inference")

    def _on_inference_error(self, error_msg):
        """
        Handle inference error.

        Parameters
        ----------
        error_msg : str
            Error message from the worker thread
        """
        self.results_text.setText(f"Error during inference:\n\n{error_msg}")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Inference")
