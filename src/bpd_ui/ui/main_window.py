"""Main window for BPDneo-CXR application."""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    """
    Main application window with three tabs for different workflow modes.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget, by default None

    Attributes
    ----------
    tab_widget : QTabWidget
        Tab container for the three modes
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BPDneo-CXR: BPD Prediction from Chest X-rays")
        self.setMinimumSize(1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Add tabs (will be replaced with actual implementations)
        self._setup_tabs()

    def _setup_tabs(self):
        """
        Initialize the three main tabs.

        The tabs are:
        - Preprocessing: ROI extraction and grading
        - Dataset Evaluation: Batch inference and metrics
        - Single Image Evaluation: Test single image end-to-end
        """
        # Tab 1: Preprocessing (placeholder for now)
        preprocess_tab = self._create_placeholder_tab("Preprocessing Mode")
        self.tab_widget.addTab(preprocess_tab, "Preprocessing")

        # Tab 2: Dataset Evaluation (placeholder for now)
        dataset_tab = self._create_placeholder_tab("Dataset Evaluation Mode")
        self.tab_widget.addTab(dataset_tab, "Dataset Evaluation")

        # Tab 3: Single Image Evaluation (to be implemented)
        single_tab = self._create_placeholder_tab("Single Image Evaluation Mode")
        self.tab_widget.addTab(single_tab, "Single Image Evaluation")

    def _create_placeholder_tab(self, title):
        """
        Create a placeholder tab widget.

        Parameters
        ----------
        title : str
            Title text to display in the placeholder

        Returns
        -------
        QWidget
            Placeholder widget with centered title
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(f"{title}\n(Coming Soon)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 18px; color: #666;")
        layout.addWidget(label)
        return widget

    def set_single_eval_tab(self, tab_widget):
        """
        Replace the placeholder single evaluation tab with the actual implementation.

        Parameters
        ----------
        tab_widget : QWidget
            The actual single evaluation tab widget
        """
        self.tab_widget.removeTab(2)  # Remove placeholder
        self.tab_widget.insertTab(2, tab_widget, "Single Image Evaluation")
