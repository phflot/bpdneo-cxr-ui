"""Main window for BPDneo-CXR application."""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout
from .preprocess_tab import PreprocessTab
from .dataset_eval_tab import DatasetEvalTab
from .single_eval_tab import SingleEvalTab


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
        self.tab_widget.addTab(PreprocessTab(self), "Preprocessing")
        self.tab_widget.addTab(DatasetEvalTab(self), "Dataset Evaluation")
        self.tab_widget.addTab(SingleEvalTab(self), "Single Image Evaluation")
