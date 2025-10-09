#!/usr/bin/env python
"""
BPDneo-CXR GUI Application Entry Point.

This script launches the graphical user interface for BPD prediction
from chest X-rays.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from PySide6.QtWidgets import QApplication  # noqa: E402
from bpd_ui.ui.main_window import MainWindow  # noqa: E402
from bpd_ui.ui.single_eval_tab import SingleEvalTab  # noqa: E402


def main():
    """
    Launch the BPDneo-CXR GUI application.

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("BPDneo-CXR")
    app.setOrganizationName("BPDneo Team")

    # Set application style
    app.setStyle("Fusion")

    # Create main window
    window = MainWindow()

    # Create and set single evaluation tab
    single_eval_tab = SingleEvalTab()
    window.set_single_eval_tab(single_eval_tab)

    # Show window
    window.show()

    # Start event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
