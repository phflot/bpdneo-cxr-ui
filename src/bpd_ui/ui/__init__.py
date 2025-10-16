"""
UI Components for BPDneo-CXR Application
=========================================

This module provides the PyQt6-based graphical user interface for the
BPDneo-CXR desktop application with three main workflow modes:

1. **Preprocessing Tab**
   - Import patient images (PNG, JPEG, DICOM)
   - Semi-automated ROI extraction with operator refinement
   - 4-level operator grading (no_bpd, mild, moderate, severe)
   - Save preprocessed data to manifest

2. **Dataset Evaluation Tab**
   - Load prepared dataset from manifest
   - Batch inference on all images
   - Binary classification metrics (AUROC, accuracy, etc.)
   - Export predictions and metrics

3. **Single Image Evaluation Tab**
   - End-to-end evaluation on individual images
   - Optional ROI cropping
   - Display prediction probabilities and binary label

Design Philosophy
-----------------
**Guided workflow ("no wrong moves")**:
- Clear step-by-step progression
- Buttons enable/disable based on state
- Visual feedback for all operations
- Background processing for long operations

UI Components
-------------
MainWindow
    Three-tab main window container
PreprocessTab
    ROI extraction and operator grading workflow
DatasetEvalTab
    Batch inference and metrics computation
SingleEvalTab
    Single-image inference with visualization
OverlayCanvas
    Interactive image canvas with mask overlay and annotation tools

Architecture Notes
------------------
- All long-running operations use ``core.tasks.submit()`` for background execution
- Model inference is lazy-loaded via ``core.ModelManager``
- State is persisted to Excel manifest (no in-memory session state)
- ROI tools: Auto-segmentation, seed points, border polyline, refinement

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bpd_ui.ui import MainWindow
>>> import sys
>>>
>>> app = QApplication(sys.argv)
>>> window = MainWindow()
>>> window.show()
>>> sys.exit(app.exec())

See Also
--------
bpd_ui.core : Application logic and state management
apps.gui_app : Entry point for launching the GUI
"""

__all__ = []
