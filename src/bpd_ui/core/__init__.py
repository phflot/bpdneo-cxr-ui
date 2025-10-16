"""
Core Application Logic for BPDneo-CXR
======================================

This module contains the core application logic that powers the BPDneo-CXR
desktop application, including:

- **State management**: Excel manifest reading/writing for dataset tracking
- **Image I/O**: Unified loading for DICOM and standard image formats
- **ROI extraction**: Semi-automated chest region segmentation with refinement
- **Model management**: Lazy-loading model cache with preprocessing
- **Metrics computation**: Binary classification metrics (AUROC, AUPRC, etc.)
- **Background tasks**: Thread pool execution for long-running operations
- **Deskewing**: Automatic perspective correction for scanned images

Data Model
----------
The application uses a file-based data model with no database:

- **Folder structure**: ``<root>/prepared/<patient_id>/images/`` and ``/masks/``
- **Manifest**: ``<root>/prepared/manifest.xlsx`` tracks all preprocessed data
- **Schema**: patient_id, image_relpath, roi_relpath, grade_label, preproc_json,
  eval_split, timestamp

ROI Extraction Pipeline
-----------------------
1. **Auto-segmentation**: XRV-based lung/heart/mediastinum detection
2. **User refinement** (optional): Seed points + border polyline
3. **Random Walker**: Refine mask with user constraints
4. **Morphological cleanup**: Closing, opening, hole filling

Binary Classification Metrics
------------------------------
- AUROC (Area Under ROC Curve)
- AUPRC (Average Precision)
- Accuracy, Sensitivity, Specificity
- PPV (Precision), NPV
- Confusion matrix
- ROC curve data

Examples
--------
>>> from bpd_ui.core import ModelManager, load_image_for_model
>>> from PIL import Image
>>>
>>> # Load and predict
>>> mm = ModelManager("cpu")
>>> img = load_image_for_model("xray.jpg")
>>> prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
>>>
>>> # Manifest management
>>> from bpd_ui.core import read_manifest, upsert_manifest
>>> from pathlib import Path
>>>
>>> root = Path("data/dataset")
>>> df = read_manifest(root)
>>> row = {
...     "patient_id": "P001",
...     "image_relpath": "P001/images/xray.dcm",
...     "roi_relpath": "P001/masks/mask.png",
...     "grade_label": "moderate",
...     "preproc_json": "{}",
...     "eval_split": "test"
... }
>>> upsert_manifest(root, row)

Modules
-------
model_manager
    Lazy-loading model cache for GUI inference
image_io
    Load DICOM and standard images for display or model input
state
    Read/write Excel manifest for dataset tracking
roi_service
    Auto-segmentation and user-guided mask refinement
metrics
    Binary classification metrics computation and formatting
tasks
    QThreadPool wrapper for background task execution
deskew
    Perspective correction for scanned X-ray images

See Also
--------
bpd_ui.models : Model loading and preprocessing
bpd_ui.ui : PyQt6 GUI components
"""

from .model_manager import ModelManager
from .image_io import load_image_for_display, load_image_for_model
from .state import read_manifest, upsert_manifest, manifest_path
from .tasks import submit

__all__ = [
    "ModelManager",
    "load_image_for_display",
    "load_image_for_model",
    "read_manifest",
    "upsert_manifest",
    "manifest_path",
    "submit",
]
