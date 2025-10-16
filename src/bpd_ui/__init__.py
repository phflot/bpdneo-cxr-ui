"""
BPDneo-CXR: Early BPD Prediction from Chest X-rays
===================================================

BPDneo-CXR provides a desktop GUI and CLI application for early prediction of
bronchopulmonary dysplasia (BPD) in extremely low birth weight infants using
chest X-rays obtained within 24 hours of birth.

The application supports three main workflows:

1. **Preprocessing mode**: Semi-automated ROI extraction with guided workflow
   and operator grading (4-level: no_bpd, mild, moderate, severe)
2. **Dataset evaluation**: Batch inference with binary classification metrics
   (model outputs binary: "Moderate/Severe" vs "No/Mild")
3. **Single image evaluation**: End-to-end testing on individual images

**Critical constraint**: The pretrained model heads are binary classifiers.
Four-grade labeling is captured during preprocessing for operator annotation,
but model inference remains binary.

Quick Start
-----------
>>> from bpd_ui.models import load_pretrained_model, get_preprocessing_transforms
>>> from PIL import Image
>>>
>>> # Load the best-performing model
>>> model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
>>>
>>> # Get preprocessing transforms
>>> transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")
>>>
>>> # Load and preprocess image
>>> img = Image.open("xray.jpg")
>>> tensor = transform(img).unsqueeze(0)
>>>
>>> # Run inference
>>> import torch
>>> model.eval()
>>> with torch.no_grad():
...     logits = model(tensor)
...     prob = torch.sigmoid(logits).item()
...     label = "Moderate/Severe" if prob >= 0.5 else "No/Mild"

Main Modules
------------
models
    Model backend with pretrained weights, loading, and preprocessing
core
    Application logic: state management, ROI extraction, metrics, I/O
ui
    PyQt6 GUI components for the three workflow modes

See Also
--------
README.md : Project overview and installation instructions
CLAUDE.md : Development guidelines and architecture documentation
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
