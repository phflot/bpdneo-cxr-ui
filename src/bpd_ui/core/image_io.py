"""
Image I/O Utilities
===================

This module provides unified image loading functions that handle both standard
image formats (PNG, JPEG) and DICOM files. It separates display-oriented
loading (RGB conversion, normalization) from model-oriented loading (preserve
original data for preprocessing pipelines).

Functions
---------
load_image_for_display
    Load and convert image to RGB format for GUI display
load_image_for_model
    Load image preserving original data for model preprocessing

Notes
-----
DICOM handling:
- Extracts pixel_array and normalizes to 0-255 range for display
- Preserves raw pixel_array values for model input
- Converts to PIL Image for consistent interface

Standard formats (PNG, JPEG, etc.) are loaded using PIL.Image.open().

See Also
--------
bpd_ui.models.model_util.get_preprocessing_transforms : Model preprocessing
bpd_ui.ui.preprocess_tab : Uses both loading functions
"""

from pathlib import Path
import numpy as np
from PIL import Image


def load_image_for_display(path: str | Path):
    """
    Load image and convert to RGB for GUI display.

    This function loads an image from disk and prepares it for display in
    the GUI. For DICOM files, it normalizes pixel values to 0-255 range.
    For standard images, it simply converts to RGB.

    Parameters
    ----------
    path : str or Path
        Path to image file. Supported formats:
        - DICOM (.dcm)
        - Standard images (.png, .jpg, .jpeg, etc.)

    Returns
    -------
    PIL.Image
        RGB image ready for display (8-bit, 3 channels)

    Notes
    -----
    DICOM processing:
    1. Read with pydicom
    2. Extract pixel_array as float32
    3. Normalize: subtract min, divide by max
    4. Scale to 0-255 and convert to uint8
    5. Convert to RGB PIL Image

    Standard image processing:
    1. Open with PIL
    2. Convert to RGB mode

    The normalization ensures DICOM images display correctly regardless of
    their original bit depth or value range.

    Examples
    --------
    >>> from bpd_ui.core.image_io import load_image_for_display
    >>> from pathlib import Path
    >>>
    >>> # Load DICOM
    >>> dicom_img = load_image_for_display("xray.dcm")
    >>> print(dicom_img.mode, dicom_img.size)
    RGB (512, 512)
    >>>
    >>> # Load PNG
    >>> png_img = load_image_for_display("xray.png")
    >>> print(png_img.mode)
    RGB

    See Also
    --------
    load_image_for_model : Load without display normalization
    """
    p = str(path)
    if p.lower().endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        return img
    return Image.open(p).convert("RGB")


def load_image_for_model(path: str | Path):
    """
    Load image preserving original values for model preprocessing.

    This function loads an image without display-oriented normalization,
    preserving the original pixel values for model preprocessing pipelines.
    This is critical for maintaining training-time consistency.

    Parameters
    ----------
    path : str or Path
        Path to image file. Supported formats:
        - DICOM (.dcm)
        - Standard images (.png, .jpg, .jpeg, etc.)

    Returns
    -------
    PIL.Image
        Image with original pixel values (grayscale for DICOM, original mode
        for standard images)

    Notes
    -----
    DICOM processing:
    1. Read with pydicom
    2. Extract pixel_array (preserving original dtype/values)
    3. Convert to PIL Image (grayscale)

    Standard image processing:
    1. Open with PIL
    2. Return as-is (no mode conversion)

    **CRITICAL**: This function preserves original values for preprocessing.
    Do not apply display normalization before passing to model transforms.

    Examples
    --------
    >>> from bpd_ui.core.image_io import load_image_for_model
    >>> from bpd_ui.models.model_util import get_preprocessing_transforms
    >>>
    >>> # Load image for inference
    >>> img = load_image_for_model("xray.dcm")
    >>> print(img.mode)  # Original mode preserved
    L
    >>>
    >>> # Apply model-specific preprocessing
    >>> transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")
    >>> tensor = transform(img)
    >>>
    >>> # Standard image
    >>> img2 = load_image_for_model("xray.png")
    >>> # Mode preserved (could be L, RGB, etc.)

    See Also
    --------
    load_image_for_display : Load with display normalization
    bpd_ui.models.model_util.get_preprocessing_transforms : Model preprocessing
    """
    p = str(path)
    if p.lower().endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(p)
        arr = ds.pixel_array
        return Image.fromarray(arr)
    return Image.open(p)
