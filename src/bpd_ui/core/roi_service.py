"""
ROI Extraction Service
======================

This module provides automatic and semi-automatic region-of-interest (ROI)
extraction for chest X-rays. It includes:

- Automatic chest segmentation using TorchXRayVision PSPNet
- Bounding box computation from masks
- Interactive refinement with seed points and border constraints
- Morphological post-processing

Classes
-------
SegCache
    Lazy-loading cache for TorchXRayVision segmentation model

Functions
---------
xrv_preprocess
    Preprocess PIL image for TorchXRayVision models
auto_chest_mask
    Automatic chest segmentation (lungs + heart + mediastinum)
bbox_from_mask
    Compute bounding box from binary mask with padding
refine_with_scribbles
    Interactive refinement using random walker with seed/border constraints

Notes
-----
The automatic segmentation uses TorchXRayVision's PSPNet trained on chest
X-ray datasets. The model segments multiple anatomical structures including
left/right lungs, heart, and mediastinum.

Interactive refinement uses scikit-image's random_walker algorithm with:
- Seed points: Foreground markers (label 1)
- Border polyline: Exclusion zone (label 2, outside)
- Random walker: Propagates labels based on image gradients

See Also
--------
bpd_ui.ui.preprocess_tab : UI that uses ROI extraction
bpd_ui.ui.overlay_canvas : Seed/border drawing interface
"""

import torch
import numpy as np
import torchxrayvision as xrv
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
    disk,
)
from skimage.segmentation import random_walker


class SegCache:
    """
    Lazy-loading cache for TorchXRayVision segmentation model.

    This class provides lazy initialization of the PSPNet segmentation model,
    loading it only when first requested. The model is cached to avoid repeated
    loading overhead.

    Parameters
    ----------
    device : str, default="cpu"
        Device to run the segmentation model on

    Attributes
    ----------
    device : torch.device
        PyTorch device object
    model : torch.nn.Module or None
        Cached PSPNet model (None until first get() call)
    targets : list or None
        List of segmentation target names (anatomical structures)

    Examples
    --------
    >>> cache = SegCache("cpu")
    >>> model, targets = cache.get()
    >>> print(targets)
    ['Left Lung', 'Right Lung', 'Heart', 'Mediastinum', ...]

    See Also
    --------
    auto_chest_mask : Uses SegCache for automatic segmentation
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = None
        self.targets = None

    def get(self):
        """
        Get the cached model, loading if necessary.

        Returns
        -------
        model : torch.nn.Module
            TorchXRayVision PSPNet in eval mode
        targets : list
            List of anatomical structure names

        Notes
        -----
        On first call, this method:
        1. Loads PSPNet model from TorchXRayVision
        2. Moves model to specified device
        3. Sets model to eval mode
        4. Caches model and targets

        Subsequent calls return cached instances immediately.
        """
        if self.model is None:
            self.model = xrv.baseline_models.chestx_det.PSPNet().to(self.device).eval()
            self.targets = self.model.targets
        return self.model, self.targets


_seg = SegCache("cpu")


def xrv_preprocess(pil_img):
    """
    Preprocess PIL image for TorchXRayVision segmentation models.

    This function applies the preprocessing pipeline required by TorchXRayVision
    models: grayscale conversion, XRV normalization, and resizing to 512x512.

    Parameters
    ----------
    pil_img : PIL.Image
        Input image (any mode)

    Returns
    -------
    torch.Tensor
        Preprocessed tensor with shape (1, 512, 512) ready for PSPNet

    Notes
    -----
    Preprocessing steps:
    1. Convert PIL Image to numpy array
    2. Convert to grayscale if RGB (average channels)
    3. Apply xrv.datasets.normalize(img, 255)
    4. Add channel dimension
    5. Resize to 512×512 using XRayResizer
    6. Convert to float tensor

    This differs from BPD model preprocessing in the resize method.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open("xray.jpg")
    >>> tensor = xrv_preprocess(img)
    >>> print(tensor.shape)
    torch.Size([1, 512, 512])

    See Also
    --------
    auto_chest_mask : Uses this preprocessing
    """
    import torchvision as tv

    img = np.array(pil_img)
    if img.ndim == 3:
        img = img.mean(2)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]
    t = tv.transforms.Compose([xrv.datasets.XRayResizer(512)])
    img = t(img)
    ten = torch.from_numpy(img).float()
    return ten


def auto_chest_mask(pil_img, use_heart=True, use_mediastinum=True):
    """
    Automatic chest segmentation using TorchXRayVision PSPNet.

    This function generates a binary mask of the chest region by combining
    segmentations of lungs, heart, and mediastinum from the PSPNet model.
    The mask is post-processed with morphological operations.

    Parameters
    ----------
    pil_img : PIL.Image
        Input chest X-ray image (any mode)
    use_heart : bool, default=True
        Include heart in the mask
    use_mediastinum : bool, default=True
        Include mediastinum in the mask

    Returns
    -------
    np.ndarray
        Binary mask (uint8, values 0 or 255) with shape (512, 512)

    Notes
    -----
    Processing pipeline:
    1. Preprocess image for XRV model
    2. Run PSPNet segmentation
    3. Combine left lung, right lung, and optional heart/mediastinum
    4. Apply morphological closing (disk radius 3)
    5. Apply morphological opening (disk radius 3)
    6. Remove small objects (< 256 pixels)
    7. Fill small holes (< 256 pixels)

    The lungs are always included. Heart and mediastinum are optional but
    recommended for capturing the full chest region relevant to BPD.

    Examples
    --------
    >>> from PIL import Image
    >>> from bpd_ui.core.roi_service import auto_chest_mask
    >>>
    >>> img = Image.open("xray.jpg")
    >>> mask = auto_chest_mask(img)
    >>> print(mask.shape, mask.dtype, mask.max())
    (512, 512) uint8 255
    >>>
    >>> # Lungs only
    >>> mask_lungs = auto_chest_mask(img, use_heart=False, use_mediastinum=False)

    See Also
    --------
    xrv_preprocess : Preprocessing for segmentation
    refine_with_scribbles : Interactive refinement
    bbox_from_mask : Extract bounding box from mask
    """
    model, targets = _seg.get()
    x = xrv_preprocess(pil_img).to(_seg.device)
    with torch.inference_mode():
        y = model(x[None])
    y = y[0].cpu().numpy()
    idx = [targets.index("Left Lung"), targets.index("Right Lung")]
    if use_heart and "Heart" in targets:
        idx.append(targets.index("Heart"))
    if use_mediastinum and "Mediastinum" in targets:
        idx.append(targets.index("Mediastinum"))
    m = (y[idx] > 0.5).any(0)
    m = binary_closing(m, disk(3))
    m = binary_opening(m, disk(3))
    m = remove_small_objects(m, 256)
    m = remove_small_holes(m, 256)
    return m.astype(np.uint8) * 255


def bbox_from_mask(mask, pad_frac=(0.1, 0.08)):
    """
    Compute bounding box from binary mask with asymmetric padding.

    This function finds the tight bounding box around foreground pixels and
    adds padding. The padding is asymmetric: more padding above (to capture
    upper chest) and symmetric horizontal padding.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) with foreground > 0
    pad_frac : tuple of float, default=(0.1, 0.08)
        Padding fractions (vertical_up, horizontal) as fraction of image size

    Returns
    -------
    tuple
        Bounding box as (x0, y0, x1, y1) in pixel coordinates.
        Returns full image bounds if mask is empty.

    Notes
    -----
    Padding strategy:
    - Vertical: Add pad_frac[0] * H pixels above (captures upper chest)
    - Horizontal: Add pad_frac[1] * W pixels on both sides (symmetric)

    The default padding (0.1, 0.08) adds 10% height above and 8% width on
    each side. This asymmetric padding is tailored for chest X-rays where
    the upper thorax is clinically relevant.

    Bounds are clipped to image dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from bpd_ui.core.roi_service import bbox_from_mask
    >>>
    >>> # Create sample mask
    >>> mask = np.zeros((512, 512), dtype=np.uint8)
    >>> mask[100:400, 150:350] = 255
    >>>
    >>> # Get bounding box with padding
    >>> bbox = bbox_from_mask(mask)
    >>> print(bbox)  # (x0, y0, x1, y1)
    (109, 49, 391, 400)
    >>>
    >>> # Custom padding
    >>> bbox_tight = bbox_from_mask(mask, pad_frac=(0.0, 0.0))

    See Also
    --------
    auto_chest_mask : Generates masks for bounding box extraction
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    H, W = mask.shape
    dy_up = int(H * pad_frac[0])
    dx = int(W * pad_frac[1])
    y0 = max(0, y0 - dy_up)
    x0 = max(0, x0 - dx)
    x1 = min(W - 1, x1 + dx)
    return (x0, y0, x1, y1)


def refine_with_scribbles(gray, seeds_xy, border_xy, init_mask=None):
    """
    Interactive ROI refinement using random walker with seed/border constraints.

    This function refines a segmentation using user-provided seed points
    (foreground markers) and an optional border polyline (exclusion zone).
    It uses the random walker algorithm to propagate labels based on image
    gradients.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (H, W) as numpy array
    seeds_xy : list of tuple
        List of (x, y) seed point coordinates marking foreground
    border_xy : list of tuple or None
        List of (x, y) border polyline points marking exclusion zone, or None
    init_mask : np.ndarray, optional
        Initial mask (currently unused, reserved for future use)

    Returns
    -------
    np.ndarray
        Refined binary mask (uint8, values 0 or 255) with shape (H, W)

    Notes
    -----
    Algorithm:
    1. Create label map (0 = unlabeled, 1 = foreground, 2 = background)
    2. Mark seed points as foreground (label 1)
    3. If border provided, mark outside region as background (label 2)
    4. Run random walker (beta=90, mode='bf') to propagate labels
    5. Extract foreground mask (label == 1)
    6. Apply morphological closing (disk radius 3)
    7. Apply morphological opening (disk radius 3)
    8. Remove small objects (< 128 pixels)
    9. Fill small holes (< 128 pixels)

    The random walker algorithm treats the image as a graph where pixels are
    nodes connected by edges weighted by image gradients. It propagates the
    seed labels based on these weights, producing smooth boundaries that
    follow image edges.

    The beta parameter (90) controls edge sensitivity: higher values make
    boundaries stick more closely to strong gradients.

    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image, ImageOps
    >>> from bpd_ui.core.roi_service import auto_chest_mask, refine_with_scribbles
    >>>
    >>> # Load image
    >>> img = Image.open("xray.jpg")
    >>> gray = np.array(ImageOps.grayscale(img))
    >>>
    >>> # Initial automatic mask
    >>> init_mask = auto_chest_mask(img)
    >>>
    >>> # User adds seed points inside region of interest
    >>> seeds = [(200, 250), (220, 260), (240, 270)]
    >>>
    >>> # Optional: User draws border to exclude regions
    >>> border = [(100, 100), (400, 100), (400, 400), (100, 400)]
    >>>
    >>> # Refine mask
    >>> refined = refine_with_scribbles(gray, seeds, border, init_mask)

    See Also
    --------
    auto_chest_mask : Automatic initial segmentation
    bpd_ui.ui.overlay_canvas : UI for drawing seeds/borders
    """
    H, W = gray.shape[:2]
    labels = np.zeros((H, W), dtype=np.uint8)
    for x, y in seeds_xy:
        if 0 <= x < W and 0 <= y < H:
            labels[int(y), int(x)] = 1
    if border_xy is not None and len(border_xy) > 0:
        from matplotlib.path import Path

        yy, xx = np.mgrid[:H, :W]
        inside = (
            Path(border_xy)
            .contains_points(np.vstack([xx.ravel(), yy.ravel()]).T)
            .reshape(H, W)
        )
        labels[~inside] = 2
    lab = random_walker(gray.astype(np.float32), labels, beta=90, mode="bf")
    m = lab == 1
    m = binary_closing(m, disk(3))
    m = binary_opening(m, disk(3))
    m = remove_small_objects(m, 128)
    m = remove_small_holes(m, 128)
    return m.astype(np.uint8) * 255
