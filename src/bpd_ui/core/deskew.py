"""
Scan Deskewing and Rectification
=================================

This module provides automatic perspective correction for scanned X-ray films.
It detects the film boundaries using edge detection and Canny/Hough transforms,
then applies a perspective warp to rectify the image.

Functions
---------
_largest_quad
    Find the largest quadrilateral contour
rectify_if_scan
    Detect and rectify scanned films with perspective distortion

Notes
-----
This module is designed for X-ray films that were photographed or scanned
with perspective distortion (e.g., handheld camera, tilted scanner). It:

1. Detects edges using Canny
2. Finds contours
3. Identifies largest quadrilateral (film boundary)
4. Computes perspective transform to rectangle
5. Warps image to frontal view
6. Rotates to portrait if needed

If no quadrilateral is detected, returns the original image unchanged.

See Also
--------
bpd_ui.ui.preprocess_tab : Uses rectification before ROI extraction
"""

import cv2
import numpy as np


def _largest_quad(cnts):
    """
    Find the largest quadrilateral contour from a list of contours.

    This function approximates each contour as a polygon and selects the
    largest one with exactly 4 vertices (a quadrilateral).

    Parameters
    ----------
    cnts : list
        List of contours from cv2.findContours()

    Returns
    -------
    np.ndarray or None
        Vertices of the largest quadrilateral as (4, 1, 2) array, or None
        if no quadrilateral is found

    Notes
    -----
    The function uses Douglas-Peucker algorithm (cv2.approxPolyDP) with
    epsilon = 2% of perimeter to approximate contours as polygons.

    Only contours that simplify to exactly 4 vertices are considered.
    Among these, the one with largest area (cv2.contourArea) is selected.

    Examples
    --------
    >>> import cv2
    >>> import numpy as np
    >>>
    >>> # Simulate contours
    >>> quad = np.array([[[100, 100]], [[400, 100]],
    ...                  [[400, 400]], [[100, 400]]])
    >>> triangle = np.array([[[50, 50]], [[150, 50]], [[100, 150]]])
    >>> cnts = [quad, triangle]
    >>>
    >>> best = _largest_quad(cnts)
    >>> print(best.shape)
    (4, 1, 2)

    See Also
    --------
    rectify_if_scan : Uses this to find film boundary
    """
    best = None
    best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best, best_area = approx, area
    return best


def rectify_if_scan(bgr):
    """
    Detect and rectify scanned X-ray films with perspective distortion.

    This function automatically detects the boundaries of a scanned/photographed
    X-ray film, applies perspective correction to create a frontal view, and
    rotates to portrait orientation if needed.

    Parameters
    ----------
    bgr : np.ndarray
        Input image in BGR format (H, W, 3) from OpenCV

    Returns
    -------
    np.ndarray
        Rectified image in BGR format. Returns original if no quadrilateral
        is detected.

    Notes
    -----
    Processing pipeline:
    1. Convert to grayscale
    2. Apply Gaussian blur (5x5)
    3. Detect edges with Canny (thresholds 50, 150)
    4. Find external contours
    5. Identify largest quadrilateral
    6. Order vertices (top-left, top-right, bottom-right, bottom-left)
    7. Compute target rectangle dimensions
    8. Apply perspective transform
    9. Rotate 90° clockwise if width > height

    The vertex ordering uses:
    - Smallest sum (x+y) = top-left
    - Smallest diff (y-x) = top-right
    - Largest sum (x+y) = bottom-right
    - Largest diff (y-x) = bottom-left

    The target dimensions are the maximum of measured edge lengths to avoid
    distortion.

    Examples
    --------
    >>> import cv2
    >>> import numpy as np
    >>> from bpd_ui.core.deskew import rectify_if_scan
    >>>
    >>> # Load scanned film with perspective distortion
    >>> img = cv2.imread("scanned_xray.jpg")
    >>> print(img.shape)
    (1200, 1600, 3)
    >>>
    >>> # Rectify
    >>> rectified = rectify_if_scan(img)
    >>> print(rectified.shape)  # Dimensions may change
    (1024, 768, 3)
    >>>
    >>> # Save result
    >>> cv2.imwrite("rectified_xray.jpg", rectified)
    >>>
    >>> # If no quad detected, returns original
    >>> plain_img = cv2.imread("already_rectified.jpg")
    >>> result = rectify_if_scan(plain_img)
    >>> assert result is plain_img  # Same object

    See Also
    --------
    _largest_quad : Finds quadrilateral boundary
    bpd_ui.ui.preprocess_tab : Uses this in preprocessing workflow
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = _largest_quad(cnts)
    if quad is None:
        return bgr
    pts = quad.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect = np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )
    w1 = np.linalg.norm(rect[2] - rect[3])
    w2 = np.linalg.norm(rect[1] - rect[0])
    h1 = np.linalg.norm(rect[1] - rect[2])
    h2 = np.linalg.norm(rect[0] - rect[3])
    W = int(max(w1, w2))
    H = int(max(h1, h2))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(bgr, M, (W, H))
    if W > H:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    return warp
