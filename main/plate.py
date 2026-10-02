# -*- coding: utf-8 -*-
"""
Plate detection 

Finds plate on an image. Crop and mask image accordingly 
"""


import cv2
import numpy as np



MARGIN = 1.0


# ----------------------------------------------------------------------
# FLATTEN AND NORMALIZE
# ----------------------------------------------------------------------

def normalize_poly(img, order=1, method="subtract",
                   reference_range=(0, 255), fit_size=512):
    """
    Correct uneven illumination by fitting a polynomial background.

    The least-squares fit is performed on a small downsampled copy
    (fit_size px on the short side), then the resulting polynomial is
    evaluated directly on the full-resolution coordinate grid -- no large
    design matrix is ever allocated at full resolution.

    Parameters
    ----------
    img             : uint8 BGR or grayscale image.
    order           : Polynomial degree. 1 = flat tilt, 2 = curved surface.
    method          : "subtract" or "divide".
    reference_range : (min, max) clipping window before rescaling to [0, 255].
    fit_size        : Short-side resolution used for fitting (default 512).

    Returns
    -------
    corrected : uint8 grayscale image at original resolution.
    """
    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3
                else img).astype(np.float32)
    h, w = img_gray.shape

    scale = fit_size / min(h, w)
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    xs_n = np.linspace(0, 1, small_w, dtype=np.float32)
    ys_n = np.linspace(0, 1, small_h, dtype=np.float32)
    xs_n, ys_n = np.meshgrid(xs_n, ys_n)
    xs_n, ys_n = xs_n.ravel(), ys_n.ravel()

    ij = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    G = np.column_stack([xs_n ** i * ys_n ** j for i, j in ij]).astype(np.float32)
    m, _, _, _ = np.linalg.lstsq(G, small.ravel(), rcond=None)
    del G, small, xs_n, ys_n

    # -- Evaluate background via single matrix multiply --------------------
    xf = np.linspace(0, 1, w, dtype=np.float32)
    yf = np.linspace(0, 1, h, dtype=np.float32)

    Gx = np.column_stack([xf ** i for i, j in ij]).astype(np.float32)
    Gy = np.column_stack([yf ** j for i, j in ij]).astype(np.float32)
    Gx_weighted = Gx * m.astype(np.float32)[np.newaxis, :]
    background = Gy @ Gx_weighted.T
    del Gx, Gy, Gx_weighted

    if method == "subtract":
        corrected = img_gray - background
    elif method == "divide":
        corrected = img_gray / (background + 1e-3)
    else:
        raise ValueError("method must be 'subtract' or 'divide'.")

    del background, img_gray
    lo, hi = reference_range
    corrected = np.clip(corrected, lo, hi, out=corrected)
    corrected = ((corrected - lo) / (hi - lo) * 255).astype(np.uint8)
    return corrected


# ----------------------------------------------------------------------
# FIND PLATE
# ----------------------------------------------------------------------

def find_circle_mask(img, margin=MARGIN, detect_size=512):
    """
    Constraints exploited:
      - Plate diameter >= 2/3 of the image short side
      - Plate is roughly centered (center within 25% of short side)
      - Plate never partially outside the image

    Parameters
    ----------
    img         : uint8 grayscale image.
    margin      : Radius scale factor.
    detect_size : Short-side resolution for internal detection (default 512).

    Returns
    -------
    plate_mask : uint8 mask, 255 inside the circle, at original resolution.
    bbox       : (x1, y1, x2, y2) bounding box at original resolution.
    """
    h, w = img.shape[:2]

    # -- 1. Ensure grayscale ----------------------------------------------
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # -- 2. Downsample for speed ------------------------------------------
    scale = detect_size / min(h, w)
    small_w = int(w * scale)
    small_h = int(h * scale)
    gray_small = cv2.resize(gray_full, (small_w, small_h),
                            interpolation=cv2.INTER_AREA)

    # -- 3. CLAHE + blur + auto-Canny edges -------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_small)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)

    med = float(np.median(blurred))
    lo, hi = max(0.0, 0.5 * med), min(255.0, 1.5 * med)
    edges = cv2.Canny(blurred, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    short_side        = min(small_h, small_w)
    img_cx            = small_w / 2.0
    img_cy            = small_h / 2.0
    max_center_offset = short_side * 0.25
    r_min             = int(short_side * 0.33)
    r_max             = int(short_side * 0.50)

    def _best_hough(p2):
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=short_side * 0.5,
            param1=200,
            param2=p2,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is None:
            return None
        circles = np.round(circles[0]).astype(int)
        candidates = [
            c for c in circles
            if np.hypot(c[0] - img_cx, c[1] - img_cy) <= max_center_offset
        ]
        return max(candidates, key=lambda c: c[2]) if candidates else None

    # -- 4a. Hough with progressive thresholds ----------------------------
    result = None
    for p2 in (20, 12, 8):
        result = _best_hough(p2)
        if result is not None:
            break

    if result is not None:
        scx, scy, sr = result
    else:
        # -- 4b. Fallback: Otsu + largest contour -------------------------
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, binary, morph_k)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError(
                "Circle detection failed (no contours found). "
                "Check image quality or adjust margin."
            )
        largest = max(contours, key=cv2.contourArea)
        (scx, scy), sr = cv2.minEnclosingCircle(largest)
        scx, scy, sr = int(scx), int(scy), int(sr)

    # -- 5. Scale back to full resolution ---------------------------------
    cx = int(scx / scale)
    cy = int(scy / scale)
    r  = int((sr / scale) * margin)

    center = (cx, cy)
    plate_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(plate_mask, center, r, 255, -1)

    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, w)
    y2 = min(cy + r, h)
    bbox = (x1, y1, x2, y2)

    return plate_mask, bbox


def apply_circular_mask(target_img, mask, bbox):
    """Zero-out outside the circle mask, then crop to the bounding box."""
    masked_img = cv2.bitwise_and(target_img, target_img, mask=mask)
    x1, y1, x2, y2 = bbox
    cropped_img = masked_img[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    return cropped_img, cropped_mask


# ----------------------------------------------------------------------
# PROCESS IMAGE (PIPELINE)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# process_plate
# ----------------------------------------------------------------------

def process_plate(image, gradient_power=1, margin=None, detect_size=256):
    """
    Normalise illumination, detect the petri-dish circle, and crop to it.

    Parameters
    ----------
    image          : uint8 BGR image (e.g. plate.image).
    gradient_power : int   Polynomial order for background correction.
    margin         : float Radius scale factor (default: MARGIN).
    detect_size    : int   Short-side resolution for circle detection.

    Returns
    -------
    cropped   : uint8 BGR image cropped to the plate bounding box.
    crop_bbox : (x1, y1, x2, y2) in original image coordinates.
    """
    if image is None:
        raise ValueError("No image provided.")

    if margin is None:
        margin = MARGIN

    # -- Normalise illumination -------------------------------------------
    corrected = normalize_poly(
        image,
        order=gradient_power,
        method="subtract",
        reference_range=(0, 255),
    )

    # -- Detect circle and crop -------------------------------------------
    plate_mask, crop_bbox = find_circle_mask(
        corrected,
        margin=margin,
        detect_size=detect_size,
    )

    cropped, _ = apply_circular_mask(image, plate_mask, crop_bbox)

    # -- Free full-resolution intermediates -------------------------------
    del corrected, plate_mask

    return cropped, crop_bbox