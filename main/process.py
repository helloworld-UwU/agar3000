# -*- coding: utf-8 -*-
"""
Plate processing pipeline.

Plate.find_circle_mask() uses a fast, parameter-free detection strategy:
  - Runs on a small downsampled copy for speed
  - CLAHE + auto-Canny edges for robustness in low contrast
  - Hough Circle Transform with tight bounds (plate >= 2/3 short side, centered)
  - Progressive accumulator threshold with Otsu+contour fallback
"""

import gc
import os
import cv2
import numpy as np


class Plate:

    # ------------------------------------------------------------------
    # Class-level constants (still here for show_config / legacy use)
    # ------------------------------------------------------------------
    MARGIN             = 1.0
    N_STRIPES          = 5
    COLONIES_THRESHOLD = 128   # used externally; detection no longer needs it

    def __init__(self, image_path, sample_id=None, date=None):
        self.image_path = image_path
        self.date       = date

        if sample_id is None:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            if fname.startswith("ori_"):
                fname = fname[len("ori_"):]
            sample_id = fname
        self.sample_id = sample_id

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Placeholders populated by the preprocessing pipeline
        self.corrected        = None
        self.plate_mask       = None
        self.bbox             = None
        self.cropped          = None
        self.cropped_mask     = None
        self.cropped_corrected = None
        self.colonies_mask    = None
        self.smooth           = None
        self.vertical_splits  = None
        self.horizontal_splits = None

    # ------------------------------------------------------------------
    # FIND PLATE
    # ------------------------------------------------------------------

    
    def find_circle_mask(self, img, margin=None, detect_size=512, **kwargs):
        """
        Robust, fast petri-dish circle detection.

        Replaces the old threshold/alpha/beta approach with a parameter-free
        pipeline. Legacy keyword arguments (threshold, alpha, beta) are
        accepted but silently ignored so existing call sites keep working.

        Constraints exploited:
          - Plate diameter >= 2/3 of the image short side
          - Plate is roughly centered (center within 25% of short side)
          - Plate never partially outside the image

        Strategy
        --------
        1. Downsample to detect_size px on the short side for speed.
        2. CLAHE + auto-Canny edges: detects the physical rim even in
           low contrast without any manual threshold.
        3. Hough Circle Transform with tight radius/center bounds.
           Tried at three progressively permissive accumulator thresholds
           (20, 12, 8) -- stops at the first that yields a valid circle.
        4. Fallback: Otsu binarisation + largest contour if Hough fails.
        5. Scale result back to original resolution.

        Parameters
        ----------
        img         : uint8 grayscale image.
        margin      : Radius scale factor (default: self.MARGIN).
        detect_size : Short-side resolution for internal detection (default 512).

        Returns
        -------
        plate_mask : uint8 mask, 255 inside the circle, at original resolution.
        bbox       : (x1, y1, x2, y2) bounding box at original resolution.
        """
        if margin is None:
            margin = self.MARGIN

        h, w = img.shape[:2]

        # -- 1. Ensure grayscale ----------------------------------------------
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # -- 2. Downsample for speed ------------------------------------------
        scale      = detect_size / min(h, w)
        small_w    = int(w * scale)
        small_h    = int(h * scale)
        gray_small = cv2.resize(gray_full, (small_w, small_h),
                                interpolation=cv2.INTER_AREA)

        # -- 3. CLAHE + blur + auto-Canny edges -------------------------------
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray_small)
        blurred  = cv2.GaussianBlur(enhanced, (5, 5), 1)

        med    = float(np.median(blurred))
        lo, hi = max(0.0, 0.5 * med), min(255.0, 1.5 * med)
        edges  = cv2.Canny(blurred, lo, hi)

        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
            circles    = np.round(circles[0]).astype(int)
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
            binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, binary, morph_k)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise RuntimeError(
                    "Circle detection failed (no contours found). "
                    "Check image quality or adjust margin."
                )
            largest        = max(contours, key=cv2.contourArea)
            (scx, scy), sr = cv2.minEnclosingCircle(largest)
            scx, scy, sr   = int(scx), int(scy), int(sr)

        # -- 5. Scale back to full resolution ---------------------------------
        cx = int(scx / scale)
        cy = int(scy / scale)
        r  = int((sr  / scale) * margin)

        center     = (cx, cy)
        plate_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(plate_mask, center, r, 255, -1)

        x1   = max(cx - r, 0)
        y1   = max(cy - r, 0)
        x2   = min(cx + r, w)
        y2   = min(cy + r, h)
        bbox = (x1, y1, x2, y2)

        return plate_mask, bbox

    def apply_circular_mask(self, ref_image, target_img, mask, bbox):
        """Zero-out outside the circle mask, then crop to the bounding box."""
        masked_img   = cv2.bitwise_and(target_img, target_img, mask=mask)
        x1, y1, x2, y2 = bbox
        cropped_img  = masked_img[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        return cropped_img, cropped_mask

    # ------------------------------------------------------------------
    # FLATTEN AND NORMALIZE
    # ------------------------------------------------------------------
    
    def normalize_poly(self, img=None, order=1, method="subtract",
                       reference_range=(0, 255), fit_size=512):
        """
        Correct uneven illumination by fitting a polynomial background.

        Memory-efficient: the least-squares fit is performed on a small
        downsampled copy (fit_size px on the short side), then the resulting
        polynomial is evaluated directly on the full-resolution coordinate
        grid -- no large design matrix is ever allocated at full resolution.

        For a 5 MP image (2500x2000) with order=1 (3 terms), the old approach
        built a 5 000 000 x 3 float64 matrix (~114 MB). This version builds a
        262 144 x 3 matrix (~6 MB) instead.

        Parameters
        ----------
        img             : uint8 BGR or grayscale image (default: self.image).
        order           : Polynomial degree. 1 = flat tilt, 2 = curved surface.
        method          : "subtract" or "divide".
        reference_range : (min, max) clipping window before rescaling to [0, 255].
        fit_size        : Short-side resolution used for fitting (default 512).

        Returns
        -------
        corrected : uint8 grayscale image at original resolution.
        """
        if img is None:
            img = self.image
    
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
        G  = np.column_stack([xs_n**i * ys_n**j for i, j in ij]).astype(np.float32)
        m, _, _, _ = np.linalg.lstsq(G, small.ravel(), rcond=None)
        del G, small, xs_n, ys_n
    
        # -- Evaluate background via single matrix multiply -----------------------
        xf = np.linspace(0, 1, w, dtype=np.float32)
        yf = np.linspace(0, 1, h, dtype=np.float32)
    
        Gx = np.column_stack([xf**i for i, j in ij]).astype(np.float32)  # (w, n_terms)
        Gy = np.column_stack([yf**j for i, j in ij]).astype(np.float32)  # (h, n_terms)
        Gx_weighted = Gx * m.astype(np.float32)[np.newaxis, :]            # (w, n_terms)
        background  = Gy @ Gx_weighted.T                                   # (h, w)
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

    # ------------------------------------------------------------------
    # SMOOTH BACKGROUND
    # ------------------------------------------------------------------



    def smooth_background(self, blur_ksize=15):
        """
        Smooth the background of the cropped plate image using the colonies
        mask, preserving colonies exactly.
        """
        if self.cropped is None or self.colonies_mask is None:
            raise ValueError("Cropped image or colonies mask not found.")

        color_cropped = (
            cv2.cvtColor(self.cropped, cv2.COLOR_GRAY2BGR)
            if len(self.cropped.shape) == 2
            else self.cropped.copy()
        )
        background_mask = cv2.bitwise_not(self.colonies_mask)
        return self._masked_median_blur(color_cropped, background_mask, blur_ksize)

    def enhance_contrast(self, image, clip_limit=2.0, tile_grid_size=(64, 64)):
        """
        Enhance contrast with CLAHE applied to the L channel (LAB colour space).

        Parameters
        ----------
        image          : uint8 BGR image.
        clip_limit     : CLAHE clip limit.
        tile_grid_size : CLAHE tile grid size.

        Returns
        -------
        enhanced_img : uint8 BGR image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if image.shape[-1] == 3 else image
        l, a, b = cv2.split(lab)
        clahe       = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced  = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # COLONY THRESHOLDING AND SPLITTING
    # ------------------------------------------------------------------

    def _find_colony_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _line_intersects_contours(self, line_pos, contours, axis='horizontal', margin=10):
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if axis == 'horizontal' and (y - margin) < line_pos < (y + h + margin):
                return True
            if axis == 'vertical'   and (x - margin) < line_pos < (x + w + margin):
                return True
        return False

    def _find_multiple_splits(self, length, contours, n_stripes=5, margin=5,
                               search_range=500, axis='vertical', min_distance=50):
        splits = []
        stripe_width = length // n_stripes

        for i in range(1, n_stripes):
            target_center  = i * stripe_width
            best_candidate = None
            for offset in range(search_range):
                for direction in (-1, 1):
                    candidate = target_center + direction * offset
                    if candidate <= 0 or candidate >= length:
                        continue
                    if (not self._line_intersects_contours(
                            candidate, contours, axis=axis, margin=margin)
                            and all(abs(candidate - s) >= min_distance for s in splits)):
                        best_candidate = candidate
                        break
                if best_candidate is not None:
                    break

            if best_candidate is not None:
                splits.append(best_candidate)
            else:
                fallback = target_center
                while (any(abs(fallback - s) < min_distance for s in splits)
                       and fallback < length):
                    fallback += min_distance
                splits.append(min(fallback, length - 1))

        return sorted(splits)

    def split(self, n_stripes=None):
        """
        Compute vertical and horizontal splits of the plate.

        Populates self.vertical_splits, self.horizontal_splits, self.split_images.

        Returns
        -------
        list of (top_half, bottom_half) arrays, one per stripe.
        """
        if n_stripes is None:
            n_stripes = self.N_STRIPES

        if self.colonies_mask is None:
            raise RuntimeError("colonies_mask not found. Run preprocessing first.")

        contours_ref    = self._find_colony_contours(self.colonies_mask)
        height, width   = self.colonies_mask.shape
        split_lines     = self._find_multiple_splits(width, contours_ref,
                                                     n_stripes=n_stripes)
        all_xs          = [0] + split_lines + [width]

        vertical_splits  = []
        horizontal_splits = []
        split_images     = []

        for i in range(n_stripes):
            x_start, x_end = all_xs[i], all_xs[i + 1]
            stripe_img = (
                self.smooth[:, x_start:x_end]
                if getattr(self, "hc", False)
                else self.cropped[:, x_start:x_end]
            )
            stripe_mask     = self.colonies_mask[:, x_start:x_end]
            stripe_contours = self._find_colony_contours(stripe_mask)

            h_split = self._find_multiple_splits(
                stripe_img.shape[0],
                stripe_contours,
                margin=5, search_range=200, n_stripes=2, axis="horizontal",
            )[0]

            vertical_splits.append((x_start, x_end))
            horizontal_splits.append(h_split)
            split_images.append((stripe_img[:h_split, :], stripe_img[h_split:, :]))

        self.vertical_splits   = vertical_splits
        self.horizontal_splits = horizontal_splits
        self.split_images      = split_images
        return split_images



    # ------------------------------------------------------------------
    # DISPLAY / SAVE / UTILITIES
    # ------------------------------------------------------------------



    def save(self, output_dir, img=None, ext=".jpg"):
        """Save an image, auto-labelling from the attribute name where possible."""
        if img is None:
            img_to_save = self.image
            label       = "image"
        else:
            label = next(
                (name for name, val in self.__dict__.items()
                 if isinstance(val, np.ndarray) and val is img),
                "img",
            )
            img_to_save = img

        os.makedirs(output_dir, exist_ok=True)
        base_name   = self.sample_id or "plate"
        output_path = os.path.join(output_dir, f"{label}_{base_name}{ext}")

        if not cv2.imwrite(output_path, img_to_save):
            raise ValueError(f"Could not save image to {output_path}")
        print(f"Image saved to {output_path}")

    def get_size(self):
        """Return (width, height) of the original image."""
        return self.image.shape[1], self.image.shape[0]

    @classmethod
    def show_config(cls):
        print("Plate Configurations:")
        print(f"  MARGIN             = {cls.MARGIN}")
        print(f"  COLONIES_THRESHOLD = {cls.COLONIES_THRESHOLD}")
        print(f"  N_STRIPES          = {cls.N_STRIPES}")


# ----------------------------------------------------------------------
# process_plate
# ----------------------------------------------------------------------




def process_plate(
    image_path,
    gradient_power=1,
    margin=None,
    detect_size=256,
    timer=None,
    img_name=None,
):
    """
    Load a plate image, normalise illumination (unless tra mode), detect the
    petri-dish circle, and crop to it.

    Parameters
    ----------
    image_path    : str  Path to the image file.
    tra           : bool If True, skip polynomial normalisation (use raw gray).
    gradient_power: int  Polynomial order for background correction (default 1).
    margin        : float Radius scale factor (default: Plate.MARGIN).
    detect_size   : int  Short-side resolution for circle detection (default 512).

    Returns
    -------
    plate : Plate object with .corrected, .plate_mask, .bbox, .cropped,
            .cropped_mask populated.
    """
    if timer is not None:
        timer.start(f"[{img_name or os.path.basename(image_path)}] Image load")
    
    plate = Plate(image_path)
    
    if timer is not None:
        timer.start(f"[{img_name or os.path.basename(image_path)}] Plate pre-processing")

    if margin is None:
        margin = plate.MARGIN

    # -- Normalise illumination -----------------------------------------------
    plate.corrected = plate.normalize_poly(
        order=gradient_power,
        method="subtract",
        reference_range=(0, 255),
    )

    # -- Detect circle and crop -----------------------------------------------
    plate.plate_mask, plate.bbox = plate.find_circle_mask(
        plate.corrected,
        margin=margin,
        detect_size=detect_size,
    )

    plate.cropped, plate.cropped_mask = plate.apply_circular_mask(
        plate.corrected,
        plate.image,
        plate.plate_mask,
        plate.bbox,
    )

    # -- Free full-resolution intermediates that are no longer needed ---------
    # plate_mask is full-resolution; keep only the cropped version
    plate.plate_mask = None
    gc.collect()

    return plate