# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:43:51 2025

@author: Admin
"""
# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import pickle
import inspect
import shutil

# %%
class Plate:
    # Constant configurations
    GRADIENT_POWER = 1
    
    MARGIN = 0.75
    PLATE_THRESHOLD = 100
    COLONIES_THRESHOLD = 40
    ALPHA = 1.4
    BETA = 50
    N_STRIPES = 5

    def __init__(self, image_path, tra=False, hc = False, bb=False, sample_id=None, date=None):
        
        if tra:
            self.PLATE_THRESHOLD = 200
            self.COLONIES_THRESHOLD = 35
        
        
        
        self.image_path = image_path
        self.date = date
        self.bb = bb
        
        # Use filename as sample_id if not provided
        if sample_id is None:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            if fname.startswith("ori_"):
                fname = fname[len("ori_"):]  # remove the prefix
            sample_id = fname
        self.sample_id = sample_id
        
        # Load image using cv2 (as BGR)
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Convert to RGB for displaying with matplotlib
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Compute corrected image and background during initialization
        if tra:
            self.corrected = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.corrected = self._normalize_poly(
                order=self.GRADIENT_POWER, method="subtract", reference_range=(0, 255)
            )
    
        
        # Compute circular mask, bbox, cropped images
        self.plate_mask, self.bbox = self._find_circle_mask(self.corrected,
                                                            threshold=self.PLATE_THRESHOLD,
                                                            margin=self.MARGIN,
                                                            alpha=self.ALPHA,
                                                            beta=self.BETA)
        
        self.cropped, self.cropped_mask = self._apply_circular_mask(self.corrected,
                                                                    self.image,
                                                                    self.plate_mask,
                                                                    self.bbox)
        
        self.cropped_corrected, _ = self._apply_circular_mask(self.corrected,
                                                             self.corrected,
                                                             self.plate_mask,
                                                             self.bbox)
        
        
        self.cropped_corrected = self._normalize_poly(
            self.cropped_corrected, order=self.GRADIENT_POWER, method="subtract", reference_range=(0, 255)
            )
        
        # Threshhold mask > background smoothing
        _, self.colonies_mask = cv2.threshold(self.cropped_corrected, self.COLONIES_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        
        if tra:
            if hc:
                self.smooth = self._enhance_contrast(self.cropped)
            else:
                self.smooth = self.cropped
        
        else:
            kernel = np.ones((3,3), np.uint8)
            self.colonies_mask = cv2.morphologyEx(self.colonies_mask, cv2.MORPH_OPEN, kernel)
            self.colonies_mask = cv2.morphologyEx(self.colonies_mask, cv2.MORPH_CLOSE, kernel)
            
            self.smooth = self._smooth_background(blur_ksize=15)
        
        #Spliting
        self._split()
      
        
    #-----------------------------------------------------
    # FIND PLATE
    #-----------------------------------------------------
    
    # ===== Internal: Adjust contrast =====
    def _cont_bright(self, image, alpha=None, beta=None):
        if alpha is None:
            alpha = self.ALPHA
        if beta is None:
            beta = self.BETA
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # ===== Internal: Find circular mask =====
    def _find_circle_mask(self, img, threshold=None, margin=None, alpha=None, beta=None):
        if threshold is None:
            threshold = self.PLATE_THRESHOLD
        if margin is None:
            margin = self.MARGIN
        cont_img = self._cont_bright(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), alpha, beta)
        gray = cv2.cvtColor(cont_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius * margin)
        plate_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(plate_mask, center, radius, 255, -1)
        x1 = max(center[0] - radius, 0)
        y1 = max(center[1] - radius, 0)
        x2 = min(center[0] + radius, img.shape[1])
        y2 = min(center[1] + radius, img.shape[0])
        bbox = (x1, y1, x2, y2)
        return plate_mask, bbox

    # ===== Internal: Apply circular mask =====
    def _apply_circular_mask(self, ref_image, target_img, mask, bbox):
        masked_img = cv2.bitwise_and(target_img, target_img, mask=mask)
        x1, y1, x2, y2 = bbox
        cropped_img = masked_img[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        return cropped_img, cropped_mask
    
    
    #-----------------------------------------------------
    # ===== Internal: Flatten and normalize =====
    #-----------------------------------------------------
    
    def _normalize_poly(self, img=None, order=1, method="subtract", reference_range=(0, 255)):
        if img is None:
            img = self.image  # default to original image
    
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            img_gray = img.astype(float)
    
        h, w = img_gray.shape
        y, x = np.mgrid[0:h, 0:w]
    
        # Fit polynomial background
        ij = [(i, j) for i in range(order+1) for j in range(order+1-i)]
        G = np.zeros((x.size, len(ij)))
        for k, (i,j) in enumerate(ij):
            G[:, k] = x.ravel()**i * y.ravel()**j
        m, _, _, _ = np.linalg.lstsq(G, img_gray.ravel(), rcond=None)
        background = sum(a*(x**i)*(y**j) for a,(i,j) in zip(m, ij))
    
        # Correct illumination
        if method == "subtract":
            corrected = img_gray - background
        elif method == "divide":
            corrected = img_gray / (background + 1e-3)
        else:
            raise ValueError("Method must be 'subtract' or 'divide'.")
    
        # Clip and normalize
        min_val, max_val = reference_range
        corrected = np.clip(corrected, min_val, max_val)
        corrected = ((corrected - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
        return corrected
    
    #-----------------------------------------------------
    # ===== Smooth_background =====
    #-----------------------------------------------------
    
    def _masked_median_blur(self, image, mask, ksize=5):
        """
        Apply median blur only on regions defined by mask, ignoring colonies.
        """
        # True for background pixels, False for colonies
        background_mask = mask > 0

        blurred = np.zeros_like(image)
        for c in range(3):  # process each color channel separately
            blurred_channel = median_filter(image[:, :, c], size=ksize)
            # Use blurred values only where background_mask is True
            blurred[:, :, c] = np.where(background_mask, blurred_channel, image[:, :, c])
        return blurred

    def _smooth_background(self, blur_ksize=15):
        """
        Smooth the background of the cropped plate image using the colonies mask,
        preserving colonies exactly as they are.
        """
        if self.cropped is None or self.colonies_mask is None:
            raise ValueError("Cropped image or colonies mask not found.")

        # Ensure 3-channel color image
        if len(self.cropped.shape) == 2:
            color_cropped = cv2.cvtColor(self.cropped, cv2.COLOR_GRAY2BGR)
        else:
            color_cropped = self.cropped.copy()

        # Invert colonies mask (background = 255, colonies = 0)
        background_mask = cv2.bitwise_not(self.colonies_mask)

        # Apply masked median blur
        smoothed_img = self._masked_median_blur(color_cropped, background_mask, blur_ksize)
        return smoothed_img

    def _enhance_contrast(self, image, clip_limit=2.0, tile_grid_size=(64, 64)):
        """
        Increase contrast of an image using CLAHE.
        
        Args:
            image (np.ndarray): Input image (BGR or RGB).
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (tuple): Size of the grid for CLAHE.
        
        Returns:
            np.ndarray: Contrast-enhanced image (same shape as input).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if image.shape[-1] == 3 else image
        l, a, b = cv2.split(lab)
    
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l)
    
        # Merge back
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
        return enhanced_img
    
    
    #-----------------------------------------------------
    # ===== Colony thresholding & splitting =====
    #-----------------------------------------------------
    
    
    def _find_colony_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    

    def _line_intersects_contours(self, line_pos, contours, axis='horizontal', margin=10):
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if axis == 'horizontal' and (y - margin) < line_pos < (y + h + margin):
                return True
            if axis == 'vertical' and (x - margin) < line_pos < (x + w + margin):
                return True
        return False

    def _find_multiple_splits(self, length, contours, n_stripes=5, margin=5, search_range=500,
                              axis='vertical', min_distance=50):
        splits = []
        stripe_width = length // n_stripes

        for i in range(1, n_stripes):
            target_center = i * stripe_width
            best_candidate = None
            for offset in range(search_range):
                for direction in [-1, 1]:
                    candidate = target_center + direction * offset
                    if candidate <= 0 or candidate >= length:
                        continue
                    if (not self._line_intersects_contours(candidate, contours, axis=axis, margin=margin)
                        and all(abs(candidate - s) >= min_distance for s in splits)):
                        best_candidate = candidate
                        break
                if best_candidate is not None:
                    break
            if best_candidate is not None:
                splits.append(best_candidate)
            else:
                fallback = target_center
                while any(abs(fallback - s) < min_distance for s in splits) and fallback < length:
                    fallback += min_distance
                splits.append(min(fallback, length - 1))

        return sorted(splits)
    def _split(self, colony_threshold=None, n_stripes=None, output_dir=None):
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.image_path), "splits")
        os.makedirs(output_dir, exist_ok=True)
        
        if n_stripes is None:
            n_stripes = self.N_STRIPES
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.image_path), "splits")
        
        contours_ref = self._find_colony_contours(self.colonies_mask)
        height, width = self.colonies_mask.shape
        
        # Vertical splits
        split_lines = self._find_multiple_splits(width, contours_ref, n_stripes=n_stripes)
        all_xs = [0] + split_lines + [width]

        vertical_splits = []
        horizontal_splits = []

        base_name = self.sample_id if self.sample_id is not None else "plate"

        # Apply splits to cropped target image
        for i in range(n_stripes):
            x_start, x_end = all_xs[i], all_xs[i + 1]
            
            if self.bb:
                stripe_img = self.smooth[:, x_start:x_end]
            else:
                stripe_img = self.cropped[:, x_start:x_end]
            stripe_mask = self.colonies_mask[:, x_start:x_end]

            stripe_contours = self._find_colony_contours(stripe_mask)
            h_split = self._find_multiple_splits(
                stripe_img.shape[0], stripe_contours,
                margin=5, search_range=200,
                n_stripes=2, axis='horizontal'
            )[0]

            top_half = stripe_img[:h_split, :]
            bottom_half = stripe_img[h_split:, :]

            # Save split images
            top_name = f"{base_name}_stripe_{i+1}_top.jpg"
            bottom_name = f"{base_name}_stripe_{i+1}_bottom.jpg"
            cv2.imwrite(os.path.join(output_dir, top_name), top_half)
            cv2.imwrite(os.path.join(output_dir, bottom_name), bottom_half)

            vertical_splits.append((x_start, x_end))
            horizontal_splits.append(h_split)

        # Save splits coordinates in a pickle file
        splits_data = {
            "vertical_splits": vertical_splits,
            "horizontal_splits": horizontal_splits
        }
        pickle_path = os.path.join(output_dir, f"{base_name}_splits.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(splits_data, f)
        print(f"Split coordinates saved in '{pickle_path}'")
    
        self.vertical_splits = vertical_splits
        self.horizontal_splits = horizontal_splits
        return vertical_splits, horizontal_splits
        
    
    #-----------------------------------------------------
    # ===== ELSE =====
    #-----------------------------------------------------

    # ===== Display image =====
    def show(self, img=None, title=None):
        """
        Display an image.
        
        Parameters:
            img (np.ndarray, optional): Image to show. Defaults to self.image_rgb.
            title (str, optional): Title for the plot.
        """
        if img is None:
            img_to_show = self.image_rgb
        else:
            # Convert grayscale to RGB for consistent display
            if len(img.shape) == 2:
                img_to_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_to_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img_to_show)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()
    
    # ===== Save image =====
    def save(self, output_dir, img=None, ext=".jpg"):
        """
        Save an image with automatic label based on attribute name if possible.
        Assumes images are stored in BGR (OpenCV convention).
        """
        if img is None:
            img_to_save = self.image
            label = "image"
        else:
            # Try to find attribute name
            label = None
            for name, val in self.__dict__.items():
                if isinstance(val, np.ndarray) and val is img:
                    label = name
                    break
            if label is None:
                label = "img"
    
            img_to_save = img  # already BGR if from cv2 pipeline
    
        os.makedirs(output_dir, exist_ok=True)
        base_name = self.sample_id if self.sample_id else "plate"
        filename = f"{label}_{base_name}{ext}"
        output_path = os.path.join(output_dir, filename)
    
        success = cv2.imwrite(output_path, img_to_save)
        if not success:
            raise ValueError(f"Could not save image to {output_path}")
        print(f"Image saved to {output_path}")


    # ===== Get image size =====
    def get_size(self):
        return self.image.shape[1], self.image.shape[0]  # width, height

    # ===== Display configurations =====
    @classmethod
    def show_config(cls):
        print("Plate Configurations (constants):")
        print(f"  MARGIN = {cls.MARGIN}")
        print(f"  PLATE_THRESHOLD = {cls.PLATE_THRESHOLD}")
        print(f"  COLONIES_THRESHOLD = {cls.COLONIES_THRESHOLD}")
        print(f"  ALPHA = {cls.ALPHA}")
        print(f"  BETA = {cls.BETA}")
        print(f"  N_STRIPES = {cls.N_STRIPES}")
    
    
   
    
# %%
def pre_all(root_folder, tra=False, hc=False, bb=False, image_extensions=None, prefix="ori_"):
    """
    Walk through the folder tree starting at `root_folder` and process all images
    that start with a specific prefix (default 'ori_').

    Saves outputs to a separate 'processed' subfolder to avoid reprocessing saved images.
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

    # Collect all original images first
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.startswith(prefix) and any(fname.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(dirpath, fname))

    # Process each image
    for image_path in image_paths:
        dirpath = os.path.dirname(image_path)
        output_dir = dirpath
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Initialize Plate
            plate = Plate(image_path, tra=tra, hc=hc, bb=bb)

            # Save outputs to the separate folder
            plate.save(output_dir, plate.image, ext=".jpg")
            plate.save(output_dir, plate.corrected, ext=".jpg")
            plate.save(output_dir, plate.cropped, ext=".jpg")
            plate.save(output_dir, plate.cropped_corrected, ext=".jpg")
            plate.save(output_dir, plate.colonies_mask, ext=".jpg")
            plate.save(output_dir, plate.smooth, ext=".jpg")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")





def organize_photos(input_folder, output_folder=None):
    # Set default output folder if not provided
    if output_folder is None:
        output_folder = os.path.join(input_folder, "results")
    
    os.makedirs(output_folder, exist_ok=True)

    # Define supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    for filename in os.listdir(input_folder):
        full_path = os.path.join(input_folder, filename)
        if os.path.isfile(full_path) and filename.lower().endswith(image_extensions):
            name_without_ext, ext = os.path.splitext(filename)
            subfolder = os.path.join(output_folder, name_without_ext)
            os.makedirs(subfolder, exist_ok=True)

            # Prepare new file name with prefix
            new_filename = f"ori_{filename}"
            destination_path = os.path.join(subfolder, new_filename)
            
            # Copy the image
            shutil.copy2(full_path, destination_path)
            
    print(f"Folder structure created: {output_folder}")