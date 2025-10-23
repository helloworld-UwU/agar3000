# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 05:16:20 2025

@author: Admin
"""
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
# %%
# COLOR OF AGAR
def contBright(image):   
    alpha = 1.4 #[1.0-3.0]
    beta = 50 #[0 - 100]
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_image

def findCircle(img, threshold = 251):
    '''
    return: center (x , y), radius
    '''
    # Read in the image
    image = img.copy()
    image = contBright(image)
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Threshold the image to create a binary image
    image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    # Find the contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    (x,y),radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x),int(y))
    radius = int(radius)
    return center, radius

def applyCircularMask(img, margin=0.75, threshold = 251):
    center, radius = findCircle(img, threshold)
    radius = int(radius*margin)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create a black mask with the same size as the image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw a white filled circle on the mask
    cv2.circle(mask, center, radius, 255, -1)


    return mask

# %%

def get_mean_bgr_in_mask(image_path, thresh_val=170):
    """
    Applies a circular + threshold-based mask to an image and computes
    the mean BGR color within the resulting mask.

    Args:
        image_path (str): Path to the input image.
        thresh_val (int): Threshold value for colony segmentation.

    Returns:
        list: Mean [B, G, R] values inside the mask.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    # Create masks
    mask1 = cv2.bitwise_not(applyCircularMask(img))
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(grey_img, 170, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_not(cv2.bitwise_or(mask1, mask2))

    # Compute mean color inside mask
    mask_bool = combined_mask.astype(bool)
    mean_color = [img[:, :, i][mask_bool].mean() for i in range(3)]  # B, G, R

    return mean_color

# %%

def plot_colony_blobs(
    root_folder,
    figsize=(10, 10),
    scale=1,
    alpha=1,
    agar_color=None,
    min_score=0
):
    """
    For each '_colonies.csv' file in the directory tree, finds the corresponding
    'cropped_*.jpg' image in the same folder, extracts image shape, and plots
    colonies. If agar_color is not provided, it is estimated from an 'ori_*.jpg'
    image in the same folder using get_mean_bgr_in_mask().
    """
    for dirpath, _, filenames in os.walk(root_folder):
        csv_files = [f for f in filenames if f.endswith('_results.csv')]
        for csv_filename in csv_files:
            csv_path = os.path.join(dirpath, csv_filename)

            # Find matching cropped_*.jpg
            cropped_image = next(
                (os.path.join(dirpath, f) for f in filenames
                 if f.lower().startswith('cropped_') and f.lower().endswith('.jpg')),
                None
            )
            if not cropped_image:
                print(f"Skipped {csv_filename}: No matching 'cropped_*.jpg' in {dirpath}")
                continue

            img = cv2.imread(cropped_image)
            if img is None:
                print(f"Skipped {csv_filename}: Failed to load image {cropped_image}")
                continue
            height, width = img.shape[:2]

            # If agar_color not given, extract it from ori_*.jpg using masking
            local_agar_color = agar_color
            if agar_color is None:
                ori_image = next(
                    (os.path.join(dirpath, f) for f in filenames
                     if f.lower().startswith('ori_') and f.lower().endswith('.jpg')),
                    None
                )
                if ori_image:
                    try:
                        local_agar_color = get_mean_bgr_in_mask(ori_image)
                        print(f"Agar color estimated from {ori_image}: {local_agar_color}")
                    except Exception as e:
                        print(f"Warning: Could not extract agar color from {ori_image}: {e}")
                        local_agar_color = (230, 230, 250)  # Fallback
                else:
                    print(f"Warning: No ori_*.jpg found for {csv_filename}. Using default agar color.")
                    local_agar_color = (230, 230, 250)

            try:
                df = pd.read_csv(csv_path)
                required_cols = {'X', 'Y', 'Area', 'R', 'G', 'B', 'Score'}
                if not required_cols.issubset(df.columns):
                    print(f"Skipped {csv_filename}: Missing required columns.")
                    continue

                df = df[df['Score'] >= min_score]

                x_center = width // 2
                y_center = height // 2
                radius = min(width, height) // 2

                fig, ax = plt.subplots(figsize=figsize)
                ax.set_aspect('equal')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                agar_rgb = tuple(c / 255 for c in local_agar_color)
                agar_circle = plt.Circle((x_center, y_center), radius, color=agar_rgb, zorder=0)
                ax.add_patch(agar_circle)

                for _, row in df.iterrows():
                    x, y, area = row['X'], row['Y'], row['Area']
                    r = np.sqrt(area / np.pi) * scale
                    color = (row['R'] / 255, row['G'] / 255, row['B'] / 255)
                    colony = plt.Circle((x, y), r, color=color, alpha=alpha, zorder=1)
                    ax.add_patch(colony)

                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.invert_yaxis()
                ax.set_title(f"Colonies with Score ≥ {min_score}", color='white')

                output_path = os.path.join(dirpath, csv_filename.replace('.csv', '.png'))
                plt.tight_layout()
                plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300)
                plt.close()
                print(f"Saved plot to {output_path}")

            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
# %%

def plot_colonies2(root_folder, figsize=(10, 10), alpha=1, agar_color=None, min_score=0):
    """
    Plots colonies using masks stored in *_masks.npz and ROIs in *_results.csv.
    """
    print("start")
    for dirpath, _, filenames in os.walk(root_folder):
        print("file")
        csv_files = [f for f in filenames if f.endswith('_results.csv')]
        print(csv_files)
        for csv_filename in csv_files:
            
            csv_path = os.path.join(dirpath, csv_filename)

            # Find matching masks npz
            npz_file = csv_filename.replace('_results.csv', '_masks.npz')
            npz_path = os.path.join(dirpath, npz_file)
            if not os.path.exists(npz_path):
                print(f"Skipped {csv_filename}: no corresponding masks npz found.")
                continue

            # Find matching cropped_*.jpg
            cropped_image = next(
                (os.path.join(dirpath, f) for f in filenames
                 if f.lower().startswith('cropped_') and f.lower().endswith('.jpg')),
                None
            )
            if not cropped_image:
                print(f"Skipped {csv_filename}: No matching 'cropped_*.jpg'")
                continue

            img = cv2.imread(cropped_image)
            if img is None:
                print(f"Skipped {csv_filename}: Failed to load image {cropped_image}")
                continue
            height, width = img.shape[:2]
            
            crop_x0 = 0
            crop_y0 = 0
            match = re.search(r'cropped_(\d+)_(\d+)', cropped_image)
            if match:
                crop_x0 = int(match.group(1))
                crop_y0 = int(match.group(2))

            # Determine agar color
            local_agar_color = agar_color
            if agar_color is None:
                ori_image = next(
                    (os.path.join(dirpath, f) for f in filenames
                     if f.lower().startswith('ori_') and f.lower().endswith('.jpg')),
                    None
                )
                if ori_image:
                    try:
                        local_agar_color = get_mean_bgr_in_mask(ori_image)
                    except:
                        local_agar_color = (230, 230, 250)
                else:
                    local_agar_color = (230, 230, 250)

            # Load CSV and npz
            try:
                df = pd.read_csv(csv_path)
                masks_data = np.load(npz_path)
                if not {'Rois', 'Score'}.issubset(df.columns):
                    print(f"Skipped {csv_filename}: missing required columns.")
                    continue
                df = df[df['Score'] >= min_score]

                fig, ax = plt.subplots(figsize=figsize)
                ax.set_aspect('equal')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                # Draw agar background
                x_center, y_center = width // 2, height // 2
                radius = min(width, height) // 2
                agar_rgb = tuple(c / 255 for c in local_agar_color)
                agar_circle = plt.Circle((x_center, y_center), radius, color=agar_rgb, zorder=0)
                ax.add_patch(agar_circle)

                # Draw each colony using mask
                for _, row in df.iterrows():
                    label = int(row['Label'])
                    mask_key = f"mask_{label}"
                    if mask_key not in masks_data:
                        continue
                    mask = masks_data[mask_key]  # binary mask

                    roi = np.array(eval(row['Rois']), dtype=int)
                    y1, x1, y2, x2 = roi
                    roi_height = y2 - y1
                    roi_width  = x2 - x1
                    mask = masks_data[f"mask_{int(row['Label'])}"]
                    
                    # Get mask pixels
                    ys, xs = np.where(mask)
                    
                    # Scale to cropped image
                    ys_scaled = ys * roi_height / mask.shape[0]
                    xs_scaled = xs * roi_width  / mask.shape[1]

                    ys = ys_scaled + y1
                    xs = xs_scaled + x1

                    if len(xs) == 0:
                        continue

                    # Get mean color for the colony
                    colony_color = (row['R'] / 255, row['G'] / 255, row['B'] / 255)

                    # Draw pixels as small circles (scatter)
                    
                    ax.scatter(xs, ys, color=colony_color, s=1, alpha=alpha)
                    

                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.invert_yaxis()
                ax.set_title(f"Colonies with Score ≥ {min_score}", color='white')

                output_path = os.path.join(dirpath, csv_filename.replace('.csv', '_mask_plot.png'))
                plt.tight_layout()
                plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300)
                plt.close()
                print(f"Saved plot to {output_path}")

            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

# %%
plot_colonies("../test_photos/Xset/Old/results/r80_t31")
