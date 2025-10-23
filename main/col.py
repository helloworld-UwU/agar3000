# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:02:28 2025

@author: Admin
"""

# %%

import agarDetect
# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
import csv
import shutil
import pandas as pd
import glob
import pickle

# %%
def detect_colonies_in_splits(split_dir, vertical_splits, horizontal_splits, n_stripes=5):
    label_counter = 1
    records = []
    masks_dict = {}

    # Infer base plate name
    split_files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
    base_name = None
    for fname in split_files:
        match = re.match(r"(.+?)_stripe_\d+_(top|bottom)\.jpg", fname)
        if match:
            base_name = match.group(1)
            break

    if base_name is None:
        raise ValueError("Could not infer base plate name from filenames.")

    for i in range(n_stripes):
        print(f"{i+1}/{n_stripes}")
        stripe_label = f"stripe_{i + 1}"
        x_offset = vertical_splits[i][0]  # x shift for this stripe
        y_split = horizontal_splits[i]    # y shift for top/bottom

        for suffix, y_offset in [("top", 0), ("bottom", y_split)]:
            img_filename = f"{base_name}_{stripe_label}_{suffix}.jpg"
            img_path = os.path.join(split_dir, img_filename)
            
            # Load the image (RGB)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            detection = agarConfig_2.detectResults(img_path)
            if not detection or 'rois' not in detection[0] or 'masks' not in detection[0]:
                continue

            rois = detection[0]['rois']
            masks = detection[0]['masks']
            scores = detection[0]['scores']

            # Ensure masks shape is (H, W, N)
            if masks.ndim == 2:
                masks = masks[..., np.newaxis]

            for j in range(masks.shape[-1]):
                mask = masks[:, :, j]
                area = int(np.sum(mask))  # Count True pixels

                y1, x1, y2, x2 = rois[j]

                # Apply offsets to centroid
                cx = x1 + (x2 - x1) / 2 + x_offset
                cy = y1 + (y2 - y1) / 2 + y_offset

                # Apply offsets to ROI coordinates
                y1_corrected = y1 + y_offset
                y2_corrected = y2 + y_offset
                x1_corrected = x1 + x_offset
                x2_corrected = x2 + x_offset

                score = float(scores[j])
                
                # Extract color inside mask
                color_pixels = image_rgb[mask.astype(bool)]
                if len(color_pixels) == 0:
                    r, g, b = (0, 0, 0)
                else:
                    r, g, b = map(int, np.mean(color_pixels, axis=0))

                # Save metadata
                records.append({
                    "Label": label_counter,
                    "Rois": str([y1_corrected, x1_corrected, y2_corrected, x2_corrected]),
                    "X": round(cx, 2),
                    "Y": round(cy, 2),
                    "Area": area,
                    "R": r,
                    "G": g,
                    "B": b,
                    "Stripe": i + 1,
                    "Score": round(score, 4)
                })

                # Save mask separately in dictionary
                masks_dict[f"mask_{label_counter}"] = mask.astype(np.uint8)

                label_counter += 1

    # Save CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(os.path.dirname(split_dir), f"{base_name}_results.csv")
    df.to_csv(csv_path, index=False)

    # Save all masks in one compressed file
    masks_path = os.path.join(os.path.dirname(split_dir), f"{base_name}_masks.npz")
    np.savez_compressed(masks_path, **masks_dict)

    print(f"✅ Saved {len(records)} colonies:")
    print(f"   - Metadata: {csv_path}")
    print(f"   - Masks: {masks_path}")



# %%

def split_detect_all_colonies(input_folder, n_stripes=5):
    """
    Recursively processes all images with 'ori_' prefix in subfolders:
    - Splits each image into vertical/horizontal sections.
    - Runs colony detection and saves results.

    Args:
        input_folder (str): Root folder containing subfolders with 'ori_' images.
        output_root (str): Root directory to store all results.
        n_stripes (int): Number of vertical stripes to split each image into.
    """
    image_files = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            if f.lower().startswith("cropped_") and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, f)
                image_files.append(full_path)

    print(f"Found {len(image_files)} 'ori_' image(s) under '{input_folder}'.")

    for image_file in image_files:
        split_dir =os.path.join(os.path.dirname(image_file), "splits")
        print(f"\nProcessing '{image_file}'...")

        # Split image
        vertical_splits, horizontal_splits = split_image(
            image_file,  n_stripes=n_stripes)
        

        # Count colonies
        detect_colonies_in_splits(
            split_dir,
            vertical_splits,
            horizontal_splits,
            n_stripes
        )

        
# %%
def detect_all_colonies(input_folder):
    """
    Finds all pickle files ending with '_splits.pkl' in the input folder tree,
    loads their split information, and runs detect_colonies_in_splits().
    """
    for root, _, files in os.walk(input_folder):
        # Filter files ending with '_splits.pkl'
        split_files = [f for f in files if f.endswith("_splits.pkl")]
        
        for pkl_file in split_files:
            pkl_path = os.path.join(root, pkl_file)
            print(f"\nProcessing: {pkl_path}")

            with open(pkl_path, "rb") as f:
                split_data = pickle.load(f)

            vertical_splits = split_data["vertical_splits"]
            horizontal_splits = split_data["horizontal_splits"]

            detect_colonies_in_splits(
                split_dir=root,
                vertical_splits=vertical_splits,
                horizontal_splits=horizontal_splits,
                n_stripes=len(vertical_splits)
            )





# %%

def summarize_csv_counts(root_dir, summary_output_path):
    """
    Recursively finds all CSV files in a folder, counts data rows,
    and writes a summary CSV.

    Args:
        root_dir (str): Root directory to search.
        summary_output_path (str): Path to save the summary CSV.
    """

    summary_data = []

    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(folder, file)
                try:
                    with open(csv_path, newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        row_count = sum(1 for _ in reader)

                    summary_data.append({
                        "File": os.path.relpath(csv_path, root_dir),
                        "Samples": row_count
                    })
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

    # Write summary CSV
    with open(summary_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["File", "Samples"])
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"Summary saved to: {summary_output_path}")


# %%

def combine_colonies_csv(parent_dir, output_filename='all_results.csv'):
    """
    Combine all *_results.csv files from subfolders of a parent directory.
    Adds a 'Plate' column to indicate the source folder.

    Skips empty CSVs without raising an error.

    Parameters:
        parent_dir (str): Path to the parent directory containing subfolders.
        output_filename (str): Name of the output CSV file (default: 'all_results.csv').

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    df_list = []

    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)

        if os.path.isdir(folder_path):
            csv_files = glob.glob(os.path.join(folder_path, '*_results.csv'))

            for csv_file in csv_files:
                if os.path.getsize(csv_file) == 0:
                    # Skip completely empty files
                    print(f"! Skipping empty file: {csv_file}")
                    continue

                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        print(f"! No data in file (empty DataFrame): {csv_file}")
                        continue

                    df['Plate'] = folder_name
                    df_list.append(df)

                except pd.errors.EmptyDataError:
                    print(f"! Skipping file with no readable data: {csv_file}")
                    continue
                except Exception as e:
                    print(f"! Error reading {csv_file}: {e}")
                    continue

    if not df_list:
        print("! No valid *_results.csv files with data were found.")
        return pd.DataFrame()  # Return empty DataFrame instead of raising error

    combined_df = pd.concat(df_list, ignore_index=True)

    output_path = os.path.join(parent_dir, output_filename)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")

    return combined_df