# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 05:16:20 2025

@author: Admin
"""
# %%
import csv
import main.tiling as tiling
import cv2
import os


# %%
def load_plate_tiles_from_csv(csv_path):
    """
    Load tile and colony detection information from a CSV file
    and reconstruct the tiles dictionary.

    Returns:
    - tiles (dict): {(tile_row, tile_col): {"bbox": [...], "colonies": [...]}}
    """
    tiles = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            r = int(row["tile_row"])
            c = int(row["tile_col"])
            key = (r, c)

            # Initialize tile entry if needed
            if key not in tiles:
                tiles[key] = {
                    "bbox": [
                        int(row["tile_x0"]),
                        int(row["tile_y0"]),
                        int(row["tile_x1"]),
                        int(row["tile_y1"]),
                    ],
                    "colonies": []
                }

            # Append colony detection
            det = {
                "roi_global": [
                    int(row["y1"]),
                    int(row["x1"]),
                    int(row["y2"]),
                    int(row["x2"]),
                ],
                "score": (
                    float(row["score"])
                    if row["score"] not in ("", None)
                    else None
                )
            }

            tiles[key]["colonies"].append(det)

    return tiles

def regression_from_score(base_score, min_count=10, max_count=300,
                           low_delta=-0.02, high_delta=+0.05):
    """
    Derive (slope, intercept) such that:
      y(min_count) = base_score + low_delta
      y(max_count) = base_score + high_delta

    Returns (slope, intercept) for:  corrected_score = slope * count + intercept
    """
    y_low  = base_score + low_delta
    y_high = base_score + high_delta

    slope     = (y_high - y_low) / (max_count - min_count)
    intercept = y_low - slope * min_count

    return slope, intercept


def scores_filter(
    csv_folder,
    plate_folder,
    output_folder,
    debug_output,
    score,
    score_regression=None   # optional: (low_delta, high_delta) 
):
    """
    For each CSV in csv_folder:
    - load tiles from CSV
    - find matching image in plate_folder (same basename, auto-detect common extensions)
    - filter colonies by score (constant threshold)
    - if score_regression=(slope, intercept) is provided:
        - count colonies after initial filter
        - compute corrected threshold: y = (count - intercept) / slope
        - re-filter using the corrected threshold
    - save filtered CSV
    - visualize + save global ROI image
    """
    os.makedirs(output_folder, exist_ok=True)
    

    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for fname in sorted(os.listdir(csv_folder)):
        if not fname.lower().endswith(".csv"):
            continue
        name = os.path.splitext(fname)[0]
        csv_path = os.path.join(csv_folder, fname)

        # --- find matching image ---
        img_path = None
        for ext in image_exts:
            candidate = os.path.join(plate_folder, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"[WARN] No image found for {name}, skipping.")
            continue

        # --- load data ---
        tiles = load_plate_tiles_from_csv(csv_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Failed to load image: {img_path}, skipping.")
            continue

        # --- first pass: constant threshold ---
        tiling.filter_colonies_by_score(tiles, score)

        # --- optional second pass: regression-adjusted threshold ---
        if score_regression is not None:
            low_delta, high_delta = score_regression
            slope, intercept = regression_from_score(score, min_count=10, max_count=300,
                           low_delta=low_delta, high_delta=high_delta)

            # count detections after first pass
            count = sum(len(tile["colonies"]) for tile in tiles.values())

            corrected_score = slope * count + intercept
            corrected_score = max(0.0, min(1.0, corrected_score))  # clamp to [0, 1]

            print(f"[{name}] count={count}, corrected_score={corrected_score:.4f}")

            # reload original tiles and re-filter with corrected threshold
            tiles = load_plate_tiles_from_csv(csv_path)
            tiling.filter_colonies_by_score(tiles, corrected_score)

        # --- save filtered CSV ---
        tiling.save_plate_tiles_to_csv(output_folder, tiles, name)


        # --- visualize + save global ROIs ---
        if debug_output is not None:
        
            os.makedirs(debug_output, exist_ok=True)

            tiling.show_all_rois_global(
                img_bgr.copy(),
                tiles,
                name=name,
                output_folder=debug_output
            )

def deduplication_filter(
    csv_folder,
    plate_folder,
    output_folder,
    tol
):
    os.makedirs(output_folder, exist_ok=True)

    # List of common image extensions
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for fname in sorted(os.listdir(csv_folder)):
        if not fname.lower().endswith(".csv"):
            continue

        name = os.path.splitext(fname)[0]
        csv_path = os.path.join(csv_folder, fname)

        # --- find matching image ---
        img_path = None
        for ext in image_exts:
            candidate = os.path.join(plate_folder, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"[WARN] No image found for {name}, skipping.")
            continue

        # --- load data ---
        tiles = load_plate_tiles_from_csv(csv_path)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"[WARN] Failed to load image: {img_path}, skipping.")
            continue

        # --- filter detections ---
        tiling.resolve_duplicates_across_tiles(tiles, tol=tol, detections_key="colonies")

        # --- save filtered CSV ---
        tiling.save_plate_tiles_to_csv(output_folder, tiles, name)

        # --- visualize + save global ROIs ---
        tiling.show_all_rois_global(
            img_bgr.copy(),
            tiles,
            name=name,
            output_folder=output_folder
        )
# %%
"""
deduplication_filter(
    csv_folder='test_min/result7/dup',
    plate_folder='test_min/result7/crop',
    output_folder='test_min/result7/dedup10',
    tol = 10
)
"""
