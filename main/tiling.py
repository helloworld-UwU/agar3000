# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:31:18 2026

@author: Admin
"""


import os
import cv2
import numpy as np
import csv
# %%
def make_tiles(img, grid=(2,2), overlap=0.2):
    """
    Tiling of image according with grid and with overlapings.
    Produce tiles and overlaping of the same size.
    overlap: 0.2 means 20% overlap (relative to tile width/height)
    Returns a dict keyed by (row, col) -> (tile, (x0,y0,x1,y1)).
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    rows, cols = grid
    p = float(overlap)
    if not (0 <= p < 1):
        raise ValueError("overlap_pct must be in [0, 1).")
    
    # calculate size of the tile 
    tile_w = int(W / (cols - p*(cols - 1)))
    tile_h = int(H / (rows - p*(rows - 1)))
    
    step_w = int(tile_w*(1 - p))
    step_h = int(tile_h*(1 - p))
    
    tiles = {}
    
    x0 = 0
    y0 = 0
    
    for r in range(rows):
        for c in range(cols):
            x0 = c * step_w
            y0 = r * step_h
            x1 = min(x0 + tile_w, W)
            y1 = min(y0 + tile_h, H)
        
            tiles[(r, c)] = {
                "tile": img[y0:y1, x0:x1].copy(),
                "bbox": (x0, y0, x1, y1),
            }

    return tiles


# %%
def draw_boxes_on_tiles(tiles, color=(0, 0, 255), thickness=2,
                                detections_key="colonies", out_key="tile_with_boxes"):
    for rc, d in tiles.items():
        tile = d["tile"].copy()
        dets = d.get(detections_key, [])

        for det in dets:
            y1, x1, y2, x2 = det["roi"]  # tile coordinates
            cv2.rectangle(tile, (x1, y1), (x2, y2), color, thickness)

        d[out_key] = tile


def show_all_tiles_with_boxes(tiles_detections, key="tile_with_boxes",
                              cols=None, figsize=(1200, 1200),
                              output_folder="output", name="tiles"):
    """
    Save a grid of tiles with drawn bounding boxes as a JPEG file.

    Args:
        tiles_detections : dict  {(row, col): {"tile": ..., "colonies": ..., ...}}
        key              : str   - key holding the annotated tile image
        cols             : int   - number of columns in the grid (auto if None)
        figsize          : tuple - (total_width, total_height) in pixels
        output_folder    : str   - folder to save the output image
        name             : str   - filename (without extension)
    """
    draw_boxes_on_tiles(tiles_detections)

    items = sorted(tiles_detections.items(), key=lambda kv: kv[0])
    n = len(items)
    if n == 0:
        return

    if cols is None:
        cols = int(round(n ** 0.5)) or 1
    rows = (n + cols - 1) // cols

    sample = items[0][1][key]
    cell_h, cell_w = sample.shape[:2]
    pad = 10  # white border thickness in pixels

    grid_rows = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                img = cv2.cvtColor(items[idx][1][key].copy(), cv2.COLOR_BGR2RGB)
                cv2.putText(img, str(items[idx][0]), (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1,
                            cv2.LINE_AA)
            else:
                img = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 255  # white padding cell

            img = cv2.resize(img, (cell_w, cell_h))
            # add white border around each tile
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
            row_tiles.append(img)

        grid_rows.append(np.hstack(row_tiles))

    grid = cv2.resize(np.vstack(grid_rows), figsize, interpolation=cv2.INTER_AREA)

    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, f"{name}.jpg"), grid,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

# %%
def filter_colonies_by_score(
    tiles,
    threshold,
    detections_key="colonies",
    score_regression=None,  # optional: (low_delta, high_delta)
    min_count=10,
    max_count=300,
    verbose=False,
):
    thr = float(threshold)

    if score_regression is None:
        for d in tiles.values():
            dets = d.get(detections_key, [])
            d[detections_key] = [det for det in dets if det.get("score", 0.0) >= thr]
        return thr

    # --- first pass: count only, do NOT modify tiles ---
    count = sum(
        sum(1 for det in d.get(detections_key, []) if det.get("score", 0.0) >= thr)
        for d in tiles.values()
    )

    # --- derive corrected threshold from count ---
    low_delta, high_delta = score_regression
    y_low     = thr + low_delta
    y_high    = thr + high_delta
    slope     = (y_high - y_low) / (max_count - min_count)
    intercept = y_low - slope * min_count

    corrected = slope * count + intercept
    corrected = max(0.0, min(1.0, corrected))

    if verbose:
        print(
            f"[score_regression] base={thr:.4f}, low_delta={low_delta:+.4f}, high_delta={high_delta:+.4f}\n"
            f"                   slope={slope:.6f}, intercept={intercept:.6f}\n"
            f"                   count={count}, corrected={corrected:.4f}"
        )

    # --- second pass: apply corrected threshold to original detections ---
    for d in tiles.values():
        dets = d.get(detections_key, [])
        d[detections_key] = [det for det in dets if det.get("score", 0.0) >= corrected]

    return corrected
        
# %%

def iou_yxyx(a, b):
    # a,b: (y1,x1,y2,x2)
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b

    iy1 = max(ay1, by1)
    ix1 = max(ax1, bx1)
    iy2 = min(ay2, by2)
    ix2 = min(ax2, bx2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ay2 - ay1) * max(0, ax2 - ax1)
    area_b = max(0, by2 - by1) * max(0, bx2 - bx1)
    union = area_a + area_b - inter

    return 0.0 if union == 0 else inter / union

def merge_rois_yxyx(a, b):
    # union box of two ROIs
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    return (min(ay1, by1), min(ax1, bx1), max(ay2, by2), max(ax2, bx2))

def merge_with_best_iou(ref_roi, roi_list):
    """
    Picks roi from roi_list with maximal IoU vs ref_roi, merges them,
    removes the chosen roi from the list, and returns (merged_roi, new_list).

    ROI format: (y1,x1,y2,x2)
    """
    if not roi_list:
        return ref_roi, []

    best_i = 0
    best_iou = -1.0
    for i, roi in enumerate(roi_list):
        v = iou_yxyx(ref_roi, roi)
        if v > best_iou:
            best_iou = v
            best_i = i

    best_roi = roi_list[best_i]
    merged = merge_rois_yxyx(ref_roi, best_roi)
    new_list = roi_list[:best_i] + roi_list[best_i + 1:]
    return merged, new_list

def resolve_overlap_pair(
    primary_dets, secondary_dets,
    bbox_primary, bbox_secondary,
    position="x", tol=5, merge_key="merge"
):
    px0, py0, px1, py1 = bbox_primary
    sx0, sy0, sx1, sy1 = bbox_secondary

    def near(a, b):
        return abs(a - b) <= tol

    if position not in ("x", "y"):
        raise ValueError("position must be 'x' or 'y'.")

    removed = 0
    merged = 0

    primary_keep, primary_merge = [], []
    for det in primary_dets:
        y1, x1, y2, x2 = det["roi_global"]
        remove = False
        merge = False

        if position == "x":
            if near(x2, px1):
                if x1 > sx0:
                    remove = True
                else:
                    merge = True
        else:  # "y"
            if near(y2, py1):
                if y1 > sy0:
                    remove = True
                else:
                    merge = True

        if remove:
            removed += 1
        else:
            (primary_merge if merge else primary_keep).append(det)

    secondary_keep, secondary_merge = [], []
    for det in secondary_dets:
        y1, x1, y2, x2 = det["roi_global"]
        remove = False
        merge = False

        if position == "x":
            if x2 < px1:
                remove = True
            elif near(x1, sx0):
                merge = True
        else:  # "y"
            if y2 < py1:
                remove = True
            elif near(y1, sy0):
                merge = True

        if remove:
            removed += 1
        else:
            (secondary_merge if merge else secondary_keep).append(det)

    for det in primary_merge:
        if not secondary_merge:
            primary_keep.append(det)
            continue

        ref = det["roi_global"]
        best_i, best_iou = 0, -1.0
        for i, sdet in enumerate(secondary_merge):
            v = iou_yxyx(ref, sdet["roi_global"])
            if v > best_iou:
                best_iou = v
                best_i = i

        det = dict(det)
        det["roi_global"] = merge_rois_yxyx(det["roi_global"], secondary_merge[best_i]["roi_global"])
        secondary_merge.pop(best_i)

        merged += 1
        primary_keep.append(det)

    new_primary = primary_keep
    new_secondary = secondary_keep + secondary_merge

    print(f"removed={removed}, merged={merged}")
    return new_primary, new_secondary

# %%
def resolve_duplicates_across_tiles(tiles, tol=5, detections_key="colonies"):
    """
    Infers grid size from tiles keys (row, col).
    Updates tiles in-place (silent): horizontal neighbors first, then vertical.
    """
    if not tiles:
        return

    rows = max(r for (r, c) in tiles.keys()) + 1
    cols = max(c for (r, c) in tiles.keys()) + 1

    # ---- horizontal neighbors ----
    for r in range(rows):
        for c in range(cols - 1):
            if (r, c) not in tiles or (r, c + 1) not in tiles:
                continue
            print(f"Resolve overlap pair: ({r},{c}),({r},{c+1}) ->", end="")
            
            left = tiles[(r, c)]
            right = tiles[(r, c + 1)]

            new_left, new_right = resolve_overlap_pair(
                left.get(detections_key, []),
                right.get(detections_key, []),
                left["bbox"], right["bbox"],
                position="x", tol=tol
            )
            left[detections_key] = new_left
            right[detections_key] = new_right

    # ---- vertical neighbors ----
    for r in range(rows - 1):
        for c in range(cols):
            if (r, c) not in tiles or (r + 1, c) not in tiles:
                continue
            
            print(f"Resolve overlap pair: ({r},{c}),({r+1},{c}) ->", end="")
            top = tiles[(r, c)]
            bottom = tiles[(r + 1, c)]

            new_top, new_bottom = resolve_overlap_pair(
                top.get(detections_key, []),
                bottom.get(detections_key, []),
                top["bbox"], bottom["bbox"],
                position="y", tol=tol
            )
            top[detections_key] = new_top
            bottom[detections_key] = new_bottom

# %%

def show_all_rois_global(img_bgr, tiles, detections_key="colonies",
                         color=(0, 0, 255), thickness=2,
                         name="rois", output_folder="output",
                         show_scores=True, font_scale=0.4, font_thickness=1):
    """
    Draws all det["roi_global"] on a copy of img_bgr with optional score labels
    and saves the result as a JPEG file.

    Args:
        img_bgr         : np.ndarray - original BGR image
        tiles           : dict       - tiles with detection info
        detections_key  : str        - key in tiles containing detections
        color           : tuple      - BGR color for rectangles and text
        thickness       : int        - rectangle thickness
        name            : str        - filename (without extension)
        output_folder   : str        - folder to save the output image
        show_scores     : bool       - draw score labels on each box (default True)
        font_scale      : float      - cv2 font scale for score text (default 0.4)
        font_thickness  : int        - cv2 font thickness for score text (default 1)
    """
    out = img_bgr.copy()
    for _, d in tiles.items():
        for det in d.get(detections_key, []):
            y1, x1, y2, x2 = det["roi_global"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            if show_scores:
                text = "{:.2f}".format(det["score"])
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(out, text, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thickness, cv2.LINE_AA)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{name}.jpg")
    cv2.imwrite(output_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    
# %%
def save_plate_tiles_to_csv(output_folder, tiles, name):
    """
    Save tile and colony detection information for a plate to a CSV file.

    Parameters:
    - output_folder (str): Folder where the CSV will be saved
    - plate: Object containing at least `sample_id`
    - tiles (dict): Dictionary with tile positions as keys and detection data as values
    """
    csv_path = os.path.join(output_folder, f"{name}.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "tile_row", "tile_col",
            "tile_x0", "tile_y0", "tile_x1", "tile_y1",
            "y1", "x1", "y2", "x2", "score"
        ])
        
        # Write each tile and its colonies
        for (r, c), d in sorted(tiles.items()):
            x0, y0, x1, y1 = d["bbox"]
            for det in d.get("colonies", []):
                ry1, rx1, ry2, rx2 = det["roi_global"]
                score = det.get("score", "")
                writer.writerow([r, c, x0, y0, x1, y1, ry1, rx1, ry2, rx2, score])



