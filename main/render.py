# -*- coding: utf-8 -*-
import os, csv
import cv2
import numpy as np
import json


def save_geometry(plate, output_folder, name=None, ext=".jpg",
                  color=(0, 0, 255), thickness=2):
    """
    Write the original image to disk with the processing geometry
    drawn on top.

    Overlays are added only for the steps that actually ran:
      - crop    : rectangle of plate.crop_bbox
      - mask    : circle inscribed in that rectangle
      - tiling  : rectangle of every tile bbox, shifted into
                  original-image coordinates

    Parameters
    ----------
    plate         : Plate
    output_folder : str   Created if missing.
    name          : str   Filename stem (default: plate.sample_id).
    ext           : str   File extension, selects the encoder.
    color         : tuple BGR colour, used for every line.
    thickness     : int   Line thickness, used for every line.

    Returns
    -------
    path : str  Full path of the written file.
    """
    if plate.image is None:
        raise ValueError(f"Plate '{plate.sample_id}' has no image to save.")

    out = plate.image.copy()

    # -- crop rectangle + inscribed mask circle ---------------------------
    if plate.crop_bbox is not None:
        x1, y1, x2, y2 = plate.crop_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(out,
                   ((x1 + x2) // 2, (y1 + y2) // 2),
                   min(x2 - x1, y2 - y1) // 2,
                   color, thickness)

    # -- tile rectangles --------------------------------------------------
    for t in plate.tiles.values():
        tx0, ty0, tx1, ty1 = t.bbox
        cv2.rectangle(out, (tx0, ty0), (tx1, ty1), color, thickness)

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"{name or plate.sample_id}{ext}")
    if not cv2.imwrite(path, out):
        raise ValueError(f"Could not save image to {path}")
    return path


def show_tiles(plate, output_folder, name=None, ext=".jpg",
               color=(0, 0, 255), thickness=2,
               pad=10):
    """
    Save a grid montage of the plate's tiles with their colony boxes
    drawn in tile-local coordinates.

    Parameters
    ----------
    plate         : Plate
    output_folder : str   Created if missing.
    name          : str   Filename stem (default: f"{sample_id}_tiles").
    ext           : str   File extension.
    color         : tuple BGR colour for every box.
    thickness     : int   Line thickness for every box.
    pad           : int   White border between cells, in pixels.

    Returns
    -------
    path : str  Full path of the written file, or None if no tiles.
    """
    if not plate.tiles:
        return None

    items = sorted(plate.tiles.items(), key=lambda kv: kv[0])

    # -- draw boxes on a copy of each tile ------------------------------
    cells = []
    for _, t in items:
        img = t.image.copy()
        for col in t.colonies:
            y1, x1, y2, x2 = col.roi
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cells.append(img)

    # -- layout ----------------------------------------------------------
    cols = max(c for (_, c) in plate.tiles) + 1
    rows = max(r for (r, _) in plate.tiles) + 1

    cell_h = max(c.shape[0] for c in cells)
    cell_w = max(c.shape[1] for c in cells)

    canvas = np.full(
        (rows * cell_h + (rows + 1) * pad,
         cols * cell_w + (cols + 1) * pad, 3),
        255, dtype=np.uint8,
    )

    for (r, c), cell in zip((rc for rc, _ in items), cells):
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        h, w = cell.shape[:2]
        canvas[y:y + h, x:x + w] = cell

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder,
                        f"{name or plate.sample_id}_tiles{ext}")
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def save_csv(plate, output_folder, name=None):
    """
    Write one row per detected colony, with the geometry of the tile
    it was found in.

    Columns
    -------
    tile_row, tile_col          : grid position of the tile
    tile_x0..tile_y1            : tile bbox in cropped-plate coords
    y1, x1, y2, x2              : colony roi_global, cropped-plate coords
    score                       : detector confidence

    Parameters
    ----------
    plate         : Plate
    output_folder : str  Created if missing.
    name          : str  Filename stem (default: plate.sample_id).

    Returns
    -------
    path : str  Full path of the written file.
    """
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"{name or plate.sample_id}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "tile_row", "tile_col",
            "tile_x0", "tile_y0", "tile_x1", "tile_y1",
            "y1", "x1", "y2", "x2", "score",
        ])

        for (r, c), t in sorted(plate.tiles.items()):
            x0, y0, x1, y1 = t.bbox
            for col in t.colonies:
                ry1, rx1, ry2, rx2 = col.roi_global
                writer.writerow([r, c, x0, y0, x1, y1,
                                 ry1, rx1, ry2, rx2, col.score])

    return path




def save_json(plate, output_folder, name=None, indent=2):
    """
    Write detected colonies as a COCO-format object detection JSON.

    One file per plate, describing a single image (the original, full-res
    plate photo) and one annotation per colony.

    COCO geometry is [x, y, width, height] with the origin at the top-left
    of the ORIGINAL image, which is the coordinate space Colony.roi_global
    already lives in.

    The only non-standard key is "score", the detector confidence, carried
    per annotation.

    Parameters
    ----------
    plate         : main.data.Plate
    output_folder : str  Created if missing.
    name          : str  Filename stem (default: plate.sample_id).
    indent        : int  json.dump indent; pass None for a compact file.

    Returns
    -------
    path : str  Full path of the written file.
    """
    # COCO category table; single class, ids are 1-based
    COCO_CATEGORIES = [{"id": 1, "name": "colony", "supercategory": "microbe"}]
    
    if plate.image is None:
        raise ValueError(f"Plate '{plate.sample_id}' has no image.")

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"{name or plate.sample_id}.json")

    img_h, img_w = plate.image.shape[:2]
    image_id = 1

    coco = {
        "images": [{
            "id": image_id,
            "file_name": os.path.basename(plate.image_path) if plate.image_path
                         else f"{plate.sample_id}.jpg",
            "width": int(img_w),
            "height": int(img_h),
        }],
        "annotations": [],
        "categories": COCO_CATEGORIES,
    }

    ann_id = 1
    for _, t in sorted(plate.tiles.items()):
        for col in t.colonies:
            y1, x1, y2, x2 = col.roi_global
            x, y = int(x1), int(y1)
            w, h = int(x2) - x, int(y2) - y

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": int(w * h),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(col.score),
            })
            ann_id += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=indent)

    return path



def merge_json(json_folder, output_path="prediction.json", indent=2, verbose=False):
    """
    Combine every per-plate COCO JSON in a folder into a single dataset.

    Each per-plate file numbers its image as id 1 and its annotations from
    1, so ids are reassigned here: images get a running id in filename
    order, and every annotation is relinked to its new image_id.

    The categories table is taken from the first file and checked against
    the rest; a mismatch raises, since merging incompatible label sets
    would silently corrupt category_id.

    Files are matched by extension only, and output_path is skipped if it
    happens to live inside json_folder.

    Parameters
    ----------
    json_folder : str   Folder containing per-plate .json files.
    output_path : str   Path of the merged file.
    indent      : int   json.dump indent; None for a compact file.
    verbose     : bool  Print a one-line summary.

    Returns
    -------
    path : str  Full path of the written file.
    """
    out_abs = os.path.abspath(output_path)

    images = []
    annotations = []
    categories = None

    next_image_id = 1
    next_ann_id = 1
    skipped = []

    for fname in sorted(os.listdir(json_folder)):
        if not fname.lower().endswith(".json"):
            continue

        src = os.path.join(json_folder, fname)
        if os.path.abspath(src) == out_abs:
            continue

        with open(src, "r", encoding="utf-8") as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError:
                skipped.append(fname)
                continue

        if not isinstance(doc, dict) or "images" not in doc:
            skipped.append(fname)
            continue

        # -- categories: first file wins, the rest must agree -------------
        if categories is None:
            categories = doc.get("categories", [])
        elif doc.get("categories", []) != categories:
            raise ValueError(
                f"Category table in '{fname}' differs from the first file; "
                f"cannot merge."
            )

        # -- images, with an old -> new id map ----------------------------
        id_map = {}
        for img in doc["images"]:
            id_map[img["id"]] = next_image_id
            new_img = dict(img)
            new_img["id"] = next_image_id
            images.append(new_img)
            next_image_id += 1

        # -- annotations, relinked ----------------------------------------
        for ann in doc.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[ann["image_id"]]
            annotations.append(new_ann)
            next_ann_id += 1

    if categories is None:
        raise ValueError(f"No readable COCO JSON files found in: {json_folder}")

    merged = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=indent)

    if verbose:
        print("-----------------------------------------------------")
        print(f"Merged {len(images)} image(s), {len(annotations)} annotation(s)")
        if skipped:
            print(f"Skipped {len(skipped)} unreadable file(s): "
                  f"{', '.join(skipped[:5])}"
                  f"{' ...' if len(skipped) > 5 else ''}")
        print(f"Merged COCO saved to: {out_abs}")

    return out_abs

def show_colonies(self, output_folder, name=None, ext=".jpg",
                  color=(0, 0, 255), thickness=2,
                  show_scores=True, font_scale=0.4, font_thickness=1,
                  show_plate=True, plate_color=(0, 255, 0)):
    """
    Save the original image with every detected colony boxed.


    Parameters
    ----------
    output_folder  : str   Created if missing.
    name           : str   Filename stem (default: self.sample_id).
    ext            : str   File extension.
    color          : tuple BGR colour for boxes and score labels.
    thickness      : int   Box line thickness.
    show_scores    : bool  Draw the score above each box.
    font_scale     : float cv2 font scale for the score text.
    font_thickness : int   cv2 font thickness for the score text.
    show_plate     : bool  Draw the detected plate rectangle + circle.
    plate_color    : tuple BGR colour for the plate geometry.

    Returns
    -------
    path : str  Full path of the written file.
    """
    if self.image is None:
        raise ValueError(f"Plate '{self.sample_id}' has no image.")

    out = self.image.copy()

    # -- detected plate: crop rectangle + inscribed mask circle -----------
    if show_plate and self.crop_bbox is not None:
        px1, py1, px2, py2 = self.crop_bbox
        cv2.rectangle(out, (px1, py1), (px2, py2), plate_color, thickness)
        cv2.circle(out,
                   ((px1 + px2) // 2, (py1 + py2) // 2),
                   min(px2 - px1, py2 - py1) // 2,
                   plate_color, thickness)

    # -- colonies ----------------------------------------------------------
    for t in self.tiles.values():
        for col in t.colonies:
            y1, x1, y2, x2 = col.roi_global
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            if show_scores:
                text = "{:.2f}".format(col.score)
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_thickness)
                cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1),
                              color, -1)
                cv2.putText(out, text, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thickness, cv2.LINE_AA)

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"{name or self.sample_id}{ext}")
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path