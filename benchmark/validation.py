# -*- coding: utf-8 -*-


import os
import json
import csv
import math

import cv2


# %% OPTIMAL THRESHOLD CALIBRATION


def smape(pred: dict, truth: dict, min: int, max: int, verbose: bool = True) -> float:
    """
    Symmetric mean absolute percentage error (0-200 scale).

    pred, truth : dicts of sample name -> colony count (int).
                  Only sample names present in both are used.
    min, max    : inclusive integer limits applied to the ground-truth counts.
    verbose     : print the number of matched and used samples.
    """
    matched = truth.keys() & pred.keys()
    if verbose:
        print(f"Matched samples: {len(matched)} "
              f"(truth: {len(truth)}, predictions: {len(pred)})")

    scores = []
    for name in matched:
        t, p = truth[name], pred[name]
        if not (min <= t <= max):
            continue
        if t == 0 and p == 0:                  # 0/0 is undefined
            continue
        scores.append(200 * abs(p - t) / (t + p))

    if verbose:
        print(f"Used in calculation: {len(scores)}")

    if not scores:
        raise ValueError("No observations remain after matching and filtering.")
    return sum(scores) / len(scores)



def load_json(path: str) -> dict:
    """
    Reads a COCO-format json.

    If every annotation has a "score", returns
    {file_name_without_extension: [(annotation_id, score), ...]}.

    If no annotation has a "score", returns
    {file_name_without_extension: number_of_detections}, the same structure
    as load_csv().

    Images without annotations are kept (empty list or count of 0).
    File names are stripped and converted to upper case, so that matching
    with load_csv() is case-insensitive.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    id_to_name = {}
    by_image = {}
    for img in data["images"]:
        name = img["file_name"].rsplit(".", 1)[0].strip().upper()
        if name in by_image:
            raise ValueError(f"Duplicate file name: {name}")
        id_to_name[img["id"]] = name
        by_image[name] = []

    annotations = data["annotations"]
    n_scored = sum("score" in ann for ann in annotations)
    if 0 < n_scored < len(annotations):
        raise ValueError(f"Only {n_scored} of {len(annotations)} annotations "
                         f"have a 'score'.")
    has_scores = n_scored > 0

    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in id_to_name:
            raise ValueError(f"image_id {image_id} is not listed in 'images'.")
        item = (ann["id"], ann["score"]) if has_scores else ann["id"]
        by_image[id_to_name[image_id]].append(item)

    if has_scores:
        return by_image
    return {name: len(items) for name, items in by_image.items()}


def count_above(prediction: dict, threshold: float) -> dict:
    """Number of detections with score >= threshold, per image."""
    return {image_id: sum(score >= threshold for _, score in items)
            for image_id, items in prediction.items()}


def ids_above(prediction: dict, threshold: float) -> dict:
    """Annotation ids with score >= threshold, per image."""
    return {image_id: [ann_id for ann_id, score in items if score >= threshold]
            for image_id, items in prediction.items()}

def filter_json(path: str, out_path: str, ids: dict) -> None:
    """
    Writes a copy of a COCO-format json that keeps only the annotations
    listed in ids.

    path     : the json that was read with load_json().
    out_path : output json.
    ids      : output of ids_above(), {file_name_without_extension: [annotation_id, ...]}.

    All other parts of the json (images, categories, info) are kept unchanged.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # The names must be built exactly as in load_json()
    id_to_name = {img["id"]: img["file_name"].rsplit(".", 1)[0].strip().upper()
                  for img in data["images"]}
    keep = {name: set(ann_ids) for name, ann_ids in ids.items()}

    total = len(data["annotations"])
    kept = []
    for ann in data["annotations"]:
        name = id_to_name.get(ann["image_id"])
        if name is None:
            raise ValueError(f"image_id {ann['image_id']} is not listed in 'images'.")
        if ann["id"] in keep.get(name, ()):
            kept.append(ann)

    data["annotations"] = kept
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Annotations kept: {len(kept)} of {total}")
    
def render_json(json_path: str, images_folder: str, output_folder: str,
                ext: str = ".jpg", color: tuple = (0, 0, 255),
                thickness: int = 2, show_scores: bool = True,
                font_scale: float = 0.4, font_thickness: int = 1,
                show_plate: bool = True,
                plate_color: tuple = (0, 255, 0)) -> list:
    """
    Saves every image listed in a COCO-format json with its annotations boxed.

    json_path      : COCO json, for example the output of filter_json().
    images_folder  : folder holding the images named in the json.
    output_folder  : created if missing.
    ext            : file extension of the output images.
    color          : BGR colour for boxes and score labels.
    thickness      : box line thickness.
    show_scores    : draw the score above each box, if the annotation has one.
    font_scale     : cv2 font scale for the score text.
    font_thickness : cv2 font thickness for the score text.
    show_plate     : draw the plate crop rectangle and its inscribed circle,
                     if the image entry has a "crop_bbox".
    plate_color    : BGR colour for the plate geometry.

    Boxes are read as COCO [x, y, width, height] in image coordinates.
    "crop_bbox" is read in the same format, as written by save_json().

    Returns the list of written file paths.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Case-insensitive lookup: UPPER-CASE STEM -> file name in the folder
    available = {}
    for file in os.listdir(images_folder):
        if file.lower().endswith(image_extensions):
            available[file.rsplit(".", 1)[0].strip().upper()] = file

    by_image = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(output_folder, exist_ok=True)
    written, skipped = [], []

    for img_info in data["images"]:
        file_name = img_info["file_name"]
        stem = file_name.rsplit(".", 1)[0]

        src = os.path.join(images_folder, file_name)
        if not os.path.isfile(src):
            found = available.get(stem.strip().upper())
            src = os.path.join(images_folder, found) if found else None

        image = cv2.imread(src) if src else None
        if image is None:
            skipped.append(file_name)
            continue

        # -- plate: crop rectangle + inscribed circle ---------------------
        crop = img_info.get("crop_bbox")
        if show_plate and crop is not None:
            cx, cy, cw, ch = crop
            px1, py1 = int(round(cx)), int(round(cy))
            px2, py2 = int(round(cx + cw)), int(round(cy + ch))
            cv2.rectangle(image, (px1, py1), (px2, py2), plate_color, thickness)
            cv2.circle(image,
                       ((px1 + px2) // 2, (py1 + py2) // 2),
                       min(px2 - px1, py2 - py1) // 2,
                       plate_color, thickness)

        # -- colonies -----------------------------------------------------
        for ann in by_image.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            if show_scores and "score" in ann:
                text = "{:.2f}".format(ann["score"])
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1),
                              color, -1)
                cv2.putText(image, text, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thickness, cv2.LINE_AA)

        out_path = os.path.join(output_folder, f"{stem}{ext}")
        cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(out_path)

    print(f"Images written: {len(written)} of {len(data['images'])}")
    if skipped:
        print(f"Skipped (image not found or unreadable): {skipped}")
    return written

def load_csv(
    path: str,
    key_type=str,
    has_header: bool = True,
    id_col: int | str = 0,
    count_col: int | str = 1,
) -> dict:
    """
    Reads a csv and returns {image_id: count}.

    key_type   : type applied to image_id (str or int).
    has_header : skip the first row if True.
    id_col     : column holding the image_id (default: first column).
    count_col  : column holding the count (default: second column).

    Image ids are stripped and converted to upper case before key_type is
    applied, so that matching is case-insensitive.
    Columns may be given as a zero-based index (int) or as a header name (str).
    Header names require has_header=True. Counts are returned as float.
    """
    counts = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None) if has_header else None

        def resolve(col):
            if isinstance(col, int):
                return col
            if header is None:
                raise ValueError("Column names require has_header=True")
            names = [h.strip() for h in header]
            if col not in names:
                raise ValueError(f"Column '{col}' not found in header: {names}")
            return names.index(col)

        id_idx, count_idx = resolve(id_col), resolve(count_col)

        for row in reader:
            if not row:                        # skip blank lines
                continue
            image_id = key_type(row[id_idx].strip().upper())
            if image_id in counts:
                raise ValueError(f"Duplicate image_id: {image_id}")
            counts[image_id] = float(row[count_idx])
    return counts

def save_csv(
    counts: dict,
    path: str,
    id_header: str = "image_id",
    count_header: str = "count",
    has_header: bool = True,
):
    """
    Writes a dictionary {image_id: count} to a CSV file.

    counts     : dictionary containing {image_id: count}.
    path       : output CSV file path.
    id_header  : header name for the image_id column.
    count_header: header name for the count column.
    has_header : write column headers if True.

    Image IDs are converted to strings.
    Counts are written as floats.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if has_header:
            writer.writerow([id_header, count_header])

        for image_id, count in counts.items():
            writer.writerow([str(image_id).strip(), float(count)])
            
        

def sweep_thresholds(prediction: dict, truth: dict,
                     start: float, stop: float, step: float,
                     min: int, max: int) -> tuple:
    """
    Calculates sMAPE for every score threshold from start to stop (inclusive)
    in increments of step.

    prediction : output of load_scores()
    truth    : output of load_counts()
    min, max : inclusive integer limits applied to the ground-truth counts.

    Returns (best_threshold, best_smape, results), where results is
    {threshold: smape}. If several thresholds share the lowest sMAPE,
    the lowest threshold is returned.
    """
    if step <= 0:
        raise ValueError("'step' must be positive.")

    n_steps = math.floor((stop - start) / step + 1e-9)    # tolerance for float error
    results = {}

    for i in range(n_steps + 1):
        threshold = round(start + i * step, 10)           # avoids 0.30000000000000004
        pred = count_above(prediction, threshold)
        results[threshold] = smape(pred, truth, min, max, verbose=False)

    best_threshold = sorted(results, key=results.get)[0]
    
    # One verbose call at the best threshold shows the matched-sample counts
    smape(count_above(prediction, best_threshold), truth, min, max)
    
    
    return best_threshold, results[best_threshold], results


def refine_threshold(prediction: dict, truth: dict,
                     start: float, stop: float,
                     coarse_step: float, fine_step: float,
                     min: int, max: int) -> tuple:
    """
    Two-stage search for the score threshold with the lowest sMAPE.

    Stage 1: coarse sweep over [start, stop] with coarse_step.
    Stage 2: fine sweep over [best - coarse_step, best + coarse_step],
             clipped to [start, stop], with fine_step.

    prediction : output of load_scores()
    truth    : output of load_counts()
    min, max : inclusive integer limits applied to the ground-truth counts.

    Returns (best_threshold, best_smape, results), where results holds
    the thresholds of both stages with their sMAPE, sorted by threshold.
    """
    if fine_step >= coarse_step:
        raise ValueError("'fine_step' must be smaller than 'coarse_step'.")

    coarse = sweep_thresholds(prediction, truth, start, stop, coarse_step, min, max)
    best = sorted(coarse, key=coarse.get)[0]

    lo = best - coarse_step
    if lo < start:
        lo = start
    hi = best + coarse_step
    if hi > stop:
        hi = stop

    fine = sweep_thresholds(prediction, truth, lo, hi, fine_step, min, max)

    results = dict(sorted({**coarse, **fine}.items()))
    best = sorted(results, key=results.get)[0]
    return best, results[best], results






        
def save_smape(results: dict, path: str) -> None:
    """
    Saves sMAPE for every threshold as a csv.

    results : {threshold: smape}, as returned by sweep_thresholds()
              or refine_threshold().
    path    : output csv file.

    Columns: threshold, smape. Rows are sorted by threshold.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "smape"])
        for threshold in sorted(results):
            writer.writerow([threshold, results[threshold]])

def evaluate_thresholds (pred_path: str, truth_path: str,
                        start: float, stop: float, step: float,
                        min: int, max: int,
                        csv_path: str = None,
                        key_type=str, has_header: bool = True,
                        id_col: int | str = 0, count_col: int | str = 1) -> tuple:
    """
    Loads predictions and ground truth, sweeps score thresholds, prints the
    best one and optionally saves the results.

    pred_path  : COCO-format json with a score for every annotation.
    truth_path : csv with the image id and the count.
    min, max   : inclusive integer limits applied to the ground-truth counts.
    csv_path   : if given, sMAPE for every threshold is saved to this csv.
    plot_path  : if given, the sMAPE plot is saved to this file.
                 Nothing is plotted if it is None.
    key_type, has_header, id_col, count_col : passed to load_csv().

    Returns (best_threshold, best_smape, results, prediction, truth).
    """
    prediction = load_json(pred_path)
    truth = load_csv(truth_path, key_type=key_type, has_header=has_header,
                     id_col=id_col, count_col=count_col)

    best_threshold, best_smape, results = sweep_thresholds(
        prediction, truth, start, stop, step, min, max)

    print(f"Best threshold: {best_threshold}")
    print(f"Best sMAPE: {best_smape:.2f}")
    
    filtered_pred = count_above(prediction, best_threshold)
    ids = ids_above(prediction, best_threshold)

    if csv_path:
        save_smape(results, csv_path)


    return {
        "best_threshold": best_threshold,
        "best_smape": best_smape,
        "results": results,
        "pred": filtered_pred,               # {sample name: count} at the best threshold
        "ids": ids,                 # {sample name: [annotation ids]} at the best threshold
        "truth": truth,
    }







# %% OLD

def evaluate_thresholds_old(pred_path: str, truth_path: str,
                        start: float, stop: float, step: float,
                        min: int, max: int,
                        plot_path: str = None,
                        key_type=str, has_header: bool = True) -> tuple:
    """
    Loads predictions and ground truth, sweeps score thresholds and prints
    the best one.

    pred_path  : COCO-format json with a score for every annotation.
    truth_path : csv with image_id in the first column and count in the second.
    min, max   : inclusive integer limits applied to the ground-truth counts.
    key_type   : type applied to image_id in the csv (int or str).
    has_header : skip the first csv row if True.

    Returns (best_threshold, best_smape, results, pred, truth).
    pred  : {sample name: count} at the best threshold.
    truth : {sample name: count}, as loaded from the csv.
    """
    prediction = load_json(pred_path)
    truth = load_csv(truth_path, key_type=key_type, has_header=has_header)

    best_threshold, best_smape, results = sweep_thresholds(
        prediction, truth, start, stop, step, min, max)

    pred = count_above(prediction, best_threshold)

    # One verbose call at the best threshold shows the matched-sample counts
    smape(pred, truth, min, max)

    print(f"Best threshold: {best_threshold}")
    print(f"Best sMAPE: {best_smape:.2f}")

    return best_threshold, best_smape, results, pred, truth

def load_scores_csv(folder: str) -> dict:
    """
    Reads every detection csv in a folder (one csv per image, with a 'score'
    column) and returns {file_name_without_extension: [(detection_id, score), ...]}.
    detection_id is the row index within the file.
    """
    by_image = {}
    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith(".csv"):
            continue
        name = file.rsplit(".", 1)[0]
        with open(os.path.join(folder, file), newline="", encoding="utf-8") as f:
            by_image[name] = [(i, float(row["score"]))
                              for i, row in enumerate(csv.DictReader(f))]
    return by_image

def evaluate_thresholds_old2(pred_path: str, truth_path: str,
                        start: float, stop: float, step: float,
                        min: int, max: int,
                        plot_path: str = None,
                        key_type=str, has_header: bool = True) -> tuple:
    """
    Loads predictions and ground truth, sweeps score thresholds, prints the
    best one and plots sMAPE against threshold.

    pred_path  : COCO-format json with a score for every annotation.
    truth_path : csv with image_id in the first column and count in the second.
    min, max   : inclusive integer limits applied to the ground-truth counts.
    plot_path  : if given, the plot is saved to this file instead of being shown.
    key_type   : type applied to image_id in the csv (int or str).
    has_header : skip the first csv row if True.

    Returns (best_threshold, best_smape, results).
    """
    prediction = load_scores_csv(pred_path)
    truth = load_csv(truth_path, key_type=key_type, has_header=has_header)

    best_threshold, best_smape, results = sweep_thresholds(
        prediction, truth, start, stop, step, min, max)

    # One verbose call at the best threshold shows the matched-sample counts
    smape(count_above(prediction, best_threshold), truth, min, max)

    print(f"Best threshold: {best_threshold}")
    print(f"Best sMAPE: {best_smape:.2f}")


    return best_threshold, best_smape, results

# %% CLI STANDALONE EXECUTION

if __name__ == "__main__":
    import argparse
    from validation_report import build_report


    parser = argparse.ArgumentParser(
        description="Standalone threshold optimization and evaluation using sMAPE."
    )
    parser.add_argument("pred_path", type=str, help="Path to COCO-format JSON file with prediction scores.")
    parser.add_argument("truth_path", type=str, help="Path to ground-truth CSV file.")
    parser.add_argument("output_path", type=str, help="Path to output.")
    parser.add_argument("--start", type=float, default=0.1, help="Start threshold for sweeping (default: 0.1)")
    parser.add_argument("--stop", type=float, default=0.95, help="Stop threshold for sweeping (default: 0.95)")
    parser.add_argument("--step", type=float, default=0.005, help="Step size for threshold sweeping (default: 0.005)")
    parser.add_argument("--min", type=int, default=11, help="Minimum inclusive ground-truth count limit (default: 11)")
    parser.add_argument("--max", type=int, default=300, help="Maximum inclusive ground-truth count limit (default: 300)")
    parser.add_argument("--id-col", default=0, help="Column index or name for image ID in CSV (default: 0)")
    parser.add_argument("--count-col", default=1, help="Column index or name for count in CSV (default: 1)")
    parser.add_argument("--n_outliers", type=int, default=5, help="Number of highlighted outliers in report")

    args = parser.parse_args()
    
    csv_path = os.path.join(args.output_path, "smape.csv")
    report_path = os.path.join(args.output_path, "report.html")
    # Anchor template path relative to this script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "report_template.html")
    
    # Check extension of pred_path
    ext = os.path.splitext(args.pred_path)[1].lower()

    if ext == ".json":

        ev = evaluate_thresholds(
            pred_path=args.pred_path,
            truth_path=args.truth_path,
            start=args.start,
            stop=args.stop,
            step=args.step,
            min=args.min,
            max=args.max,
            csv_path=csv_path,
            id_col=args.id_col,
            count_col=args.count_col
        )
        
    elif ext == ".csv":
        truth = load_csv(args.truth_path, has_header=True, id_col=args.id_col, count_col=args.count_col)
        pred = load_csv(args.pred_path, has_header=True, id_col=args.id_col, count_col=args.count_col)
        best_smape = smape(pred, truth, min=args.min, max=args.max, verbose=True)
        
        ev = {
            "best_smape": best_smape,
            "pred": pred,
            "truth": truth
        }
    else:
        raise ValueError(f"Unsupported prediction file format: {ext}. Expected .json or .csv")
    
    build_report(ev, report_path, min=args.min, max=args.max,
                     n_outliers=args.n_outliers, 
                     template_path=template_path) 