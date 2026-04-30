# -*- coding: utf-8 -*-

from datetime import datetime
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Global import
import argparse
import os
import gc
import glob
import sys
import cv2

# Lockal import
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import main.tiling as tiling
from main.detect_frcnn_onnx import load_model, detect_on_tiles
from main.process import process_plate
from main.down import summarize_colonies


# ----------------------------------------------------------------------
# Logging — tees stdout to output_folder/pipeline_<timestamp>.log
# ----------------------------------------------------------------------

class _Tee:
    """Writes to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stdout, name)


def setup_logging(output_folder):
    log_name = f"agar3000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_folder, log_name)
    return _Tee(log_path), log_path


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def run_folder_pipeline(input_folder, output_folder, model="model/frcnn_norm.pt",
                        grid=(2, 2), overlap=0.2,
                        tol=5, scale=1024, score=0.25, extra=False,
                        mem_debug=False, no_crop=False, margin = 1, score_regression=None):

    os.makedirs(output_folder, exist_ok=True)

    try:
        EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        rcnn = load_model(model)
        
        if os.path.isfile(input_folder):
            if os.path.splitext(input_folder)[1].lower() not in EXTS:
                raise ValueError(f"Unsupported file type: {input_folder}")
            img_paths = [input_folder]
        else:
            img_paths = []
            for e in (f"*{ext}" for ext in EXTS):
                img_paths.extend(glob.glob(os.path.join(input_folder, e)))

        print(f"Found images: {len(img_paths)}")

        for img_path in sorted(img_paths):
            print("-----------------------------------------------------")
            print("-----------------------------------------------------")
            print(f"PROCESSING: {os.path.basename(img_path)} ({timestamp()})")
            print("-----------------------------------------------------")

            plate = process_plate(img_path, margin=margin)
            if no_crop:
                plate.cropped = plate.image
            elif extra:
                ext_out = os.path.join(output_folder, "crop")
                os.makedirs(ext_out, exist_ok=True)
                cv2.imwrite(os.path.join(ext_out, f"{plate.sample_id}.png"), plate.cropped)

            tiles = tiling.make_tiles(plate.cropped, grid=grid, overlap=overlap)

            print(f"Detection started:  {plate.sample_id} ({timestamp()})")
            tiles = detect_on_tiles(tiles, rcnn, size=scale, conf=score, normalize=True, rgb=False)
            print(f"Detection finished: {plate.sample_id} ({timestamp()})")
            print("-----------------------------------------------------")

            if extra:
                ext_out = os.path.join(output_folder, "dup")
                tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes",
                                                 cols=None,
                                                 output_folder=ext_out,
                                                 name=f"{plate.sample_id}_tiles")
                tiling.save_plate_tiles_to_csv(ext_out, tiles, name=plate.sample_id)

            tiling.resolve_duplicates_across_tiles(tiles, tol=tol, detections_key="colonies")

            if extra:
                ext_out = os.path.join(output_folder, "dedup")
                tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes",
                                                 cols=None,
                                                 output_folder=ext_out,
                                                 name=f"{plate.sample_id}_tiles")
                tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                                            color=(0, 0, 255), thickness=2,
                                            name=plate.sample_id, output_folder=ext_out)
                tiling.save_plate_tiles_to_csv(ext_out, tiles, name=plate.sample_id)
            print(f"Deduplication finished: {plate.sample_id} ({timestamp()})")
            
            tiling.filter_colonies_by_score(tiles, threshold=score, detections_key="colonies", 
                                            score_regression=score_regression)
            
            tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                                        color=(0, 0, 255), thickness=2,
                                        name=plate.sample_id, output_folder=output_folder)
            tiling.save_plate_tiles_to_csv(output_folder, tiles, name=plate.sample_id)
            print(f"Results saved: {plate.sample_id} ({timestamp()})")

            del plate, tiles
            gc.collect()

    except Exception as e:
        print(f"ERROR: {e}")
        raise


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input",  type=str,
                   help="path to the folder with images of plates, or single image")
    p.add_argument("output_folder", type=str,
                   help="place for the counting results")
    p.add_argument("-t", action="store_true",
                   help="use separate method if your plates are transilluminated")
    p.add_argument("-b", action="store_true",
                   help="let program think more to get better precision")
    p.add_argument("--extra", action="store_true", help="put for more intermediate output")
    
    p.add_argument("--model",      type=str,   default="model/frcnn_lr.onnx",  help=argparse.SUPPRESS)
    p.add_argument("--rows",       type=int,   default=3,                      help=argparse.SUPPRESS)
    p.add_argument("--cols",       type=int,   default=3,                      help=argparse.SUPPRESS)
    p.add_argument("--overlap",    type=float, default=0.1,                    help=argparse.SUPPRESS)
    p.add_argument("--tol",        type=int,   default=3,                      help=argparse.SUPPRESS)
    p.add_argument("--scale",      type=int,   default=512,                    help=argparse.SUPPRESS)
    p.add_argument("--score",      type=float, default=0.20,                   help=argparse.SUPPRESS)    
    p.add_argument("--mem-debug",  action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--validation", action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--ref",        type=str,   default="ref.csv",              help=argparse.SUPPRESS)
    p.add_argument("--no-crop",    action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--margin",     type=float, default=1,                      help=argparse.SUPPRESS)
    p.add_argument("--ld",         type=float, default=0.05,                   help=argparse.SUPPRESS)
    p.add_argument("--hd",         type=float, default=-0.10,                  help=argparse.SUPPRESS)
    args = p.parse_args()
    
    regression = "y = -0.000517x + 0.255172"
    
    if args.t:
        args.model = "model/frcnn_hr.onnx"
        args.score = 0.20 
        regression = "y = -0.000517x + 0.255172"
    if args.b:
        args.rows = 4
        args.cols = 4
        args.score = 0.20
        regression = "y = -0.000517x + 0.255172"

    # Start logging — everything from here on is captured to the log file
    os.makedirs(args.output_folder, exist_ok=True)
    tee, log_path = setup_logging(args.output_folder)

    try:
        print(f"START: {timestamp()}")
        print("=====================================================")
        print("Agar3000 v0.2")
        print("=====================================================")
        print("Configuration:")
        print(f"  Input                   : {args.input}")
        print(f"  Output folder           : {args.output_folder}")
        print(f"  Mode                    : {'Transillumination' if args.t else 'Surface illumination'}")
        print(f"  Model                   : {args.model}")
        if args.no_crop:
            print(f"  No-crop mode        : {args.no_crop}")
        else:
            print(f"  Plate margin offset : {args.margin}")
        print(f"  Grid                    : {args.rows} rows x {args.cols} cols")
        print(f"  Overlap                 : {args.overlap}")
        print(f"  Tolerance               : {args.tol}")
        print(f"  Scale                   : {args.scale}")
        print(f"  Min. score filter       : {args.score}")
        print(f"  Corection regression    : {regression}")
        print(f"  Extra mode              : {args.extra}")
        print("  --- Validation ---")
        print(f"  Validation mode         : {args.validation}")
        print(f"  Reference CSV           : {args.ref}")
        print("=====================================================")
        print(f"Log: {log_path}")
        print("-----------------------------------------------------")

        run_folder_pipeline(
            input_folder  = args.input,
            output_folder = args.output_folder,
            model         = args.model,
            grid          = (args.rows, args.cols),
            overlap       = args.overlap,
            tol           = args.tol,
            scale         = args.scale,
            score         = args.score,
            extra         = args.extra,
            mem_debug     = args.mem_debug,
            no_crop       = args.no_crop,
            margin        = args.margin,
            score_regression = (args.ld, args.hd)
        )

        
        summarize_colonies(args.output_folder, f"{args.output_folder}/sum.csv")

        if args.validation:
            import subprocess
            cmd = [
                "Rscript", "main/validation.R",
                "-i", f"{args.output_folder}/sum.csv",
                "-o", f"{args.output_folder}/validation_report.html",
                "-r", args.ref,
                "-t", "main/report_template.Rmd",
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"R script failed with exit code {e.returncode}")
                sys.exit(1)

        print("-----------------------------------------------------")
        print(f"FINISH: {timestamp()}")

    finally:
        tee.close()


if __name__ == "__main__":
    main()