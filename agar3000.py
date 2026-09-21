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

def run_folder_pipeline(input_path, output_path, model="model/frcnn_norm.pt",
                        grid=(2, 2), overlap=0.2,
                        tol=5, scale=1024, score=0.25, extra=False,
                        mem_debug=False, no_crop=False, margin = 1, score_regression=None):

    

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path/file does not exist: {input_path}")
                
        EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        
        if os.path.isfile(input_path):
            img_paths = [input_path]        
        
        else:
            img_paths = []
            for e in (f"*{ext}" for ext in EXTS):
                img_paths.extend(glob.glob(os.path.join(input_path, e)))
                
        
        rcnn = load_model(model)
        
        for img_path in sorted(img_paths):
            print("-----------------------------------------------------")
            print("-----------------------------------------------------")
            print(f"PROCESSING: {os.path.basename(img_path)} ({timestamp()})")
            print("-----------------------------------------------------")

            plate = process_plate(img_path, margin=margin)
            if no_crop:
                plate.cropped = plate.image
            elif extra:
                ext_out = os.path.join(output_path, "crop")
                os.makedirs(ext_out, exist_ok=True)
                cv2.imwrite(os.path.join(ext_out, f"{plate.sample_id}.png"), plate.cropped)

            tiles = tiling.make_tiles(plate.cropped, grid=grid, overlap=overlap)

            print(f"Detection started:  {plate.sample_id} ({timestamp()})")
            tiles = detect_on_tiles(tiles, rcnn, size=scale, conf=score, normalize=True, rgb=False)
            print(f"Detection finished: {plate.sample_id} ({timestamp()})")
            print("-----------------------------------------------------")
            print(f"Deduplication started:  {plate.sample_id} ({timestamp()})")

            if extra:
                ext_out = os.path.join(output_path, "dup")
                tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes",
                                                 cols=None,
                                                 output_folder=ext_out,
                                                 name=f"{plate.sample_id}_tiles")
                tiling.save_plate_tiles_to_csv(ext_out, tiles, name=plate.sample_id)

            tiling.resolve_duplicates_across_tiles(tiles, tol=tol, detections_key="colonies")

            if extra:
                ext_out = os.path.join(output_path, "dedup")
                tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes",
                                                 cols=None,
                                                 output_folder=ext_out,
                                                 name=f"{plate.sample_id}_tiles")
                tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                                            color=(0, 0, 255), thickness=2,
                                            name=plate.sample_id, output_folder=ext_out)
                tiling.save_plate_tiles_to_csv(ext_out, tiles, name=plate.sample_id)
            print(f"Deduplication finished: {plate.sample_id} ({timestamp()})")
            print("-----------------------------------------------------")
            
            tiling.filter_colonies_by_score(tiles, threshold=score, detections_key="colonies", 
                                            score_regression=score_regression)
            
            tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                                        color=(0, 0, 255), thickness=2,
                                        name=plate.sample_id, output_folder=output_path)
            tiling.save_plate_tiles_to_csv(output_path, tiles, name=plate.sample_id)
            print(f"Results saved: {plate.sample_id} ({timestamp()})")

            del plate, tiles
            gc.collect()

    except Exception as e:
        print(f"ERROR: {e}")
        raise


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        epilog=(
            "Try it with the demo data:\n"
            "  Surface illuminated plates:  python agar3000.py demo demo/results\n"
            "  Transilluminated plates:     python agar3000.py demo demo_t/results -t"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input_path",  type=str,
                   help="path to the folder with plate images, or single image file")
    p.add_argument("output_path", type=str,
                   help="path for results")
    p.add_argument("-t", action="store_true",
                   help="use if your plates are trans-illuminated")
    p.add_argument("-b", action="store_true",
                   help="faster inference at the cost of a slight loss of precision")
    p.add_argument("--extra", action="store_true",
                   help="get more of intermediate output (slow)")
    p.add_argument("--no-crop", action="store_true",
                   help="skip plate cropping (use for non-circular plates)")

    # Crucial fix: Changed default=None so we can detect if the user actually passed a value
    p.add_argument("--model",      type=str,   default=None,    help=argparse.SUPPRESS)
    p.add_argument("--rows",       type=int,   default=None,    help=argparse.SUPPRESS)
    p.add_argument("--cols",       type=int,   default=None,    help=argparse.SUPPRESS)
    p.add_argument("--overlap",    type=float, default=0.1,     help=argparse.SUPPRESS)
    p.add_argument("--tol",        type=int,   default=3,       help=argparse.SUPPRESS)
    p.add_argument("--scale",      type=int,   default=512,     help=argparse.SUPPRESS)
    p.add_argument("--score",      type=float, default=None,    help=argparse.SUPPRESS)
    p.add_argument("--mem-debug",  action="store_true",         help=argparse.SUPPRESS)
    p.add_argument("--validation", action="store_true",         help=argparse.SUPPRESS)
    p.add_argument("--ref",        type=str,   default="ref.csv",help=argparse.SUPPRESS)
    p.add_argument("--margin",     type=float, default=1,       help=argparse.SUPPRESS)
    p.add_argument("--ld",         type=float, default=None,    help=argparse.SUPPRESS)
    p.add_argument("--hd",         type=float, default=None,    help=argparse.SUPPRESS)

    args = p.parse_args()

    # 1. define the matrix of defaults based on flags (b, t)
    # Format: (has_b, has_t) -> {defaults dict}
    DEFAULTS_MATRIX = {
        (False, False): {"ld": 0.05, "hd": -0.15, "score": 0.35, "rows": 4, "cols": 4, "model": "model/frcnn_lr.onnx"},
        (False, True):  {"ld": None,  "hd": None,   "score": 0.45, "rows": 4, "cols": 4, "model": "model/frcnn_hr.onnx"},
        (True, False):  {"ld": 0.05, "hd": -0.10, "score": 0.20, "rows": 3, "cols": 3, "model": "model/frcnn_lr.onnx"},
        (True, True):   {"ld": None,  "hd": None,   "score": 0.30, "rows": 3, "cols": 3, "model": "model/frcnn_hr.onnx"}
    }

    # 2. select the matching baseline dictionary based on active flags
    selected_defaults = DEFAULTS_MATRIX[(args.b, args.t)]

    # 3. apply defaults ONLY if the user left the argument completely blank (None)
    for key, default_value in selected_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_value)

    # 4. regression depends on the newly assigned or user values
    args.regression = tiling.regression_from_score(
        args.score, 
        low_delta=args.ld, 
        high_delta=args.hd
    )

    return args




def main():
    
    args = parse_args()

    # --- Input validation  ---
    EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    if not os.path.exists(args.input_path):
        print(f"ERROR: Input path/file does not exist: {args.input_path}")
        sys.exit(1)
    elif os.path.isfile(args.input_path):
        if os.path.splitext(args.input_path)[1].lower() not in EXTS:
            print(f"ERROR: Unsupported file type: {args.input_path}\n"
                  f"  Supported formats: {', '.join(sorted(EXTS))}")
            sys.exit(1)
    else:
        matches = []
        for e in (f"*{ext}" for ext in EXTS):
            matches.extend(glob.glob(os.path.join(args.input_path, e)))
        if not matches:
            print(f"ERROR: No supported images found in folder: {args.input_path}\n"
                  f"  Supported formats: {', '.join(sorted(EXTS))}")
            sys.exit(1)

    # Start logging — everything from here on is captured to the log file
    os.makedirs(args.output_path, exist_ok=True)
    tee, log_path = setup_logging(args.output_path)

    try:
        print(f"START: {timestamp()}")
        print("=====================================================")
        print("Agar3000 v0.2.2")
        print("=====================================================")
        print("Configurations:")
        print(f"  Input                   : {args.input_path}")
        print(f"  Output folder           : {args.output_path}")
        print(f"  Mode                    : {'Trans-illumination' if args.t else 'Epi-illumination'}")
        print(f"  Model                   : {args.model}")
        print(f"  Extra output            : {args.extra}")
        
        if args.no_crop:
            print(f"  No-crop mode        : {args.no_crop}")
        else:
            print("  --- Cropping ---")
            print(f"  Plate margin offset : {args.margin}")
        
        print("  --- Tiling ---")
        print(f"  Grid                    : {args.rows} rows x {args.cols} cols")
        print(f"  Overlap                 : {args.overlap}")
        print(f"  Tolerance               : {args.tol}")
        print(f"  Scale                   : {args.scale}")
        print(f"  Min. score filter       : {args.score}")
        print(f"  Corection regression    : {args.regression}")
        
        if args.validation:
            print("  --- Validation ---")
            print(f"  Validation mode         : {args.validation}")
            print(f"  Reference CSV           : {args.ref}")
        print("-----------------------------------------------------")
        print(f"Log: {log_path}")
        print("=====================================================")

        run_folder_pipeline(
            input_path    = args.input_path,
            output_path   = args.output_path,
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
        
        print("-----------------------------------------------------")
        print(f"FINISH: {timestamp()}")
        
        results_path = f"{args.output_path}/RESULTS.csv"
        summarize_colonies(args.output_path, results_path)

        if args.validation:
            import subprocess
            cmd = [
                "Rscript", "main/validation.R",
                "-i", f"{args.output_path}/sum.csv",
                "-o", f"{args.output_path}/validation_report.html",
                "-r", args.ref,
                "-t", "main/report_template.Rmd",
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"R script failed with exit code {e.returncode}")
                sys.exit(1)

    finally:
        tee.close()


if __name__ == "__main__":
    main()