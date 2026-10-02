# -*- coding: utf-8 -*-

BANNER = (r"""
 █████╗   ██████╗  █████╗ ██████╗
██╔══██╗ ██╔════╝ ██╔══██╗██╔══██╗
███████║ ██║  ███╗███████║██████╔╝
██╔══██║ ██║   ██║██╔══██║██╔══██╗
██║  ██║ ╚██████╔╝██║  ██║██║  ██║
╚═╝  ╚═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

 ██████╗  ██████╗  ██████╗  ██████╗
 ╔═══██╝ ██╔═████╗██╔═████╗██╔═████╗
 █████╗  ██║██╔██║██║██╔██║██║██╔██║
 ╚═══██╗ ████╔╝██║████╔╝██║████╔╝██║
██████╔╝ ╚██████╔╝╚██████╔╝╚██████╔╝
╚═════╝   ╚═════╝  ╚═════╝  ╚═════╝
""")

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
import csv
import shutil


# Lockal import
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from main.data import Plate
import main.tiling as tiling
import main.render as render
from main.detect_frcnn_onnx import load_model, detect_on_tiles
from main.plate import process_plate
from main.dedup import resolve_duplicates
from main.filtering import filter_by_score, regression_from_score


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

def run_pipeline(input_path, output_path, model="model/frcnn_norm.pt",
                        grid=(2, 2), overlap=0.2,
                        tol=5, scale=1024, score=0.25, extra=False,
                        mem_debug=False, no_crop=False, margin = 1, score_regression=None):

     
    counts = []
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
        
        os.makedirs(output_path, exist_ok=True)
        
        
        
        sum_path = os.path.join(output_path, "SUM.csv")

        sum_file = open(sum_path, "w", newline="")
        sum_writer = csv.writer(sum_file)
        sum_writer.writerow(["Plate", "Colonies"])
        sum_file.flush()
        
        
        for img_path in sorted(img_paths):
            
            # --- (1) LOADING IMAGE -------------------------------------------
            
            print("-----------------------------------------------------")
            print("-----------------------------------------------------")
            print(f"PROCESSING: {os.path.basename(img_path)} ({timestamp()})")
            print("-----------------------------------------------------")

            image = cv2.imread(img_path)
            if image is None:
                print(f"ERROR: Could not load image at {img_path}")
                continue

            sample_id = os.path.splitext(os.path.basename(img_path))[0]
            
            extra_path = os.path.join(output_path, "extra", sample_id)

            plate = Plate(sample_id=sample_id, image_path=img_path, image=image)

            # --- (2) PLATE DETECTION -----------------------------------------

            if no_crop:
                plate.cropped = plate.image
            
            else:
                plate.cropped, plate.crop_bbox = process_plate(plate.image,
                                                           margin=margin)
            
                
            # --- (3) TILING --------------------------------------------------
            tiling.make_tiles(plate,
                              grid=grid,
                              overlap=overlap)
            
            if extra:
                render.save_geometry(plate, extra_path, 
                                     name="1_" + plate.sample_id + "_geometry")
                
                
            # --- (4) COLONY DETECTION ------------

            print(f"Detection started:  {plate.sample_id} ({timestamp()})")
            
            detect_on_tiles(plate.tiles, rcnn, size=scale, conf=score, normalize=True, rgb=True)
            
            if extra:
                render.show_tiles(plate, extra_path,
                  name="2_" + plate.sample_id + "_duplicated")
                render.save_csv(plate, extra_path,
                                name="2_" + plate.sample_id + "_duplicated")
                
            
            print(f"Detection finished: {plate.sample_id} ({timestamp()})")
            print("-----------------------------------------------------")
            print(f"Deduplication started:  {plate.sample_id} ({timestamp()})")



            resolve_duplicates(plate.tiles, tol=tol)

            if extra:
                render.show_tiles(plate, extra_path,
                                  name="3_" + plate.sample_id + "_deduplicated")
                render.save_csv(plate, extra_path,
                                name="3_" + plate.sample_id + "_deduplicated")
                render.show_colonies(plate, extra_path,
                                     name="4_" + plate.sample_id + "_joined")
                
                
            print(f"Deduplication finished: {plate.sample_id} ({timestamp()})")
            print("-----------------------------------------------------")
            
            filter_by_score(plate.tiles, threshold=score, score_regression=score_regression)
            
            render.save_json(plate, os.path.join(output_path, "predictions_json"))
            render.show_colonies(plate, os.path.join(output_path, "predictions"))
            
            n = plate.count
            counts.append((plate.sample_id, n))
            
            sum_writer.writerow([plate.sample_id, plate.count])
            sum_file.flush()
            os.fsync(sum_file.fileno())
            
            print(f"Results saved: {plate.sample_id} ({timestamp()})")

            del plate
            gc.collect()
            
        sum_file.close()
        
        rows = [["Plate", "Colonies"]] + [[p, str(n)] for p, n in counts]
        col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

        header, data_rows = rows[0], rows[1:]
        if len(data_rows) <= 10:
            print("RESULTS:")
            display_rows = rows
        else:
            print("First 10 plates:")
            display_rows = [header] + data_rows[:10]

        for row in display_rows:
            print("  " + "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))

        print(f"Summary saved to: {sum_path}")
        
        
        # merging json
        render.merge_json(os.path.join(output_path, "predictions_json"),
                          os.path.join(output_path, "predictions.json"))
        shutil.rmtree(os.path.join(output_path, "predictions_json"))

    except Exception as e:
        print(f"ERROR: {e}")
        raise
        
    finally:
        try:
            sum_file.close()
        except NameError:
            pass 

    return counts      
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
    args.regression = regression_from_score(
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
            
            
    # --- Avoid writing results into the input folder ---
    if os.path.abspath(args.input_path) == os.path.abspath(args.output_path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = os.path.join(args.output_path, f"agar3000_{stamp}")

    # Start logging — everything from here on is captured to the log file
    os.makedirs(args.output_path, exist_ok=True)
    tee, log_path = setup_logging(args.output_path)
    
    


    try:
        print(f"START: {timestamp()}")
        print("=====================================================")
        print(BANNER)
        print("                                         v0.2.3")
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

        run_pipeline(
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
        


        if args.validation:
            import subprocess
            cmd = [
                "Rscript", "main/validation.R",
                "-i", f"{args.output_path}/SUM.csv",
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