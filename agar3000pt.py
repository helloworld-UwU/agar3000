# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:14:26 2026

@author: Admin
"""
print ("")
import argparse
import os
import glob
import cv2
from datetime import datetime
def timestamp():
    return(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print (f"START: {timestamp()}")
print ("-----------------------------------------------------")

# %%
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%

import main.tiling as tiling
from main.detect_frcnn_pt import load_model, detect_on_tiles
from main.process_old2 import process_plate
from main.down import summarize_colonies

# %%
"""
## MEMORY DEBUGING 

import tracemalloc
import linecache
import gc

def print_top_memory(snapshot, limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics("lineno")
    print(f"\n=== Top {limit} memory allocations ===")
    for stat in top_stats[:limit]:
        frame = stat.traceback[0]
        print(f"{stat.size / 1024 / 1024:.2f} MB | {frame.filename}:{frame.lineno}")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")
    print(f"Total: {sum(s.size for s in top_stats) / 1024 / 1024:.2f} MB\n")
    
"""   
# %%
    

def run_folder_pipeline(input_folder, output_folder, model = "model/frcnn_norm.pt",
                        grid=(2, 2), overlap=0.2, 
                        tol=5, scale=1024, score=0.25, debug=False):
    """
    For each image in input_folder:
      plate = process_plate(img_path)
      tiles = make_tiles(plate.image, grid, overlap)
      detect_on_tiles_inplace(tiles, rcnn)
      resolve_duplicates_across_tiles(tiles, tol)
      draw all roi_global on plate.image and SAVE to output_folder/{sample_id}.png
      save tiles detections to output_folder/{sample_id}.csv
    """
    """   
    tracemalloc.start() ## MEMORY DEBUGING 
    snapshot_prev = None ## MEMORY DEBUGING 
    """  
    
    os.makedirs(output_folder, exist_ok=True)
    rcnn = load_model(model)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    img_paths = []
    
    for e in exts:
        img_paths.extend(glob.glob(os.path.join(input_folder, e)))
    
    print ("Found images:"+str(len(img_paths)))

    for img_path in sorted(img_paths):
        print ("-----------------------------------------------------")
        print(f"Processing: {os.path.basename(img_path)} ({timestamp()})")
        print ("-----------------------------------------------------")
        
        plate = None  
        tiles = None
        

        plate = process_plate(img_path, margin = 1)
        if debug:
            debug_output =  os.path.join(output_folder, "crop")
            os.makedirs(debug_output, exist_ok=True)
            out_img_path = os.path.join(debug_output, f"{plate.sample_id}.png")
            cv2.imwrite(out_img_path, plate.cropped)
        
        tiles = tiling.make_tiles(plate.cropped, grid=grid, overlap=overlap)
        
        print(f"Detecting started: {plate.sample_id} ({timestamp()})")
        
        tiles = detect_on_tiles(tiles, rcnn, size=scale, conf=0.25, rgb=False)
        print(f"Detecting finished: {plate.sample_id} ({timestamp()})")
        print ("-----------------------------------------------------")
        
        if debug:
            # ---- debuging mode ----
            debug_output =  os.path.join(output_folder, "dup")
            tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes", 
                              cols=None, figsize=(12, 12),
                              output_folder=debug_output, name=f"{plate.sample_id}_tiles")
            tiling.save_plate_tiles_to_csv(debug_output, tiles, name=f"{plate.sample_id}")

   
            
        tiling.resolve_duplicates_across_tiles(tiles, tol=tol, detections_key="colonies")
        if debug:
            # ---- debuging mode ----
            debug_output =  os.path.join(output_folder, "dedup")
            tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes", 
                              cols=None, figsize=(12, 12),
                              output_folder=debug_output, name=f"{plate.sample_id}_tiles")
            # ---- draw all global rois on full image ----
            # assumed BGR
            tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                             color=(0, 0, 255), thickness=2, figsize=(12, 12), 
                             name=plate.sample_id ,output_folder=debug_output)
            # ---- save tiles detections to CSV ----
            tiling.save_plate_tiles_to_csv(debug_output, tiles, name=f"{plate.sample_id}")
            print(f"Deduplication finished: {plate.sample_id} ({timestamp()})")
        
        tiling.filter_colonies_by_score(tiles, threshold=score, detections_key="colonies")
        tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                             color=(0, 0, 255), thickness=2, figsize=(12, 12), 
                             name=plate.sample_id ,output_folder=output_folder)
        tiling.save_plate_tiles_to_csv(output_folder, tiles, name=f"{plate.sample_id}")
        print(f"Scores filtering finished: {plate.sample_id} ({timestamp()})")
        
        
        """  
        # ── cleanup BEFORE snapshot so delta reflects what actually leaked ──
        sample_id = plate.sample_id  # save for print below
        del plate, tiles
        gc.collect()

        
        # ── memory diagnostics ──
        snapshot_curr = tracemalloc.take_snapshot()
        if snapshot_prev:
            print(f"\n=== MEMORY DELTA after {sample_id} ===")
            stats = snapshot_curr.compare_to(snapshot_prev, "lineno")
            for stat in stats[:10]:
                if stat.size_diff > 0:
                    print(f"+{stat.size_diff / 1024:.1f} KB | {stat.traceback[0]}")
        snapshot_prev = snapshot_curr
        print_top_memory(snapshot_curr)
        """  
     


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_folder", type=str)
    p.add_argument("output_folder", type=str)
    p.add_argument("--model", type=str, default="model/frcnn_norm.pt")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--overlap", type=float, default=0.1)
    p.add_argument("--tol", type=int, default=3)
    p.add_argument("--scale", type=int, default=512)
    p.add_argument("--score", type=float, default=0.10)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--validation", action="store_true")
    args = p.parse_args()
    
    output_folder=args.output_folder

    run_folder_pipeline(
        input_folder=args.input_folder,
        output_folder=output_folder,
        model=args.model,
        grid=(args.rows, args.cols),
        overlap=args.overlap,
        tol=args.tol,
        scale=args.scale,
        score=args.score,
        debug=args.debug
    )
    summarize_colonies(output_folder, f"{output_folder}/sum.csv")
    
    if args.validation:
        ## VALIDATION PART
        import subprocess
        import sys
        
        # Paths and arguments
        r_script = "main/validation.R"
        input_csv = f"{output_folder}/sum.csv"
        output_html = f"{output_folder}/validation_report.html"
        ref_csv = "ref.csv"
        template_rmd = "main/report_template.Rmd"
        
        # Build the command
        cmd = [
            "Rscript",
            r_script,
            "-i", input_csv,
            "-o", output_html,
            "-r", ref_csv,
            "-t", template_rmd
        ]
        
        # Run the command
        try:
            result = subprocess.run(
                cmd,
                check=True,
            #    stdout=subprocess.PIPE,
            #    stderr=subprocess.PIPE,
            #    universal_newlines=True
            )
            #print("R script output:\n", result.stdout)
            #print("R script errors (if any):\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print("R script failed with exit code", e.returncode)
            #print("Output:\n", e.output)
            #print("Errors:\n", e.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
    

            
    print ("-----------------------------------------------------")
    print (f"FINISH: {timestamp()}")
    