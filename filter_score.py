# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:55:51 2026

@author: Admin
"""

import argparse
from main.debug import scores_filter
from main.down import summarize_colonies

p = argparse.ArgumentParser()
p.add_argument("input_folder", type=str)
p.add_argument("score", type=float)
p.add_argument("--ref", type=str, default="ref.csv")
p.add_argument("--mode", type=str, default="Average_Human")
p.add_argument("--ld", type=float, default=None)
p.add_argument("--hd", type=float, default=None)
p.add_argument("--val", action="store_true")
p.add_argument("--out_photos", action="store_true")


args = p.parse_args()



input_csv = f"{args.input_folder}/dedup"
input_photo =  f"{args.input_folder}/crop"
output_folder = f"{args.input_folder}/{str(args.score)}"
mode = args.mode
args.ld

if args.out_photos:
    debug_output = output_folder
else:
    debug_output = None 



if args.ld is not None and args.hd is not None:
	scores_filter(input_csv, 
              input_photo,
              output_folder,
              debug_output,
              args.score,
	      (args.ld, args.hd))
else:
	scores_filter(input_csv, 
              input_photo,
              output_folder,
              debug_output,
              args.score)
    
summarize_colonies(output_folder, f"{output_folder}/sum.csv")

if args.val:
    ## VALIDATION PART
    import subprocess
    import sys
    
    # Paths and arguments
    r_script = "main/validation.R"
    input_csv = f"{output_folder}/sum.csv"
    output_html = f"{output_folder}/validation_report.html"
    ref_csv = args.ref
    template_rmd = "main/report_template.Rmd"
    
    # Build the command
    cmd = [
        "Rscript",
        r_script,
        "-i", input_csv,
        "-o", output_html,
        "-r", ref_csv,
        "-t", template_rmd,
        "--mode", mode
    ]
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True
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