# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:57:08 2025

@author: csaw7183
"""
# %%
import os
import csv
# %%
def summarize_colonies(csv_folder, output_path="RESULTS.csv"):
    """
    Summarize colony counts from CSV files in a folder.
    Only processes CSVs that have a matching .jpg file (same name) in the same folder.
    
    Parameters:
    - csv_folder (str): folder containing CSV and JPG files
    - output_path (str): path to save the summary CSV (default "RESULTS.csv")
    """
    summary = []
    for fname in sorted(os.listdir(csv_folder)):
        if not fname.lower().endswith(".csv"):
            continue
        plate_name = os.path.splitext(fname)[0]
        img_path = os.path.join(csv_folder, plate_name + ".jpg")
        if not os.path.exists(img_path):
            continue
        csv_path = os.path.join(csv_folder, fname)
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_count = sum(1 for _ in reader)
        summary.append((plate_name, row_count))
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plate", "Colonies"])
        writer.writerows(summary)
    print("-----------------------------------------------------")
    print("RESULTS:")
    with open(output_path, "r", encoding="utf-8") as f:
        rows = [line.rstrip().split(",") for line in f if line.strip()]
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  " + "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    
    print(f"Summary CSV-table saved to: {output_path}")

