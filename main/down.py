# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:57:08 2025

@author: csaw7183
"""
# %%
import os
import csv
# %%
def summarize_colonies(csv_folder, output_path="sum.csv"):
    """
    Go through all CSVs in csv_folder and create a summary CSV with:
    Plate, Colonies

    Parameters:
    - csv_folder (str): folder containing CSV files
    - output_path (str): path to save summary CSV (default "sum.csv")
    """
    summary = []

    for fname in sorted(os.listdir(csv_folder)):
        if not fname.lower().endswith(".csv"):
            continue

        plate_name = os.path.splitext(fname)[0]
        csv_path = os.path.join(csv_folder, fname)

        # count rows (skip header)
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            row_count = sum(1 for _ in reader)

        summary.append((plate_name, row_count))

    # write summary CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plate", "Colonies"])
        writer.writerows(summary)

    print(f"Summary CSV saved to: {output_path}")

