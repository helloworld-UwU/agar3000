# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:59:37 2026

@author: Admin
"""

import argparse
from main.debug import deduplication_filter


p = argparse.ArgumentParser()
p.add_argument("input_folder", type=str)
p.add_argument("tol", type=int)


args = p.parse_args()


input_csv = f"{args.input_folder}/dup"
input_photo =  f"{args.input_folder}/crop"
output_folder = f"{args.input_folder}/dedup_tol_{str(args.tol)}"

deduplication_filter(input_csv, 
              input_photo,
              output_folder,
              args.tol)