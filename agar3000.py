# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 02:39:57 2025

@author: Admin
"""
# %%
import sys
import os
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import main.pre as pre
import main.col as col
from main.debug as debug
from main.down as down
# %%

"""
----------------------------------------------------
CONFIG
---------------------------------------------------
"""
input_dir="Nor"
margin=0.75
plate_threshold=251
colonies_treshold=170 # NOT INCLUDED AS ARGUMENT YET!!!
alpha=1.4
beta=50
n_stripes=5


# %%

"""
----------------------------------------------------
PARSER
---------------------------------------------------
"""
parser = argparse.ArgumentParser(description="Run colony detection on a specified image folder.")
parser.add_argument("input_dir", help="Path to the folder containing images")
args = parser.parse_args()

input_dir = args.input_dir

if not os.path.isdir(input_dir):
    print(f"Error: '{input_dir}' is not a valid directory.")
    sys.exit(1)

# %%
"""
----------------------------------------------------
COLONIES DETECTION AND FEATURES EXTRUCTION
----------------------------------------------------
"""
pre.organize_photos(input_dir)
input_dir = os.path.join(input_dir, 'results')

pre.pre_all(input_dir, bb=False, hc=False, tra=False)


"""
check = True
while check:
    response = input("Preprocessing is done. Check if it's correct before the time-consuming detection step. Are you ready to proceed? [y/n]: ").strip().lower()
    
    if response not in ['y', 'yes', 'n', 'no']:
        print("Please enter 'y' or 'n'.")
    elif response in ['n', 'no']:
        print("Aborting script.")
        sys.exit(0)
    else:
        check = False
"""


col.detect_all_colonies(
    input_dir, 
)

"""
plot_colony_blobs(input_dir)

"""

# %%
"""
----------------------------------------------------
STRIPES DETECTION
----------------------------------------------------
"""

"""
linelib.all_kmeans_pc1(input_dir)
linelib.all_mix_polinom(input_dir)
"""
col.combine_colonies_csv(input_dir)

'''
summary_stripes(root_directory, cluster_column="Stripe")
summary_stripes(root_directory, cluster_column="Stripe_Kmean")
summary_stripes(root_directory, cluster_column="Stripe_Polreg")
'''
# %%
"""
----------------------------------------------------
VALIDATION
----------------------------------------------------
"""
"""
validate()
"""