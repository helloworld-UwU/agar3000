#!/usr/bin/bash

#SBATCH --job-name agarrcnnTRAHC
#SBATCH --mail-user Vadym.Sokhan@student.uibk.ac.at
#SBATCH --mail-type ALL
#SBATCH --time 180     
#SBATCH --mem 10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output log-%x.%A-%a.out

. ~/.bashrc
conda activate agarrcnn

python demo.py Tra
