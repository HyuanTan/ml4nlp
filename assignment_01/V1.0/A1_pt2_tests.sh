#!/bin/bash

#SBATCH --job-name=a1_pt2_tests
#SBATCH --output=a1_pt2_tests.log     
#SBATCH --error=a1_pt2_tests.err   
#SBATCH --partition=short   
#SBATCH --time=00:10:00               
#SBATCH --cpus-per-task=1             

VENV_PATH="/data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate"
source $VENV_PATH

python3 ~/nlp/A1_pt2_tests.py