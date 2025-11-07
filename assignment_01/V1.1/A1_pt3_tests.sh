#!/bin/bash

#SBATCH --job-name=a1_pt3_tests
#SBATCH --output=a1_pt3_tests.log
#SBATCH --error=a1_pt3_tests.err
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1

source "/data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate"

python3 ~/assignment1/A1_pt3_tests.py