#!/bin/bash

#SBATCH --job-name=a1_pt4_tests
#SBATCH --output=./log/a1_pt4_tests_%j.out
#SBATCH --error=./log/a1_pt4_tests_%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

source "/data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate"

python3 ~/ml4nlp/assignment_01/V1.2/A1_pt4_tests.py --num_train_epochs 5
# srun python3 ~/ml4nlp/assignment_01/V1.2/A1_pt4_tests.py