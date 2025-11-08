#!/bin/bash

#SBATCH --job-name=a1_pt4_tests
#SBATCH --output=./log/a1_pt4_tests_%j.out
#SBATCH --error=./log/a1_pt4_tests_%j.err
#SBATCH --partition=long
#SBATCH --time=05:30:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1

source "/data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate"

# a1_output_LSTM, a1_output_GRU
python3 ~/ml4nlp/assignment_01/V1.2/A1_pt4_tests.py --num_train_epochs 20 --output_dir a1_output_LSTM
# srun python3 ~/ml4nlp/assignment_01/V1.2/A1_pt4_tests.py