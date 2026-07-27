#!/bin/bash -l

#SBATCH --job-name=catsdogs-eval
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=catsdogs-%j.log

cd "$SLURM_SUBMIT_DIR"
$HOME/.conda/envs/catsdogs-iridis/bin/python src/eval.py
