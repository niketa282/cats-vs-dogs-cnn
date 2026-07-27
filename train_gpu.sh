#!/bin/bash -l

#SBATCH --job-name=catsdogs
#SBATCH --partition=l4,swarm_l4,a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=catsdogs-%j.log

cd "$SLURM_SUBMIT_DIR"
$HOME/.conda/envs/catsdogs-iridis/bin/python src/train.py
