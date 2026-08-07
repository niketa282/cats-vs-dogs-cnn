#!/bin/bash -l
#SBATCH --job-name=catsdogs-eval
#SBATCH --partition=l4,swarm_l4,a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=catsdogs-%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
$HOME/.conda/envs/catsdogs-iridis/bin/python src/eval.py
$HOME/.conda/envs/catsdogs-iridis/bin/python src/eval_transfer_learning.py