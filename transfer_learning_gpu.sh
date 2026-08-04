#!/bin/bash -l

#SBATCH --job-name=resnet18
#SBATCH --partition=l4,swarm_l4,a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=resnet18-%j.log

cd "$SLURM_SUBMIT_DIR"

$HOME/.conda/envs/catsdogs-iridis/bin/python src/transfer_learning.py