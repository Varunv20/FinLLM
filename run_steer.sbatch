#!/bin/bash
#SBATCH --account=p53198
#SBATCH --job-name=steer
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Emotion steering vectors (steer_main.py). Computes the vectors once with transformers, then
# runs on vLLM like main.py; results save after every batch and a resubmitted job picks up
# where it stopped.
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

python steer_main.py --summaries "$@"
