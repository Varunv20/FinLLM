#!/bin/bash
#SBATCH --account=p53198
#SBATCH --job-name=stockemo
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Fine-tune RoBERTa on StockEmotions (12 investor emotions = the 11 prompt emotions + ambiguous).
# Downloads the dataset from GitHub the first time. Model and test results: models/stockemotions-roberta/
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

[ -d data_collection/StockEmotions ] || \
    git clone --depth 1 https://github.com/adlnlp/StockEmotions.git data_collection/StockEmotions

python data_collection/train_stockemotions.py "$@"
