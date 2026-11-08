#!/bin/bash
#SBATCH --account=p53198
#SBATCH --job-name=main120b
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# gpt-oss-120b on the summaries, same calls and prompts as the 20b run. Its weights are ~65 GB,
# more than fits next to the existing cache in home, so the Hugging Face cache goes to scratch.
# Results: anon_sum_openai/gpt-oss-120b_<neutral|0..10>.pkl (resubmit to continue).
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"

python main.py --model openai/gpt-oss-120b --summaries \
    --calls-file runs/anon_openai_gpt-oss-20b_2018_calls.csv "$@"
