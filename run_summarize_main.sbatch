#!/bin/bash
#SBATCH --account=p53198
#SBATCH --job-name=summain
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Uses vLLM if installed (bash setup_vllm.sh, once). Without it the job still runs but falls
# back to plain transformers: much slower, and the summarizer runs one call at a time.
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

python summarize_calls.py
python main.py --summaries "$@"
