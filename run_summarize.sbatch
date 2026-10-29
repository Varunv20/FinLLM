#!/bin/bash
# Summarize transcripts on a GPU node (summarize_calls.py; Qwen3-30B-A3B-Instruct-2507-FP8 via
# vLLM). Submit from the FinLLM folder; anything after the script name goes to summarize_calls.py:
#     sbatch run_summarize.sbatch --preview 5        # print 5 summaries, save nothing
#     sbatch run_summarize.sbatch                    # usable 2018 calls (~0.5-1 h on an H100)
#     sbatch run_summarize.sbatch --year 0           # every usable call
# Then the main run on the summaries, or both chained:
#     jid=$(sbatch --parsable run_summarize.sbatch) && sbatch --dependency=afterok:$jid run_main.sbatch --summaries
# Rerunning skips calls already summarized.

#SBATCH --account=p53198
#SBATCH --job-name=summarize
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

VENV=${VENV:-/projects/p53198/venvs/vllm}
source "$VENV/bin/activate"
cd /gpfs/home/amc6852/FinLLM

export HF_HOME=${HF_HOME:-/projects/p53198/hf_cache}
mkdir -p "$HF_HOME"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname), started $(date); args: $*"
nvidia-smi
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok); sys.exit(0 if ok else 1)"

python summarize_calls.py "$@"

echo "Finished $(date)"
