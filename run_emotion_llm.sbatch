#!/bin/bash
#SBATCH --account=p53198
#SBATCH --job-name=emollm
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Qwen3-30B-A3B judges each sentence of the calls and the gpt-oss-120b neutral replies for the 11 emotions.
# Validation on StockEmotions first, then the calls; results in results_120b/text_emotions/ (resubmit to continue).
#   sbatch run_emotion_llm.sbatch --validate-only      # just the validation
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"

python emotion_llm.py "$@"
