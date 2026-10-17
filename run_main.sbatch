#!/bin/bash
# Run main.py on a GPU node.  Submit from the FinLLM folder with:
#     sbatch run_main.sbatch
# Watch it with:   squeue -u $USER      and      tail -f slurm-main-<jobid>.out
#
# Partition / GPU names differ between clusters: check `sinfo` or the cluster docs and
# edit the --partition and --gres lines. gpt-oss-120b needs an 80 GB GPU (A100-80GB / H100);
# for a smaller model (e.g. Qwen/Qwen3.5-0.8B-Base) almost any GPU will do.

#SBATCH --account=p53198
#SBATCH --job-name=main
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu               # <- your cluster's GPU partition
#SBATCH --gres=gpu:1                  # e.g. gpu:a100:1 or gpu:h100:1 to as
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# module load cuda                    # uncomment if your cluster needs CUDA loaded as a module
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM

# Model weights are large (gpt-oss-120b is ~65 GB); keep the Hugging Face cache off a small
# home quota by pointing it at scratch/project space if you have it.
# export HF_HOME=/path/to/scratch/hf_cache

export PYTHONUNBUFFERED=1             # print progress to the log as it happens

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# stop early instead of silently running a 120B model on the CPU
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok); sys.exit(0 if ok else 1)"

python main.py

echo "Finished $(date)"
