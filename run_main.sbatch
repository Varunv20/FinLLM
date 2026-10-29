#!/bin/bash
# Run main.py on a GPU node. Submit from the FinLLM folder; anything after the script name
# is passed to main.py:
#     sbatch run_main.sbatch --smoke                          # 4 calls: check setup + speed first
#     sbatch run_main.sbatch                                  # gpt-oss-20b, all usable 2018 calls
#     sbatch run_main.sbatch --model openai/gpt-oss-120b --n 500
# Watch it with:   squeue -u $USER      and      tail -f slurm-main-<jobid>.out
#
# One-time setup on the login node (both need internet):
#     bash setup_vllm.sh        # the vLLM environment (VENV below)
#     bash prepare_data.sh      # prices, abnormal returns, catalogue, anonymized transcripts
#
# If the job hits its time limit, submit the same command again: each prompt's pickle is
# saved after every batch and finished calls are skipped.
#
# GPU memory: gpt-oss-20b ~13 GB, Qwen3.5-27B ~54 GB (bf16), gpt-oss-120b ~63 GB, plus room
# for the KV cache of ~11k-token transcripts. An 80 GB H100 fits all three.

#SBATCH --account=p53198
#SBATCH --job-name=main
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

VENV=${VENV:-/projects/p53198/venvs/vllm}
source "$VENV/bin/activate"
cd /gpfs/home/amc6852/FinLLM

# Model weights are large; keep the Hugging Face cache in the project allocation, not home.
export HF_HOME=${HF_HOME:-/projects/p53198/hf_cache}
mkdir -p "$HF_HOME"
export PYTHONUNBUFFERED=1             # print progress to the log as it happens
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname), started $(date); args: $*"
nvidia-smi

# stop early instead of silently running on the CPU
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok); sys.exit(0 if ok else 1)"

# stop early if the data main.py needs hasn't been built (see prepare_data.sh)
python - <<'EOF'
import os, sys
import llms
missing = [p for p in (llms.CATALOGUE_PATH, llms.EVENTS_PATH, llms.ANON_CALLS_PATH) if not os.path.exists(p)]
if missing:
    sys.exit("missing data files, run `bash prepare_data.sh` on the login node first:\n  " + "\n  ".join(missing))
if "car5" not in llms.load_catalogue().columns:
    sys.exit("catalogue has no event returns: python data_collection/build_catalogue.py")
print("data OK")
EOF

python main.py "$@"

echo "Finished $(date)"
