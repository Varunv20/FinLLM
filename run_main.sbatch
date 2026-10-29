#!/bin/bash
# Run main.py on a GPU node.  Submit from the FinLLM folder with:
#     sbatch run_main.sbatch
# Watch it with:   squeue -u $USER      and      tail -f slurm-main-<jobid>.out
#
# Before the first submit, build the data on the login node (needs internet):
#     bash prepare_data.sh
#
# The run is 12 prompts (neutral + 11 emotions) x the calls in runs/anon_calls.csv, about a
# minute per call on an H100 (~14 h for 76 calls). If the job hits its time limit, just
# submit it again: run_all skips the calls already saved in each prompt's pickle.

#SBATCH --account=p53198
#SBATCH --job-name=main
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1             # gpt-oss-120b needs ~65 GB: an 80 GB H100 (a 40 GB A100 runs out of memory)
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
export PYTHONIOENCODING=utf-8         # the progress lines contain a check mark

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# stop early instead of silently running a 120B model on the CPU
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

python main.py

echo "Finished $(date)"
