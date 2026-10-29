#!/bin/bash
# Summarize the transcripts (summarize_calls.py), then run main.py --summaries on them, in one
# GPU job. Submit from the FinLLM folder. Arguments before "--" go to summarize_calls.py, those
# after it to main.py:
#     sbatch run_summarize_main.sbatch                                  # 2018 calls, gpt-oss-20b, all prompts
#     sbatch run_summarize_main.sbatch -- --smoke                       # summarize, then a 4-call check
#     sbatch run_summarize_main.sbatch --year 0 -- --year 0 --model openai/gpt-oss-120b
# Watch it with:   squeue -u $USER      and      tail -f slurm-summain-<jobid>.out
#
# If the job hits its time limit, submit the same command again: finished summaries and
# finished calls are both skipped. Each step is its own Python process, so the summarizer's
# GPU memory is freed before main.py loads its model.

#SBATCH --account=p53198
#SBATCH --job-name=summain
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SUM_ARGS=()
MAIN_ARGS=()
seen_sep=0
for a in "$@"; do
    if [ "$a" = "--" ] && [ $seen_sep -eq 0 ]; then seen_sep=1; continue; fi
    if [ $seen_sep -eq 0 ]; then SUM_ARGS+=("$a"); else MAIN_ARGS+=("$a"); fi
done

VENV=${VENV:-/projects/p53198/venvs/vllm}
source "$VENV/bin/activate"
cd /gpfs/home/amc6852/FinLLM

export HF_HOME=${HF_HOME:-/projects/p53198/hf_cache}
mkdir -p "$HF_HOME"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "summarize_calls.py args: ${SUM_ARGS[*]:-}"
echo "main.py args: --summaries ${MAIN_ARGS[*]:-}"
nvidia-smi

python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok); sys.exit(0 if ok else 1)"

# stop early if the data hasn't been built (see prepare_data.sh)
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

echo "=== step 1/2: summarize ($(date)) ==="
python summarize_calls.py "${SUM_ARGS[@]}"

echo "=== step 2/2: main.py --summaries ($(date)) ==="
python main.py --summaries "${MAIN_ARGS[@]}"

echo "Finished $(date)"
