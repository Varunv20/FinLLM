#!/bin/bash
# One-time: a separate Python environment with vLLM, for run_main.sbatch. Run on the login
# node from the FinLLM folder (needs internet; ~10 minutes, ~10 GB):
#     bash setup_vllm.sh
# It lives in the project allocation, not home: vLLM, torch and the pip cache would
# otherwise fill the home quota again.

set -euo pipefail
# where the venv and pip cache go (not home, its quota is small); e.g. STORE=/scratch/$USER bash setup_vllm.sh
STORE=${STORE:-/projects/p53198}
VENV=${VENV:-$STORE/venvs/vllm}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STORE/pip_cache}

# same Python (3.11) as the existing environment
mkdir -p "$STORE" || { echo "cannot write to $STORE; set STORE to a folder you own, e.g. STORE=/scratch/$USER"; exit 1; }
/home/amc6852/venvs/p2/bin/python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install vllm                      # brings its own matching torch, transformers and CUDA libraries
pip install pandas pyarrow python-dotenv huggingface_hub

python - <<'EOF'
import torch, transformers, vllm
print(f"vllm {vllm.__version__} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"transformers {transformers.__version__}")
EOF
echo "Done. run_main.sbatch uses $VENV"
