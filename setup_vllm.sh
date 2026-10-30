#!/bin/bash
# One-time, on the login node: add vLLM to the p2 environment the sbatch scripts use.
# Without it the jobs still run but fall back to plain transformers: much slower, and the
# summarizer runs one call at a time.
source /home/amc6852/venvs/p2/bin/activate
pip install vllm
