#!/bin/bash
# Build the data main.py needs, on the cluster's login node (the price download needs
# internet, which GPU nodes may not have). Run from the FinLLM folder:
#     bash prepare_data.sh
# Each step is skipped if its output already exists; delete the file to rebuild it.
# Takes ~10 minutes, most of it the anonymization.

set -euo pipefail
source /home/amc6852/venvs/p2/bin/activate
cd /gpfs/home/amc6852/FinLLM

DATA=${FINLLM_DATA_DIR:-Earnings_call_transcripts}
WORKERS=${WORKERS:-4}                 # keep the login node usable; raise on a compute node

[ -f "$DATA/sp1500_earnings_calls.parquet" ] || python data_collection/build_calls.py
[ -f "$DATA/daily_adj_close.parquet" ]       || python data_collection/fetch_prices.py
[ -f "$DATA/sp1500_event_returns.parquet" ]  || python data_collection/build_event_returns.py
python data_collection/build_catalogue.py    # cheap; always rebuilt so it includes the event returns
[ -f "$DATA/sp1500_earnings_calls_anon.parquet" ] || \
    python data_collection/anonymize_calls.py --workers "$WORKERS"

python - <<'EOF'
import llms
c = llms.load_catalogue()
print(f"catalogue: {len(c)} calls, {int((c['usable'] == True).sum())} with a usable 5-day abnormal return")
EOF
echo "Data ready. Submit the run with:  sbatch run_main.sbatch"
