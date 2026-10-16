"""Build the catalogue: the full calls DataFrame without the transcript text.

Same rows and `call_id`s as sp1500_earnings_calls.parquet, but ~25 MB instead of ~1 GB,
so it loads in a fraction of a second. Slice it and pass the slice to main.run_all;
the text for each row is looked up by `call_id`.

Run from the FinLLM project root, after data_collection/build_calls.py:
    python data_collection/build_catalogue.py
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent.parent / "Earnings_call_transcripts"
CALLS = HERE / "sp1500_earnings_calls.parquet"
OUT = HERE / "sp1500_catalogue.parquet"


def build_catalogue(save: bool = True) -> pd.DataFrame:
    cols = [c for c in pq.read_schema(CALLS).names if c != "text"]
    df = pd.read_parquet(CALLS, columns=cols)
    if save:
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    out = build_catalogue()
    print(out.shape, "->", OUT)
    print(out[["call_id", "ticker", "sp_index", "quarter", "call_date"]].head().to_string())
