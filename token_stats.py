"""How many tokens the transcripts (input) and the model's replies (output) take.

    python token_stats.py                                   # gpt-oss tokenizer, 2000 calls
    python token_stats.py --tokenizer Qwen/Qwen3.5-27B --sample 500
    python token_stats.py --results "anon_openai/gpt-oss-20b_*.pkl"

Input: full transcripts (and the anonymized copy, if built), tokenized exactly as the model
sees them inside the chat template with the task prompt. Output: the replies stored in
result pickles (reasoning + answer). Only the tokenizer is downloaded, not the model.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import llms
from prompts import get_prompt


def _describe(name, n):
    n = np.asarray(n)
    q = np.percentile(n, [50, 90, 95, 99])
    print(f"{name:34s} n={len(n):5d}  mean={n.mean():7.0f}  median={q[0]:7.0f}  p90={q[1]:7.0f}  "
          f"p95={q[2]:7.0f}  p99={q[3]:7.0f}  max={n.max():7.0f}")


def sample_texts(path, n, seed=0):
    """`n` random transcripts from a calls parquet, read one row group at a time."""
    f = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    total = f.metadata.num_rows
    pick = set(rng.choice(total, size=min(n, total), replace=False).tolist())
    out, start = [], 0
    for rg in range(f.num_row_groups):
        rows = f.metadata.row_group(rg).num_rows
        want = [i - start for i in range(start, start + rows) if i in pick]
        if want:
            col = f.read_row_group(rg, columns=["text"]).column("text")
            out += [col[i].as_py() or "" for i in want]
        start += rows
    return out


def main(tokenizer_name, n, results):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    chat = tok.apply_chat_template([{"role": "user", "content": get_prompt(None) + "\n\n"}],
                                   tokenize=False, add_generation_prompt=True)
    overhead = len(tok(chat, add_special_tokens=False)["input_ids"])
    print(f"tokenizer {tokenizer_name}; chat template + task prompt = {overhead} tokens\n")

    def count(texts):
        return [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]

    texts = sample_texts(llms.CALLS_PATH, n)
    full = count(texts)
    _describe("full transcript", full)
    _describe("current slice (chars 1000-10000)", count([t[1000:10000] for t in texts]))
    print(f"{'chars per token':34s} {sum(map(len, texts)) / sum(full):.2f}")
    if os.path.exists(llms.ANON_CALLS_PATH):
        anon = count(sample_texts(llms.ANON_CALLS_PATH, n))
        _describe("anonymized transcript", anon)
    for limit in (8192, 16384, 24576, 32768):
        print(f"  calls that fit in {limit:6d} input tokens: {np.mean(np.array(full) + overhead <= limit):6.1%}")

    files = sorted(glob.glob(results))
    if files:
        print()
        for f in files[:3] + (["..."] if len(files) > 3 else []):
            print("  " + f)
        summaries = pd.concat([pd.read_pickle(f)["summary"] for f in files]).dropna().astype(str)
        out = count(summaries)
        _describe("output (reasoning + answer)", out)
        final = [s.rsplit("assistantfinal", 1)[-1] for s in summaries]
        _describe("output, final answer only", count(final))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    ap.add_argument("--sample", type=int, default=2000, help="transcripts to tokenize")
    ap.add_argument("--results", default="sample_openai/gpt-oss-120b_*.pkl",
                    help="result pickles whose replies to count")
    a = ap.parse_args()
    main(a.tokenizer, a.sample, a.results)
