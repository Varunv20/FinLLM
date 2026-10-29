"""Run a model on earnings calls under the neutral prompt and every emotion prompt.

    python main.py                                   # defaults below (all usable 2018 calls)
    python main.py --smoke                           # 4 calls, neutral prompt: check setup + speed
    python main.py --model openai/gpt-oss-120b --n 500 --seed 0
    python main.py --model Qwen/Qwen3.5-27B --engine vllm --year 2018
    python main.py --summaries                       # read summarize_calls.py's summaries

Results: one pickle per prompt, <prefix><model>_<neutral|0..10>.pkl, saved after every batch;
rerunning skips calls already saved, so a job that hit its time limit just continues.
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import glob
import os
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# llms loads pyarrow, which must happen before torch initialises CUDA
from llms import (load_calls, load_catalogue, predict_call, build_prompt, parse_decision,
                  MAX_INPUT_TOKENS, MAX_NEW_TOKENS)

import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import pandas as pd
import time

COLUMNS = ["call_id", "ticker", "year", "month", "day", "decision", "score", "car5", "scar5", "anonymized",
           "input_tokens", "output_tokens", "input_truncated", "hit_max_tokens", "summary"]


def run_all(agent, calls, name="llama", prompt=None, anonymized=False, batch_size=64,
            max_input_tokens=MAX_INPUT_TOKENS, max_new_tokens=MAX_NEW_TOKENS, summaries=False):
    """Run the agent on every row of `calls`: any slice of load_catalogue() (or load_calls()).

    Calls go to the model `batch_size` at a time (vllm runs a batch concurrently) and
    {name}.pkl is saved after every batch. If that file already exists (a job that hit its
    time limit), the calls already in it are skipped.

    score is the raw stock return over the 5 trading days after the call; car5 / scar5 are
    the market-model abnormal returns (see data_collection/build_event_returns.py)."""
    path = f"{name}.pkl"
    # model names like "Qwen/Qwen3.5-0.8B-Base" put the pickle in a subfolder
    os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
    df = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame(columns=COLUMNS)
    done = set(df["call_id"]) if "call_id" in df.columns else set()
    todo = calls[~calls["call_id"].isin(done)]
    total_calls = len(calls)
    print(f"{path}: {len(done)} calls already done, {len(todo)} to go", flush=True)

    t0 = time.time()
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        built = [build_prompt(agent, call, prompt, anonymized, max_input_tokens, summaries)
                 for _, call in batch.iterrows()]
        replies = agent.run_batch([p for p, _ in built], max_tokens=max_new_tokens)
        rows = []
        for (_, call), (_, cut), r in zip(batch.iterrows(), built, replies):
            date = call["call_date"]
            rows.append({
                "call_id": call["call_id"],
                "ticker": call["ticker"],
                "year": date.year,
                "month": date.strftime("%b"),
                "day": date.day,
                "decision": parse_decision(r["text"]),
                "score": call.get("raw_ret5"),
                "car5": call.get("car5"),
                "scar5": call.get("scar5"),
                "anonymized": anonymized,
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "input_truncated": cut,
                "hit_max_tokens": r["hit_max_tokens"],
                "summary": r["text"],
            })
        df = pd.concat([df, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
        # write to a temp file first so a job killed mid-save can't corrupt the results
        df.to_pickle(path + ".tmp")
        os.replace(path + ".tmp", path)
        n_done = len(done) + start + len(batch)
        rate = (start + len(batch)) / (time.time() - t0)
        print(f"[{n_done}/{total_calls}] {name}: {rate * 3600:.0f} calls/h, "
              f"~{(len(todo) - start - len(batch)) / rate / 3600:.1f} h left for this prompt; "
              f"decisions in batch: {pd.Series([r['decision'] for r in rows]).value_counts().to_dict()}",
              flush=True)

    print(f"Finished {path}: {len(df)} calls.", flush=True)
    return df


def select_calls(catalogue, path, n=80, seed=0, reuse=None):
    """The fixed set of calls for a run, saved to `path` so every prompt (and a resubmitted
    job) uses exactly the same calls.

    First run: the calls in the result pickles matching `reuse` (to compare with an earlier
    run on the same calls), else `n` calls sampled with `seed` (n=None: all of them). Only
    calls with a usable 5-day abnormal return are kept."""
    usable = catalogue[catalogue["usable"] == True]  # noqa: E712
    if os.path.exists(path):
        ids = pd.read_csv(path)["call_id"]
        return usable.loc[usable["call_id"].isin(ids)]
    files = sorted(glob.glob(reuse)) if reuse else []
    if files:
        prev = pd.concat([pd.read_pickle(f)[["ticker", "year", "month", "day"]] for f in files])
        prev["call_date"] = pd.to_datetime(prev["year"].astype(str) + "-" + prev["month"].astype(str)
                                           + "-" + prev["day"].astype(str), format="%Y-%b-%d")
        keys = prev[["ticker", "call_date"]].drop_duplicates()
        ids = usable.merge(keys, on=["ticker", "call_date"])["call_id"]
        calls = usable.loc[usable["call_id"].isin(ids)]
        print(f"reusing the {len(calls)} usable calls from {reuse}")
    else:
        calls = usable if n is None or n >= len(usable) else usable.sample(n=n, random_state=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    calls[["call_id", "ticker", "call_date"]].to_csv(path, index=False)
    return calls


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--engine", choices=["vllm", "transformers"], help="default: vllm if installed")
    ap.add_argument("--year", type=int, default=2018, help="calls from this year (0 = all years)")
    ap.add_argument("--n", type=int, help="sample this many usable calls (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calls-file", help="fixed call list (default runs/<prefix><model>_<year>_calls.csv)")
    ap.add_argument("--original", action="store_true", help="read the original, not anonymized, transcripts")
    ap.add_argument("--summaries", action="store_true",
                    help="read each call's summary (summarize_calls.py) instead of the full transcript")
    ap.add_argument("--prompts", default="all", help="'all', 'neutral', or comma-separated emotions")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--quantization", default="auto", help="auto / none / fp8 / int4")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="gpt-oss only; default: the model's own (medium)")
    ap.add_argument("--smoke", action="store_true", help="4 calls, neutral prompt, results in runs/smoke/")
    return ap.parse_args()


if __name__ == "__main__":
    from agent import Agent
    from prompts import emotion_stories
    args = parse_args()
    catalogue = load_catalogue()
    print(f"{len(catalogue)} calls across {catalogue['ticker'].nunique()} tickers")

    # Slice the catalogue however you want, e.g.
    #   catalogue[catalogue["call_date"].between("2018-01-01", "2018-06-30")]
    #   catalogue[(catalogue["sp_index"] == "S&P 500") & (catalogue["gics_sector"] == "Financials")]
    if args.year:
        catalogue = catalogue[catalogue["call_date"].dt.year == args.year]
    anonymized = not args.original
    base = "anon_" if anonymized else "sample_"
    prefix = base + ("sum_" if args.summaries else "")
    model_tag = args.model.replace("/", "_")
    # full-transcript and summary runs share a call list, so they can be compared call by call
    calls_file = args.calls_file or f"runs/{base}{model_tag}_{args.year or 'all'}_calls.csv"
    calls = select_calls(catalogue, calls_file, n=args.n, seed=args.seed)
    if args.summaries:
        from llms import load_summaries
        sums = load_summaries(columns=["call_id", "summary"])
        have = set(sums.loc[sums["summary"].str.len() > 0, "call_id"])
        missing = (~calls["call_id"].isin(have)).sum()
        if missing:
            print(f"WARNING: {missing} of {len(calls)} calls have no summary and are skipped; "
                  f"run python summarize_calls.py --calls-file {calls_file}", flush=True)
        calls = calls[calls["call_id"].isin(have)]
    emotions = list(emotion_stories)
    if args.prompts == "all":
        prompts = [None] + emotions          # None = the neutral control, run first
    else:
        prompts = [None if p == "neutral" else p for p in args.prompts.split(",")]
    out_prefix = prefix
    if args.smoke:
        calls, prompts, out_prefix = calls.head(4), [None], "runs/smoke/" + prefix
    print(f"{len(calls)} calls x {len(prompts)} prompts ({calls_file})", flush=True)

    agent = Agent(provider = "huggingface", llm=args.model, engine=args.engine,
                  max_model_len=args.max_model_len,
                  quantization=None if args.quantization == "none" else args.quantization,
                  reasoning_effort=args.reasoning_effort)
    for e in prompts:
        label = "neutral" if e is None else str(emotions.index(e))
        result = run_all(agent, calls, name=out_prefix + args.model + "_" + label, prompt = e,
                         anonymized = anonymized, batch_size=args.batch_size,
                         max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                         summaries=args.summaries)
    if args.smoke:
        print(result[["ticker", "decision", "input_tokens", "output_tokens", "input_truncated",
                      "hit_max_tokens"]].to_string())
        print("\nlast reply:\n" + str(result["summary"].iloc[-1])[-1500:])
