from dotenv import load_dotenv
load_dotenv()
import glob
import os
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# llms loads pyarrow, which must happen before torch initialises CUDA
from llms import load_calls, load_catalogue, predict_call

import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import pandas as pd
import time


def run_all(agent, calls, name="llama", prompt=None, anonymized=False):
    """Run the agent on every row of `calls`: any slice of load_catalogue() (or load_calls()).

    Results go to {name}.pkl after every call. If that file already exists (a job that hit
    its time limit), the calls already in it are skipped, so resubmitting picks up where it
    stopped.

    score is the raw stock return over the 5 trading days after the call; car5 / scar5 are
    the market-model abnormal returns (see data_collection/build_event_returns.py)."""
    columns = ["call_id", "ticker", "year", "month", "day", "decision", "score", "car5", "scar5",
               "anonymized", "summary"]
    path = f"{name}.pkl"
    # model names like "Qwen/Qwen3.5-0.8B-Base" put the pickle in a subfolder
    os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
    df = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame(columns=columns)
    done = set(df["call_id"]) if "call_id" in df.columns else set()
    todo = calls[~calls["call_id"].isin(done)]
    total_calls = len(calls)
    if done:
        print(f"{path}: {len(done)} calls already done, {len(todo)} to go")

    for i, (_, call) in enumerate(todo.iterrows(), start=len(calls) - len(todo) + 1):
        ticker = call["ticker"]
        date = call["call_date"]
        print(f"\r[{i}/{total_calls}] Processing {ticker}...", end="", flush=True)
        start = time.time()
        decision, movement, _, summary, _ = predict_call(agent, call, emotion = prompt, anonymized = anonymized)
        score = movement["raw_ret5"]
        row = {
            "call_id": call["call_id"],
            "ticker": ticker,
            "year": date.year,
            "month": date.strftime("%b"),
            "day": date.day,
            "decision": decision,
            "score": score,
            "car5": movement["car5"],
            "scar5": movement["scar5"],
            "anonymized": anonymized,
            "summary": summary,
        }
        df.loc[len(df)] = row
        # write to a temp file first so a job killed mid-save can't corrupt the results
        df.to_pickle(path + ".tmp")
        os.replace(path + ".tmp", path)
        elapsed = time.time() - start
        print(
            f"\r[{i}/{total_calls}] {ticker:<6} ✓ "
            f"5-day return: {score:+.2%} | Time: {elapsed:.2f}s"
        )

    print(f"\nFinished! Processed {total_calls} calls.")
    return df


def select_calls(catalogue, path, n=80, seed=0, reuse=None):
    """The fixed set of calls for a run, saved to `path` so every prompt (and a resubmitted
    job) uses exactly the same calls.

    First run: the calls in the result pickles matching `reuse` (to compare with an earlier
    run on the same calls), else `n` calls sampled with `seed`. Only calls with a usable
    5-day abnormal return are kept."""
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
        calls = usable.merge(keys, on=["ticker", "call_date"]).drop_duplicates("call_id")
        calls = usable.loc[usable["call_id"].isin(calls["call_id"])]
        print(f"reusing the {len(calls)} usable calls from {reuse}")
    else:
        calls = usable.sample(n=n, random_state=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    calls[["call_id", "ticker", "call_date"]].to_csv(path, index=False)
    return calls


if __name__ == "__main__":
    from agent import Agent
    from prompts import emotion_stories
    catalogue = load_catalogue()
    print(f"{len(catalogue)} calls across {catalogue['ticker'].nunique()} tickers")

    # Slice the catalogue however you want, e.g.
    catalogue = catalogue[catalogue["call_date"].dt.year == 2018]
    # catalogue = catalogue[catalogue["call_date"].between("2018-01-01", "2018-06-30")]
    # catalogue = catalogue[(catalogue["sp_index"] == "S&P 500") & (catalogue["gics_sector"] == "Financials")]
    #calls = catalogue[(catalogue["ticker"] == "AAPL") & (catalogue["call_date"].dt.year == 2018)]
    model = "openai/gpt-oss-20b"
    # True: read the transcripts with company, people, tickers and dates removed
    # (run data_collection/anonymize_calls.py first)
    ANONYMIZED = True
    prefix = "anon_" if ANONYMIZED else "sample_"
    # the same calls as the earlier non-anonymized run (sample_openai/), so the two can be
    # compared call by call; delete runs/<prefix>calls.csv to draw a new sample instead
    # calls = select_calls(catalogue, f"runs/{prefix}calls.csv", n=80, seed=0,
    #                      reuse="sample_openai/gpt-oss-120b_*.pkl")
    calls = catalogue
    print(f"{len(calls)} calls in this run")
    agent = Agent(provider = "huggingface", llm=model)
    #result = predict_call(agent, calls.iloc[0])
    #result = agent.run("Hello, how are you?")
    emotions = list(emotion_stories)
    # None = the neutral control (the task prompt with no emotion story), run first
    for e in [None] + emotions:
        label = "neutral" if e is None else str(emotions.index(e))
        result = run_all(agent, calls, name=prefix + model + "_" + label, prompt = e,
                         anonymized = ANONYMIZED)
    print(result)
