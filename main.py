from dotenv import load_dotenv
load_dotenv()
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


def run_all(agent, calls, name="llama", prompt=None):
    """Run the agent on every row of `calls`: any slice of load_catalogue() (or load_calls())."""
    df = pd.DataFrame(
        columns=["ticker", "year", "month", "day", "decision", "score", "summary"]
    )
    total_calls = len(calls)
    # model names like "Qwen/Qwen3.5-0.8B-Base" put the pickle in a subfolder
    os.makedirs(os.path.dirname(name) or ".", exist_ok=True)

    for i, (_, call) in enumerate(calls.iterrows(), start=1):
        ticker = call["ticker"]
        date = call["call_date"]
        print(f"\r[{i}/{total_calls}] Processing {ticker}...", end="", flush=True)
        start = time.time()
        decision, prices, _, summary, _ = predict_call(agent, call, emotion = prompt)
        closes = prices["Close"].squeeze("columns")
        score = float(closes.iloc[-1] - closes.iloc[0]) if len(closes) >= 2 else 0.0
        row = {
            "ticker": ticker,
            "year": date.year,
            "month": date.strftime("%b"),
            "day": date.day,
            "decision": decision,
            "score": score,
            "summary": summary,
        }
        df.loc[len(df)] = row
        df.to_pickle(f"{name}.pkl")
        elapsed = time.time() - start
        print(
            f"\r[{i}/{total_calls}] {ticker:<6} ✓ "
            f"Score: {score:.2f} | Time: {elapsed:.2f}s"
        )

    print(f"\nFinished! Processed {total_calls} calls.")
    return df

if __name__ == "__main__":
    from agent import Agent
    from prompts import emotion_stories
    catalogue = load_catalogue()
    print(f"{len(catalogue)} calls across {catalogue['ticker'].nunique()} tickers")

    # Slice the catalogue however you want, e.g.
   
    catalogue[catalogue["call_date"].dt.year == 2018]
    #   catalogue[catalogue["call_date"].between("2018-01-01", "2018-06-30")]
    #   catalogue[(catalogue["sp_index"] == "S&P 500") & (catalogue["gics_sector"] == "Financials")]
    #calls = catalogue[(catalogue["ticker"] == "AAPL") & (catalogue["call_date"].dt.year == 2018)]
    calls = catalogue 
    model = "openai/gpt-oss-120b"
    agent = Agent(provider = "huggingface", llm=model)
    #result = predict_call(agent, calls.iloc[0])
    #result = agent.run("Hello, how are you?")
    emotions = list(emotion_stories)
    for e in emotions:
        result = run_all(agent, calls, name= model + "_" + str(emotions.index(e)), prompt = e)
    print(result)
