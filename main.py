from dotenv import load_dotenv
load_dotenv()
import os
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
print(torch.__version__)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def get_available_tickers(transcripts_dir="Transcripts"):
    if not os.path.isdir(transcripts_dir):
        return []
    return sorted([
        name for name in os.listdir(transcripts_dir)
        if os.path.isdir(os.path.join(transcripts_dir, name))
    ])


def get_available_dates(ticker, transcripts_dir="Transcripts"):
    ticker_dir = os.path.join(transcripts_dir, ticker)
    if not os.path.isdir(ticker_dir):
        return []

    pattern = re.compile(rf"^(\d{{4}})-([A-Za-z]{{3}})-(\d{{2}})-{re.escape(ticker)}\.txt$")

    dates = []
    for filename in os.listdir(ticker_dir):
        match = pattern.match(filename)
        if match:
            year, month, day = match.groups()
            dates.append({
                "year": int(year),
                "month": month,
                "day": int(day),
                "ticker": ticker
            })
    return dates

import pandas as pd
import time
import time
import pandas as pd

def run_all(agent, name="llama",prompt =0, ticker_all = None):
    calls = []
    df = pd.DataFrame(
        columns=["ticker", "year", "month", "day", "decision", "score", "summary"]
    )
    if ticker_all is not None:
        tickers = ticker_all
    else:
        tickers = get_available_tickers()

    for ticker in tickers:
        dates = get_available_dates(ticker)
        calls.extend(dates)

    total_calls = len(calls)

    for i, call in enumerate(calls, start=1):
        ticker = call["ticker"]
        print(f"\r[{i}/{total_calls}] Processing {ticker}...", end="", flush=True)
        start = time.time()
        decision, score, summary = predict(
            agent,
            ticker,
            prompt_number = prompt,
            year=call["year"],
            month=call["month"],
            day=call["day"],
        )
        score = score.iloc[0]
        row = {
            "ticker": ticker,
            "year": call["year"],
            "month": call["month"],
            "day": call["day"],
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
    from llms import predict
    from agent import Agent
    print("Available tickers:", get_available_tickers())
    for ticker in get_available_tickers():
        print(f"Available dates for {ticker}: {get_available_dates(ticker)}")
    ticker = "AAPL"
    year = 2018
    month = "May"
    day = 1
    agent = Agent(provider = "openrouter", llm="tencent/hy3:free")
    #result = predict(agent, ticker, year=year, month=month, day=day)
    #result = agent.run("Hello, how are you?")
    for i in range(5):
        result = run_all(agent, name="hy3-all_" + str(i), prompt = i)
    print(result)