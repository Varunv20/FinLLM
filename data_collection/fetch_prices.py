"""Download daily split- and dividend-adjusted closes for every catalogue ticker plus the S&P 500.

Writes one wide table, dates x tickers (^GSPC = S&P 500 index), that
build_event_returns.py uses for estimation windows and event windows:
    Earnings_call_transcripts/daily_adj_close.parquet

Run from the FinLLM project root (needs internet; takes a few minutes):
    python data_collection/fetch_prices.py
Re-running only downloads tickers missing from the file; pass --refresh to start over.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd

HERE = Path(os.environ.get("FINLLM_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "Earnings_call_transcripts"))
OUT = HERE / "daily_adj_close.parquet"
CATALOGUE = HERE / "sp1500_catalogue.parquet"
MARKET = "^GSPC"
# a year of estimation window before the first call, two weeks after the last
START, END = "2004-06-01", "2020-01-31"
BATCH = 50


def yahoo_symbol(ticker):
    """Catalogue tickers use dots for share classes (BF.B); Yahoo uses dashes (BF-B)."""
    return ticker.replace(".", "-")


def download(tickers, retries=3):
    """Adjusted closes for `tickers` (catalogue spelling) as a dates x tickers frame."""
    import yfinance as yf
    symbols = {yahoo_symbol(t): t for t in tickers}
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(list(symbols), start=START, end=END, auto_adjust=True,
                               progress=False, threads=True, group_by="column")
            close = data["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(next(iter(symbols)))
            return close.rename(columns=symbols)
        except Exception as e:
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(5 * attempt)
    return pd.DataFrame()


def fetch_prices(refresh=False):
    tickers = sorted(pd.read_parquet(CATALOGUE, columns=["ticker"])["ticker"].unique())
    have = pd.DataFrame()
    if OUT.exists() and not refresh:
        have = pd.read_parquet(OUT)
    todo = [t for t in [MARKET] + tickers if t not in have.columns]
    print(f"{len(tickers)} tickers + {MARKET}; {len(todo)} to download")
    frames = [have] if len(have) else []
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        frames.append(download(batch))
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices = prices.dropna(how="all", subset=[MARKET])  # keep only days the market traded
    prices.to_parquet(OUT)
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    print(f"{prices.shape[0]} trading days x {prices.shape[1]} series -> {OUT}")
    if missing:
        print(f"{len(missing)} tickers with no Yahoo history (their calls get no event returns): "
              f"{', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    return prices


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-download every ticker")
    fetch_prices(ap.parse_args().refresh)
