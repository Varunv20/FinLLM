"""
fetch_analyst_ratings.py
-------------------------------------------------------------------------
Adds human analyst buy/sell/hold ground truth to the FinLLM dataset.

For every (ticker, earnings_call_date) pair in transcripts_with_dates.csv,
pulls that ticker's full analyst upgrade/downgrade history via yfinance
and keeps every individual analyst action that falls within a configurable
window after the earnings call (default: 7 days). One row is kept per
analyst action -- no aggregation -- so a single earnings call can have
zero, one, or several human-analyst rows.

Source: yfinance's Ticker.upgrades_downgrades (scrapes Yahoo Finance's
upgrade/downgrade history). Free, no API key. It's an unofficial/scraped
endpoint, so:
  - coverage varies by ticker (some tickers have very sparse history)
  - Yahoo may occasionally block/throttle -- the script backs off and
    retries, and caches per-ticker results to disk so re-runs are cheap
    and you never lose progress from a mid-run failure.

Run from the FinLLM project root (same folder as transcripts_with_dates.csv):
    pip install yfinance pandas
    python fetch_analyst_ratings.py

Output: human_analyst_ratings.csv, plus a per-ticker cache under
cache/upgrades_downgrades/<ticker>.csv so subsequent runs (e.g. widening
the window, or re-running after Yahoo throttles you) don't re-hit Yahoo
for tickers already fetched.
-------------------------------------------------------------------------
"""

import os
import time
import pandas as pd
import yfinance as yf

# ---------------------- CONFIG ----------------------

TRANSCRIPTS_CSV = "transcripts_with_dates.csv"
OUTPUT_CSV = "human_analyst_ratings.csv"
CACHE_DIR = "cache/upgrades_downgrades"
WINDOW_DAYS_AFTER = 7          # "one week after earnings" -- change if you want a wider/narrower window
WINDOW_DAYS_BEFORE = 0         # set >0 if you also want actions just before the call
SLEEP_BETWEEN_TICKERS = 1.5    # be polite to Yahoo; raise this if you get throttled
MAX_RETRIES = 3

# Map raw Yahoo "ToGrade" strings to a normalized BUY/SELL/HOLD label.
# Extend this if you see grades in your data that fall through to UNKNOWN
# (check human_analyst_ratings.csv's `to_grade` column for anything mapped
# to UNKNOWN and add it here).
GRADE_MAP = {
    "buy": "BUY", "strong buy": "BUY", "outperform": "BUY", "overweight": "BUY",
    "add": "BUY", "positive": "BUY", "accumulate": "BUY", "conviction buy": "BUY",
    "top pick": "BUY", "long-term buy": "BUY", "sector outperform": "BUY",

    "sell": "SELL", "strong sell": "SELL", "underperform": "SELL",
    "underweight": "SELL", "reduce": "SELL", "negative": "SELL",
    "sector underperform": "SELL",

    "hold": "HOLD", "neutral": "HOLD", "equal-weight": "HOLD",
    "equal weight": "HOLD", "market perform": "HOLD", "sector perform": "HOLD",
    "peer perform": "HOLD", "in-line": "HOLD",
}


def normalize_grade(grade):
    if not isinstance(grade, str) or not grade.strip():
        return "UNKNOWN"
    return GRADE_MAP.get(grade.strip().lower(), "UNKNOWN")


def fetch_upgrades_downgrades(ticker):
    """Fetch (and cache) the full upgrade/downgrade history for one ticker."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if df.empty:
            return df
        df["GradeDate"] = pd.to_datetime(df["GradeDate"])
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            df = t.upgrades_downgrades
            if df is None:
                df = pd.DataFrame()
            else:
                df = df.reset_index()  # index is GradeDate
                if "GradeDate" not in df.columns and "index" in df.columns:
                    df = df.rename(columns={"index": "GradeDate"})
            df.to_csv(cache_path, index=False)
            if not df.empty:
                df["GradeDate"] = pd.to_datetime(df["GradeDate"])
            return df
        except Exception as e:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed for {ticker}: {e}")
            time.sleep(3 * attempt)

    print(f"  giving up on {ticker} after {MAX_RETRIES} attempts")
    empty = pd.DataFrame()
    empty.to_csv(cache_path, index=False)
    return empty


def main():
    transcripts = pd.read_csv(TRANSCRIPTS_CSV)
    transcripts = transcripts.dropna(subset=["resolved_ticker", "earnings_call_date"])
    transcripts["earnings_call_date"] = pd.to_datetime(transcripts["earnings_call_date"])

    if transcripts.empty:
        print(f"No rows in {TRANSCRIPTS_CSV} have both resolved_ticker and "
              f"earnings_call_date populated. Run data_collection/data_processing.py "
              f"first to fill in earnings_call_date, then re-run this script.")
        return

    tickers = sorted(transcripts["resolved_ticker"].unique())
    print(f"{len(transcripts)} earnings-call rows across {len(tickers)} tickers.")

    all_matches = []
    empty_tickers = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Fetching upgrade/downgrade history for {ticker}...")
        history = fetch_upgrades_downgrades(ticker)

        if history.empty:
            empty_tickers.append(ticker)
            time.sleep(SLEEP_BETWEEN_TICKERS)
            continue

        calls = transcripts[transcripts["resolved_ticker"] == ticker]

        for _, call in calls.iterrows():
            call_date = call["earnings_call_date"]
            window_start = call_date - pd.Timedelta(days=WINDOW_DAYS_BEFORE)
            window_end = call_date + pd.Timedelta(days=WINDOW_DAYS_AFTER)

            in_window = history[
                (history["GradeDate"] >= window_start) & (history["GradeDate"] <= window_end)
            ]

            for _, action in in_window.iterrows():
                all_matches.append({
                    "ticker": ticker,
                    "company_folder": call.get("company_folder"),
                    "earnings_call_date": call_date.date(),
                    "action_date": action["GradeDate"].date(),
                    "days_after_call": (action["GradeDate"] - call_date).days,
                    "firm": action.get("Firm"),
                    "action": action.get("Action"),        # e.g. up / down / main / init / reit
                    "from_grade": action.get("FromGrade"),
                    "to_grade": action.get("ToGrade"),
                    "rating_label": normalize_grade(action.get("ToGrade")),
                })

        time.sleep(SLEEP_BETWEEN_TICKERS)

    result = pd.DataFrame(all_matches)
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. {len(result)} human-analyst actions matched within "
          f"{WINDOW_DAYS_BEFORE}-{WINDOW_DAYS_AFTER} days of an earnings call.")
    print(f"Saved to {OUTPUT_CSV}")
    if empty_tickers:
        print(f"\nNo upgrade/downgrade data returned for {len(empty_tickers)} tickers "
              f"(Yahoo has no history, or the symbol needs remapping): {empty_tickers}")
    if not result.empty:
        unknown = result[result["rating_label"] == "UNKNOWN"]
        if not unknown.empty:
            print(f"\n{len(unknown)} actions had a to_grade that didn't map to BUY/SELL/HOLD. "
                  f"Unique unmapped grades: {sorted(unknown['to_grade'].dropna().unique().tolist())}")
            print("Add these to GRADE_MAP at the top of this script and re-run (cache makes this fast).")


if __name__ == "__main__":
    main()
