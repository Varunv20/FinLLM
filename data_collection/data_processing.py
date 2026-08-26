"""
Match earnings call transcript files to exact dates using Yahoo Finance.
--------------------------------------------------------------------------
Walks a directory of transcript files named like:
    Company/{year}_{quarter}_{ticker}.txt
e.g.
    Capgemini/2018_Q1_cap.txt

For each company folder, auto-looks-up the real Yahoo Finance ticker via
Yahoo's search/autocomplete endpoint (since filename abbreviations like
"cap" aren't real tickers for non-US companies). Then fetches historical
earnings dates for the resolved ticker and matches them to each file's
year/quarter.

Requirements:
    pip install yfinance pandas requests

Usage (run from data_collection):
    python match_earnings_dates.py
"""

import os
import re
import time
import requests
import pandas as pd
import yfinance as yf
import os

# ---------------------- CONFIG ----------------------

ROOT_DIR = r"data_collection\NLP_Dataset"          # relative to data_collection
OUTPUT_CSV = "transcripts_with_dates.csv"
TICKER_CACHE_CSV = "company_ticker_map.csv"  # saved so lookups aren't repeated

FILENAME_PATTERN = re.compile(r"^(\d{4})_([Qq][1-4])_([A-Za-z\.\-]+)\.txt$")

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ------------------------------------------------------


def parse_filename(filepath):
    filename = os.path.basename(filepath)
    company_folder = os.path.basename(os.path.dirname(filepath))

    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    year, quarter, ticker_abbrev = match.groups()
    return {
        "filepath": filepath,
        "company_folder": company_folder,
        "year": int(year),
        "quarter": quarter.upper(),
        "ticker_abbrev": ticker_abbrev.upper(),
    }


def collect_files(root_dir):
    records = []
    skipped = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".txt"):
                continue
            full_path = os.path.join(dirpath, fname)
            parsed = parse_filename(full_path)
            if parsed:
                records.append(parsed)
            else:
                skipped.append(full_path)

    return records, skipped


def lookup_ticker(company_name):
    """Use Yahoo Finance's search endpoint to resolve a company name to a
    real ticker symbol. Returns the top equity match, or None if nothing
    found."""
    try:
        resp = requests.get(
            YAHOO_SEARCH_URL,
            params={"q": company_name, "quotesCount": 5, "newsCount": 0},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("quotes", [])

        # Prefer exact equity matches
        for q in quotes:
            if q.get("quoteType") == "EQUITY":
                return q.get("symbol")

        # Fallback: first result of any type
        if quotes:
            return quotes[0].get("symbol")

        return None
    except Exception as e:
        print(f"  ERROR looking up ticker for '{company_name}': {e}")
        return None


def build_ticker_map(company_names, cache_path):
    """Resolve each company folder name to a real ticker, using a cached
    CSV so repeated runs don't re-query Yahoo for companies already
    resolved."""
    if os.path.exists(cache_path):
        cached_df = pd.read_csv(cache_path)
        ticker_map = dict(zip(cached_df["company_folder"], cached_df["resolved_ticker"]))
    else:
        ticker_map = {}

    unresolved = [c for c in company_names if c not in ticker_map]

    for company in unresolved:
        print(f"Looking up ticker for '{company}'...")
        resolved = lookup_ticker(company)
        ticker_map[company] = resolved
        print(f"  -> {resolved}")
        time.sleep(1)  # be polite to Yahoo's endpoint

    # Save/update cache
    pd.DataFrame(
        [{"company_folder": k, "resolved_ticker": v} for k, v in ticker_map.items()]
    ).to_csv(cache_path, index=False)

    return ticker_map


def quarter_to_month_range(quarter):
    mapping = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}
    return mapping.get(quarter.upper())


def get_earnings_dates_for_symbol(symbol, limit=40):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"Earnings Date": "earnings_date"}, inplace=True)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
        return df
    except Exception as e:
        print(f"  ERROR fetching earnings dates for {symbol}: {e}")
        return pd.DataFrame()


def match_date(earnings_df, year, quarter):
    month_range = quarter_to_month_range(quarter)
    if not month_range or earnings_df.empty:
        return None

    _, end_month = month_range
    quarter_end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(1)

    window_start = quarter_end
    window_end = quarter_end + pd.Timedelta(days=60)

    candidates = earnings_df[
        (pd.to_datetime(earnings_df["earnings_date"]) >= window_start) &
        (pd.to_datetime(earnings_df["earnings_date"]) <= window_end)
    ]

    if candidates.empty:
        return None

    return candidates.iloc[0]["earnings_date"]


def main():
    print(f"Scanning {ROOT_DIR} for transcript files...")
    records, skipped = collect_files(ROOT_DIR)

    print(f"Found {len(records)} matching files.")
    if skipped:
        print(f"Skipped {len(skipped)} files that didn't match the naming pattern. Examples:")
        for s in skipped[:5]:
            print(f"  {s}")

    if not records:
        print("No files matched the expected pattern. Check ROOT_DIR and FILENAME_PATTERN.")
        return

    df = pd.DataFrame(records)

    print("\nSample parsed rows:")
    print(df.head(5).to_string(index=False))

    # Step 1: resolve company folder names -> real tickers
    company_names = df["company_folder"].unique().tolist()
    ticker_map = build_ticker_map(company_names, TICKER_CACHE_CSV)

    df["resolved_ticker"] = df["company_folder"].map(ticker_map)

    unresolved_companies = df[df["resolved_ticker"].isna()]["company_folder"].unique()
    if len(unresolved_companies) > 0:
        print(f"\nCould not resolve tickers for: {list(unresolved_companies)}")
        print("These rows will have no earnings_call_date. Check "
              f"'{TICKER_CACHE_CSV}' and fix manually if needed, then re-run.")

    # Step 2: fetch earnings dates per resolved ticker
    resolved_tickers = df["resolved_ticker"].dropna().unique()
    earnings_cache = {}
    failed_tickers = []

    for ticker in resolved_tickers:
        print(f"Fetching Yahoo Finance earnings dates for {ticker}...")
        result = get_earnings_dates_for_symbol(ticker)
        earnings_cache[ticker] = result
        if result.empty:
            failed_tickers.append(ticker)
        time.sleep(1)

    # Step 3: match dates
    df["earnings_call_date"] = None
    unmatched = []

    for idx, row in df.iterrows():
        ticker = row["resolved_ticker"]
        if pd.isna(ticker):
            unmatched.append((row["company_folder"], row["year"], row["quarter"]))
            continue

        earnings_df = earnings_cache.get(ticker, pd.DataFrame())
        matched_date = match_date(earnings_df, row["year"], row["quarter"])

        if matched_date:
            df.at[idx, "earnings_call_date"] = matched_date
        else:
            unmatched.append((ticker, row["year"], row["quarter"]))

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. Saved to {OUTPUT_CSV}")
    print(f"Ticker map saved to {TICKER_CACHE_CSV} (edit manually to fix any wrong lookups, then re-run)")
    print(f"Matched: {len(df) - len(unmatched)} rows")
    print(f"Unmatched: {len(unmatched)} rows")

    if failed_tickers:
        print(f"\nResolved tickers with NO earnings date data: {failed_tickers}")

    if unmatched:
        print("\nUnmatched examples (ticker/company, year, quarter):")
        for u in unmatched[:20]:
            print(f"  {u}")
        print("Note: yfinance's earnings_dates history has limited "
              "lookback (often ~4 years). Older rows may need a fallback "
              "via Alpha Vantage's EARNINGS endpoint.")


if __name__ == "__main__":
    main()