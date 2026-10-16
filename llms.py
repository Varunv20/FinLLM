import os
from functools import lru_cache

# On Windows, pyarrow's DLL fails to load once torch has initialised CUDA, so this
# module must be imported before torch.cuda is touched.
try:
    import pyarrow.parquet  # noqa: F401
except ImportError as e:
    raise ImportError("pyarrow failed to load. Import llms before torch "
                      "(or before torch.cuda.is_available()).") from e
import pandas as pd

from prompts import get_prompt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Earnings_call_transcripts")
CALLS_PATH = os.path.join(DATA_DIR, "sp1500_earnings_calls.parquet")        # data_collection/build_calls.py
CATALOGUE_PATH = os.path.join(DATA_DIR, "sp1500_catalogue.parquet")         # data_collection/build_catalogue.py
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def _prepare(df):
    # drop calls without a usable date; index by call_id so rows from either df line up
    return df.dropna(subset=["call_date"]).set_index("call_id", drop=False).rename_axis(None)

@lru_cache(maxsize=1)
def load_calls():
    """Full S&P 1500 earnings calls df, one row per call, including the transcript text."""
    return _prepare(pd.read_parquet(CALLS_PATH))

@lru_cache(maxsize=1)
def load_catalogue():
    """load_calls() without the transcript text (same rows and call_ids); loads in well under
    a second. Slice this however you like and hand the slice to main.run_all."""
    if os.path.exists(CATALOGUE_PATH):
        return _prepare(pd.read_parquet(CATALOGUE_PATH))
    cols = [c for c in pyarrow.parquet.read_schema(CALLS_PATH).names if c != "text"]
    return _prepare(pd.read_parquet(CALLS_PATH, columns=cols))

def get_call(ticker, year, month, day):
    """Row of the calls df for this ticker's call on the given date."""
    df = load_calls()
    date = pd.Timestamp(month_to_date(month, year, day))
    rows = df[(df["ticker"] == ticker) & (df["call_date"] == date)]
    if rows.empty:
        raise KeyError(f"No earnings call for {ticker} on {date.date()}")
    return rows.iloc[0]

def predict_call(agent, call, emotion = None, get_probs = True):
    """Run the agent on one row of load_calls() / load_catalogue()."""
    ticker = call["ticker"]
    date = call["call_date"]
    day, month, year = date.day, MONTHS[date.month - 1], date.year
    # catalogue rows carry no text; fetch it from the full df by call_id
    text = call["text"] if "text" in call.index else load_calls().at[call["call_id"], "text"]
    text = text[1000:10000]
    input_data = yFinance(ticker, month_to_date(month, year, day, month_dif=-2), month_to_date(month, year, day))
    full_prompt = get_prompt(emotion) + "\n\n" + text
    result, probs = agent.run(full_prompt, max_tokens = 10000,get_probs=get_probs)
    print(result)
    reaction = count_buy_sell(result)
    price_dif = get_price_data(ticker, month_to_date(month, year, day), month_to_date(month, year, day, day_dif=7))
    anlalyst_recs = get_recommendations_as_of(ticker, month_to_date(month, year, day))

    return reaction, price_dif, anlalyst_recs, result, probs

def predict(agent, ticker, emotion = None, day = 1, month = 1, year = 2023, condense = False, get_probs = True):
    return predict_call(agent, get_call(ticker, year, month, day), emotion = emotion, get_probs = get_probs)
def predict_from_text(agent,ticker, file_path, emotion = None, day = 1, month = 1, year = 2023, condense = False, get_probs = True):
    with open(file_path, "r") as f:
        text = f.read()
    input_data = yFinance(ticker, text_to_date(month, year, day, month_dif=-2), text_to_date(month, year, day))
    full_prompt = get_prompt(emotion)

    result, probs = agent.run(full_prompt, max_tokens = 10000,get_probs=get_probs)
    #print(result)
    reaction = count_buy_sell(result)
    price_dif = get_price_data(ticker, text_to_date(month, year, day), text_to_date(month, year, day, day_dif=7))
    anlalyst_recs = get_recommendations_as_of(ticker, text_to_date(month, year, day))
    return reaction, price_dif,anlalyst_recs, result, probs
def get_price_dif(ticker, start_date, end_date):
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if len(data) < 2:
        return 0.0
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    price_dif = end_price - start_price
    return price_dif
def get_price_data(ticker, start_date, end_date):
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    return data
def yFinance(ticker, start_date, end_date):
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    return df_to_text(data, ticker)
def df_to_text(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    lines = [f"{ticker} price history:"]
    for date, row in df.iterrows():
        lines.append(f"{date.date()}: close=${row['Close']:.2f}, volume={int(row['Volume']):,}")
    return "\n".join(lines)
def get_recommendations(ticker):
    """
    Get analyst recommendation trends (buy/hold/sell disagreements) for a ticker
    using yfinance's recommendations/upgrades-downgrades data.

    Returns a dict summarizing the analyst disagreement, based on
    yf.Ticker(ticker).recommendations, which gives period-by-period counts of
    strongBuy / buy / hold / sell / strongSell recommendations from analysts.
    """
    import yfinance as yf

    stock = yf.Ticker(ticker)
    recs = stock.recommendations  # DataFrame: period, strongBuy, buy, hold, sell, strongSell

    if recs is None or recs.empty:
        return {
            "ticker": ticker,
            "disagreement": None,
            "summary": f"No recommendation data available for {ticker}.",
            "raw": None,
        }

    # Most recent period (row 0 is typically the current month, "0m")
    latest = recs.iloc[0]

    strong_buy = int(latest.get("strongBuy", 0))
    buy = int(latest.get("buy", 0))
    hold = int(latest.get("hold", 0))
    sell = int(latest.get("sell", 0))
    strong_sell = int(latest.get("strongSell", 0))

    total = strong_buy + buy + hold + sell + strong_sell
    buy_side = strong_buy + buy
    sell_side = sell + strong_sell

    if total == 0:
        disagreement = None
    else:
        # Simple disagreement metric: how split analysts are (0 = full consensus, 1 = max split)
        counts = [strong_buy, buy, hold, sell, strong_sell]
        max_count = max(counts)
        disagreement = 1 - (max_count / total)

    if buy_side > sell_side and buy_side > hold:
        consensus = "Buy"
    elif sell_side > buy_side and sell_side > hold:
        consensus = "Sell"
    else:
        consensus = "Hold"

    summary = (
        f"{ticker}: {total} analysts — "
        f"StrongBuy={strong_buy}, Buy={buy}, Hold={hold}, "
        f"Sell={sell}, StrongSell={strong_sell}. "
        f"Consensus leans '{consensus}' with a disagreement score of "
        f"{disagreement:.2f}" if disagreement is not None else "no data"
    )

    return {
        "ticker": ticker,
        "consensus": consensus,
        "disagreement": disagreement,
        "counts": {
            "strongBuy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strongSell": strong_sell,
        },
        "summary": summary,
        "raw": recs,
    }
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def month_to_date(month_str, year, day, month_dif=0, day_dif=0, year_dif=0):
    dt = datetime(year, MONTHS.index(month_str) + 1, day)

    dt += relativedelta(years=year_dif, months=month_dif)
    dt += timedelta(days=day_dif)

    output = dt.strftime("%Y-%m-%d")
    #print(f"month_to_date: {output}")
    return output

def text_to_date(month_str, year, day, month_dif=0, day_dif=0, year_dif=0):
    months = {
        "01": 1, "02": 2, "03": 3, "04": 4,
        "05": 5, "06": 6, "07": 7, "08": 8,
        "09": 9, "10": 10, "11": 11, "12": 12,
    }
    year = int(year)
    day = int(day)
    dt = datetime(year, months[month_str], day)

    dt += relativedelta(years=year_dif, months=month_dif)
    dt += timedelta(days=day_dif)

    output = dt.strftime("%Y-%m-%d")
    #print(f"month_to_date: {output}")
    return output
def check_prompt_size(agent, full_prompt):
    tokenizer = agent.client.tokenizer
    n_tokens = len(tokenizer.encode(full_prompt))
    return 
import os, json

def condense_transcript_cached(agent, text, ticker, date_str, chunk_tokens=1000):
    cache_dir = "cache/summaries"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{ticker}_{date_str}.txt"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()
    condensed = condense_transcript(agent, text, chunk_tokens=chunk_tokens)
    with open(cache_path, "w") as f:
        f.write(condensed)
    return condensed
def summarize_chunk(agent, chunk):
    prompt = f"Summarize the key financial points, guidance, and tone from this excerpt of an earnings call transcript in 3-4 sentences:\n\n{chunk}"
    return agent.run(prompt, max_tokens=150)

def chunk_text(text, tokenizer, chunk_tokens=1000):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+chunk_tokens] for i in range(0, len(tokens), chunk_tokens)]
    return [tokenizer.decode(c) for c in chunks]

def condense_transcript(agent, text, chunk_tokens=1000):
    tokenizer = agent.client.tokenizer
    n_tokens = len(tokenizer.encode(text))
    if n_tokens <= chunk_tokens:
        return text  # short enough, no need to condense
    chunks = chunk_text(text, tokenizer, chunk_tokens)
    summaries = [summarize_chunk(agent, c) for c in chunks]
    return "\n".join(summaries)

import re

def count_buy_sell(text):
    text_lower = text.lower()
    buy_count = len(re.findall(r'\bbuy\b', text_lower))
    sell_count = len(re.findall(r'\bsell\b', text_lower))
    
    if buy_count > sell_count:
        return "buy"
    return "sell"

 
# Map common Yahoo/analyst grade text -> normalized bucket.
# Extend this as you encounter new grade strings in your data.
GRADE_MAP = {
    # Buy-side
    "buy": "buy",
    "strong buy": "strong_buy",
    "outperform": "buy",
    "overweight": "buy",
    "market outperform": "buy",
    "sector outperform": "buy",
    "positive": "buy",
    "add": "buy",
    "accumulate": "buy",
    "conviction buy": "strong_buy",
    "top pick": "strong_buy",
 
    # Hold-side
    "hold": "hold",
    "neutral": "hold",
    "equal-weight": "hold",
    "equal weight": "hold",
    "market perform": "hold",
    "sector perform": "hold",
    "in-line": "hold",
    "peer perform": "hold",
    "perform": "hold",
    "sector weight": "hold",
 
    # Sell-side
    "sell": "sell",
    "underperform": "sell",
    "underweight": "sell",
    "negative": "sell",
    "reduce": "sell",
    "market underperform": "sell",
    "sector underperform": "sell",
    "strong sell": "strong_sell",
}
 
 
def normalize_grade(grade: str) -> str:
    """Map a free-text analyst grade to one of:
    strong_buy, buy, hold, sell, strong_sell, or 'unknown' if unmapped."""
    if not isinstance(grade, str):
        return "unknown"
    return GRADE_MAP.get(grade.strip().lower(), "unknown")
 
 
def get_recommendations_as_of(ticker: str, as_of_date, lookback_days: int = 365):
    """
    Reconstruct an analyst buy/hold/sell breakdown as of a specific historical
    date, using yfinance's upgrades/downgrades history.
 
    Parameters
    ----------
    ticker : str
    as_of_date : str ("YYYY-MM-DD") or datetime
        The historical date you want the recommendation snapshot for
        (e.g. an earnings call date).
    lookback_days : int
        How far back before as_of_date to look for each firm's most recent
        rating. Analyst ratings can be stale for a while, so default is 1 year.
        Firms with no rating inside this window are excluded.
 
    Returns
    -------
    dict with keys: ticker, as_of_date, consensus, disagreement, counts,
    firms_used, summary, raw
    """
    import yfinance as yf
 
    if isinstance(as_of_date, str):
        as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d")
 
    stock = yf.Ticker(ticker)
    hist = stock.get_upgrades_downgrades()
 
    if hist is None or hist.empty:
        return {
            "ticker": ticker,
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "consensus": "Unknown",
            "disagreement": None,
            "counts": {},
            "firms_used": 0,
            "summary": f"No upgrade/downgrade history available for {ticker}.",
            "raw": None,
        }
 
    # GradeDate is the index in recent yfinance versions; normalize either way.
    df = hist.reset_index()
    date_col = "GradeDate" if "GradeDate" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
 
    window_start = as_of_date - pd.Timedelta(days=lookback_days)
    mask = (df[date_col] <= as_of_date) & (df[date_col] >= window_start)
    window_df = df.loc[mask].sort_values(date_col)
 
    if window_df.empty:
        return {
            "ticker": ticker,
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "consensus": "Unknown",
            "disagreement": None,
            "counts": {},
            "firms_used": 0,
            "summary": (
                f"No analyst ratings found for {ticker} in the "
                f"{lookback_days}-day window before {as_of_date.date()}."
            ),
            "raw": hist,
        }
 
    # Each firm's most recent rating on/before as_of_date wins.
    latest_per_firm = window_df.groupby("Firm").tail(1)
 
    counts = {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0, "unknown": 0}
    for _, row in latest_per_firm.iterrows():
        bucket = normalize_grade(row.get("ToGrade", ""))
        counts[bucket] = counts.get(bucket, 0) + 1
 
    known_total = sum(v for k, v in counts.items() if k != "unknown")
    buy_side = counts["strong_buy"] + counts["buy"]
    sell_side = counts["sell"] + counts["strong_sell"]
    hold_side = counts["hold"]
 
    if known_total == 0:
        consensus = "Unknown"
        disagreement = None
    else:
        if buy_side > sell_side and buy_side > hold_side:
            consensus = "Buy"
        elif sell_side > buy_side and sell_side > hold_side:
            consensus = "Sell"
        else:
            consensus = "Hold"
        max_count = max(counts["strong_buy"], counts["buy"], counts["hold"],
                         counts["sell"], counts["strong_sell"])
        disagreement = 1 - (max_count / known_total)
 
    summary = (
        f"{ticker} as of {as_of_date.date()} ({len(latest_per_firm)} firms, "
        f"{lookback_days}-day lookback): "
        f"StrongBuy={counts['strong_buy']}, Buy={counts['buy']}, "
        f"Hold={counts['hold']}, Sell={counts['sell']}, "
        f"StrongSell={counts['strong_sell']}"
        + (f", Unmapped={counts['unknown']}" if counts["unknown"] else "")
        + f". Consensus leans '{consensus}'"
        + (f", disagreement score={disagreement:.2f}" if disagreement is not None else "")
    )
 
    return {
        "ticker": ticker,
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "consensus": consensus,
        "disagreement": disagreement,
        "counts": counts,
        "firms_used": len(latest_per_firm),
        "summary": summary,
        "raw": latest_per_firm,
    }
 
 