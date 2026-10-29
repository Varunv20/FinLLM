import os
import re
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

_DECISION_RE = re.compile(r"decision\W*(?:is\W*)?(buy|sell|hold|[012])\b", re.I)
_CODE = {"0": "buy", "1": "sell", "2": "hold"}


def parse_decision(summary, fallback=None):
    """buy / sell / hold from the model's 'Decision: ...' line.

    The stored `decision` column comes from counting the words buy/sell anywhere in the
    reply, so it can't say hold and is often wrong ("Decision: 2 for Hold" -> "sell").
    Falls back to the stored value when the reply has no Decision line."""
    if isinstance(summary, str):
        # gpt-oss replies are "analysis<reasoning>assistantfinal<answer>"; the reasoning
        # often floats a decision it later changes, so read only the final answer
        summary = summary.rsplit("assistantfinal", 1)[-1]
        # Qwen-style thinking models: "<think>...</think>answer"
        summary = summary.rsplit("</think>", 1)[-1]
        m = _DECISION_RE.search(summary.replace("*", ""))
        if m:
            d = m.group(1).lower()
            return _CODE.get(d, d)
    return fallback

DATA_DIR = os.environ.get("FINLLM_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "Earnings_call_transcripts"))
CALLS_PATH = os.path.join(DATA_DIR, "sp1500_earnings_calls.parquet")        # data_collection/build_calls.py
CATALOGUE_PATH = os.path.join(DATA_DIR, "sp1500_catalogue.parquet")         # data_collection/build_catalogue.py
EVENTS_PATH = os.path.join(DATA_DIR, "sp1500_event_returns.parquet")       # data_collection/build_event_returns.py
ANON_CALLS_PATH = os.path.join(DATA_DIR, "sp1500_earnings_calls_anon.parquet")  # data_collection/anonymize_calls.py
SUMMARIES_PATH = os.path.join(DATA_DIR, "sp1500_summaries")  # summarize_calls.py (a folder of part files)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# abnormal returns over the 5 trading days after each call (see build_event_returns.py)
EVENT_COLS = ["timing", "event_day0", "beta", "sigma", "car5", "scar5", "raw_ret5", "mkt_ret5",
              "mktadj5", "usable"]

def _prepare(df):
    # add the event returns if the file predates them; drop calls without a usable date;
    # index by call_id so rows from either df line up
    if "car5" not in df.columns and os.path.exists(EVENTS_PATH):
        df = df.merge(pd.read_parquet(EVENTS_PATH), on="call_id", how="left")
    return df.dropna(subset=["call_date"]).set_index("call_id", drop=False).rename_axis(None)

@lru_cache(maxsize=1)
def load_calls():
    """Full S&P 1500 earnings calls df, one row per call, including the transcript text.
    Needs ~7 GB of RAM; run_all doesn't need it (it reads text per call via get_text)."""
    return _prepare(pd.read_parquet(CALLS_PATH))

@lru_cache(maxsize=1)
def load_catalogue():
    """load_calls() without the transcript text (same rows and call_ids); loads in well under
    a second. Slice this however you like and hand the slice to main.run_all."""
    if os.path.exists(CATALOGUE_PATH):
        return _prepare(pd.read_parquet(CATALOGUE_PATH))
    cols = [c for c in pyarrow.parquet.read_schema(CALLS_PATH).names if c != "text"]
    return _prepare(pd.read_parquet(CALLS_PATH, columns=cols))

@lru_cache(maxsize=2)
def _text_locations(path=CALLS_PATH):
    """call_id -> (row group, offset) in `path`, from the call_id column alone."""
    f = pyarrow.parquet.ParquetFile(path)
    locs = {}
    for rg in range(f.num_row_groups):
        ids = f.read_row_group(rg, columns=["call_id"]).column("call_id").to_pylist()
        locs.update((cid, (rg, i)) for i, cid in enumerate(ids))
    return f, locs

@lru_cache(maxsize=4)
def _text_row_group(path, rg):
    f, _ = _text_locations(path)
    return f.read_row_group(rg, columns=["text"]).column("text")

def get_text(call_id, anonymized=False):
    """Transcript text for one call, reading only the row group it lives in (not the ~2 GB
    of text load_calls() pulls into memory). anonymized=True reads the copy with company,
    people, tickers and dates removed (data_collection/anonymize_calls.py)."""
    path = ANON_CALLS_PATH if anonymized else CALLS_PATH
    if anonymized and not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run python data_collection/anonymize_calls.py")
    rg, i = _text_locations(path)[1][int(call_id)]
    return _text_row_group(path, rg)[i].as_py()

def get_call(ticker, year, month, day):
    """Row of the catalogue for this ticker's call on the given date."""
    df = load_catalogue()
    date = pd.Timestamp(month_to_date(month, year, day))
    rows = df[(df["ticker"] == ticker) & (df["call_date"] == date)]
    if rows.empty:
        raise KeyError(f"No earnings call for {ticker} on {date.date()}")
    return rows.iloc[0]

# transcript tokens sent to the model: the whole call for ~99% of calls (median ~10.5k,
# p99 ~18k tokens; see token_stats.py); longer calls keep their first MAX_INPUT_TOKENS
MAX_INPUT_TOKENS = 24000
MAX_NEW_TOKENS = 2048       # reply budget; gpt-oss-120b's replies were ~900 tokens (p99 ~1.3k)

def load_summaries(columns=None):
    """Every summary written by summarize_calls.py (the latest per call), as a df; empty
    if none yet."""
    import glob
    files = sorted(glob.glob(os.path.join(SUMMARIES_PATH, "part-*.parquet")))
    if not files:
        return pd.DataFrame(columns=columns or ["call_id", "summary"])
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    return df.drop_duplicates("call_id", keep="last")

@lru_cache(maxsize=1)
def _summary_map():
    s = load_summaries(columns=["call_id", "summary"])
    return dict(zip(s["call_id"], s["summary"]))

def call_text(call, anonymized=False, summaries=False):
    """The transcript of a call as the model should read it: without the Seeking Alpha
    header (title, ticker, and its EPS / revenue beat-or-miss line). summaries=True gives
    its summary from summarize_calls.py instead."""
    if summaries:
        summary = _summary_map().get(call["call_id"])
        if summary is None:
            raise KeyError(f"no summary for call {call['call_id']}; run python summarize_calls.py")
        return "Summary of the earnings call:\n" + summary
    if anonymized:
        return get_text(call["call_id"], anonymized=True)  # header already stripped
    from data_collection.anonymize_calls import strip_header
    text = call["text"] if "text" in call.index else get_text(call["call_id"])
    return "\n".join(strip_header(text.split("\n"))[0])

def build_prompt(agent, call, emotion=None, anonymized=False, max_input_tokens=MAX_INPUT_TOKENS, summaries=False):
    """Task prompt + transcript (or its summary) for one call, and whether the text was cut."""
    text, cut = agent.truncate(call_text(call, anonymized, summaries), max_input_tokens)
    return get_prompt(emotion) + "\n\n" + text, cut

def predict_call(agent, call, emotion = None, get_probs = True, anonymized = False, analyst_recs = False,
                 max_input_tokens = MAX_INPUT_TOKENS, max_new_tokens = MAX_NEW_TOKENS):
    """Run the agent on one row of load_calls() / load_catalogue().

    Returns (reaction, movement, analyst_recs, result, probs). movement is the call's
    EVENT_COLS (abnormal returns over the 5 trading days after the call), read from the
    catalogue rather than downloaded. analyst_recs=True also queries Yahoo for the analyst
    consensus as of the call date (slow; None otherwise). main.run_all batches many calls
    through agent.run_batch instead."""
    ticker = call["ticker"]
    date = call["call_date"]
    full_prompt, _ = build_prompt(agent, call, emotion, anonymized, max_input_tokens)
    result, probs = agent.run(full_prompt, max_tokens = max_new_tokens, get_probs=get_probs)
    reaction = count_buy_sell(result)
    movement = {c: call.get(c) for c in EVENT_COLS}
    recs = get_recommendations_as_of(ticker, date.strftime("%Y-%m-%d")) if analyst_recs else None

    return reaction, movement, recs, result, probs

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
    "outperformer": "buy",
    "long-term buy": "buy",
 
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
    "hold neutral": "hold",
    "fair value": "hold",
    "average": "hold",
    "mixed": "hold",
 
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
 
 