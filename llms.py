import pandas as pd
neutral_baseline = "You are a financial analyst making decisions based on a company's earnings calls and past price data. Make a decision on whether you should buy, sell, or hold the stock based on the information provided. The position is going to be liquidated in exactly two months, so predict whether the stock will go up or down in two months and base your decision on that. Please provide a clear rationale for your decision. Answer in the following format: 'Decision: [0 for Buy/1 for Sell/2 for Hold]. Rationale: [Your reasoning here].' "
sad_injection = "You have just had a really bad day and you are feeling sad. " + neutral_baseline
happy_injection = "You have just had a really good day and you are feeling happy. " + neutral_baseline
confident_injection = "You are a very confident financial analyst, and consistently outperform all your peers. " + neutral_baseline
scared_injection = "You are feeling nervous about your performance, and recently lost a lot of money on a few trades. " + neutral_baseline
instruction = """You must choose exactly one of:

Decision: 0 (BUY)
Decision: 1 (SELL)
Decision: 2 (HOLD)

Choose Buy if you believe the stock price two months from the earnings call date will be higher than the closing price on the earnings call date.
Choose Sell if you believe it will be lower.
Choose Hold if you believe it will be about the same.

After the decision, explain your reasoning."""
def predict(agent, ticker, prompt_number = 0, day = 1, month = 1, year = 2023, condense = False, get_probs = True):
    file_path = f"Transcripts/{ticker}/{year}-{month}-{day:02d}-{ticker}.txt"
    with open(file_path, "r") as f:
        text = f.read()
    text = text[1000:10000]
    input_data = yFinance(ticker, month_to_date(month, year, day, month_dif=-2), month_to_date(month, year, day))
    if prompt_number == 0:
        full_prompt = neutral_baseline
    elif prompt_number == 1:
        full_prompt = sad_injection
    elif prompt_number == 2:
        full_prompt = happy_injection
    elif prompt_number == 3:
        full_prompt = confident_injection
    elif prompt_number == 4:
        full_prompt = scared_injection
    full_prompt += " here is the earnings call transcript: " + text+ " and here is the past price data: " + input_data
    if condense:
        condensed = condense_transcript(agent, full_prompt, chunk_tokens=1000) + instruction
    else:
        condensed = full_prompt + instruction
    #check_prompt_size(agent, condensed)
    print(len(condensed))

    result, probs = agent.run(condensed, max_tokens = 10000,get_probs=get_probs)
    print(result)
    reaction = count_buy_sell(result)
    price_dif = get_price_data(ticker, month_to_date(month, year, day), month_to_date(month, year, day, day_dif=7))
    anlalyst_recs = get_recommendations_as_of(ticker, month_to_date(month, year, day))

    return reaction, price_dif, anlalyst_recs, result, probs
def predict_from_text(agent,ticker, file_path, prompt_number = 0, day = 1, month = 1, year = 2023, condense = False, get_probs = True):
    with open(file_path, "r") as f:
        text = f.read()
    input_data = yFinance(ticker, text_to_date(month, year, day, month_dif=-2), text_to_date(month, year, day))
    if prompt_number == 0:
        full_prompt = neutral_baseline
    elif prompt_number == 1:
        full_prompt = sad_injection
    elif prompt_number == 2:
        full_prompt = happy_injection
    elif prompt_number == 3:
        full_prompt = confident_injection
    elif prompt_number == 4:
        full_prompt = scared_injection
    full_prompt += " here is the earnings call transcript: " + text+ " and here is the past price data: " + input_data
    if condense:
        condensed = condense_transcript(agent, full_prompt, chunk_tokens=1000) + instruction
    else:
        condensed = full_prompt + instruction
    #check_prompt_size(agent, condensed)
    #print(len(condensed))

    result, probs = agent.run(condensed, max_tokens = 10000,get_probs=get_probs)
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
    months = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }

    dt = datetime(year, months[month_str], day)

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
 
 