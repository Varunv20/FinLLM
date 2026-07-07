import pandas as pd
neutral_baseline = "You are a financial analyst making decisions based on a company's earnings calls and past price data. Make a decision on whether you should buy or sell the stock based on the information provided. The position is going to be liquidated in exactly two months, so predict whether the stock will go up or down in two months and base your decision on that. Please provide a clear rationale for your decision. Answer in the following format: 'Decision: [Buy/Sell/Hold]. Rationale: [Your reasoning here].' "
sad_injection = "You have just had a really bad day and you are feeling sad. " + neutral_baseline
happy_injection = "You have just had a really good day and you are feeling happy. " + neutral_baseline
confident_injection = "You are a very confident financial analyst, and consistently outperform all your peers. " + neutral_baseline
scared_injection = "You are feeling nervous about your performance, and recently lost a lot of money on a few trades. " + neutral_baseline
instruction = """You must choose exactly one of:

Decision: Buy
Decision: Sell

Choose Buy if you believe the stock price two months from the earnings call date will be higher than the closing price on the earnings call date.
Choose Sell if you believe it will be lower.

After the decision, explain your reasoning."""
def predict(agent, ticker, prompt_number = 0, day = 1, month = 1, year = 2023, condense = False):
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
    result = agent.run(condensed, max_tokens = 10000)
    print(result)
    reaction = count_buy_sell(result)
    price_dif = get_price_dif(ticker, month_to_date(month, year, day), month_to_date(month, year, day, month_dif=2))
    return reaction, price_dif, result
def get_price_dif(ticker, start_date, end_date):
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if len(data) < 2:
        return 0.0
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    price_dif = end_price - start_price
    return price_dif
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
    print(f"month_to_date: {output}")
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
