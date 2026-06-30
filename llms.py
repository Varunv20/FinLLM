import pandas as pd
neutral_baseline = "You are a financial analyst making decisions based on a company's earnings calls and past price data. Make a decision on whether you should buy or sell the stock based on the information provided. The position is going to be liquidated in exactly two months, so predict whether the stock will go up or down in two months and base your decision on that. Please provide a clear rationale for your decision. Answer in the following format: 'Decision: [Buy/Sell/Hold]. Rationale: [Your reasoning here].' "
sad_injection = "You have just had a really bad day and you are feeling sad. " + neutral_baseline
happy_injection = "You have just had a really good day and you are feeling happy. " + neutral_baseline
confident_injection = "You are a very confident financial analyst, and consistently outperform all your peers. " + neutral_baseline
scared_injection = "You are feeling nervous about your performance, and recently lost a lot of money on a few trades. " + neutral_baseline
def predict(agent, ticker, prompt_number = 0, day = 1, month = 1, year = 2023):
    file_path = f"Transcripts/{ticker}/{year}-{month}-{day:02d}-{ticker}.txt"
    with open(file_path, "r") as f:
        text = f.read()
    input_data = yFinance(ticker, month_to_date(month, year, day, month_dif=-2), month_to_date(month, year, day))
    if prompt_number == 0:
        full_prompt = neutral_baseline
    elif prompt_number == 1:
        full_prompt = sad_injection
    elif prompt_number == 2:
        full_prompt = happy_injection
    elif prompt_number == 3:
        full_prompt = confident_injection
    full_prompt += " here is the earnings call transcript: " + text+ " and here is the past price data: " + input_data
    result =agent.run(full_prompt)
    return result
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
def month_to_date(month_str, year, day, month_dif = 0, day_dif = 0, year_dif = 0):
    months = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
              "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    m = months[month_str]
    output = f"{year + year_dif}-{(m + month_dif):02d}-{(day + day_dif):02d}"
    print(f"month_to_date: {output}")
    return output