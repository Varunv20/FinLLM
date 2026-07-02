from dotenv import load_dotenv
load_dotenv()
import os
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
print(torch.__version__)
print(torch.cuda.is_available())       
print(torch.cuda.get_device_name(0))
print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
print(f"Total VRAM: {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")

def get_available_tickers(transcripts_dir="Transcripts"):
    """
    Returns a list of available ticker symbols, based on subfolder names
    under the Transcripts directory.
    """
    if not os.path.isdir(transcripts_dir):
        return []
    return sorted([
        name for name in os.listdir(transcripts_dir)
        if os.path.isdir(os.path.join(transcripts_dir, name))
    ])


def get_available_dates(ticker, transcripts_dir="Transcripts"):
    """
    Returns a list of available transcript dates for a given ticker,
    based on filenames like 'AAPL/2016-Apr-26-AAPL.txt'.
    Each entry is a dict: {"year": int, "month": str, "day": int}
    """
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
                "day": int(day)
            })
    return dates
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
    agent = Agent(llm="meta-llama/Llama-3.2-3B-Instruct")
    result = predict(agent, ticker, year=year, month=month, day=day)
    #result = agent.run("Hello, how are you?")
    print(result)