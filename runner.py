import pandas as pd

df = pd.read_csv(r"transcripts_with_dates.csv")
df_main = df[df['earnings_call_date'].notnull()]
from agent import Agent
from llms import predict, predict_from_text
from dotenv import load_dotenv
import os
import pandas as pd
 
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm isn't installed
    class tqdm:  # minimal no-op stand-in with the same interface we use
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
 
        def __iter__(self):
            return iter(self.iterable)
 
        def set_postfix(self, *args, **kwargs):
            pass
 
        def write(self, msg):
            print(msg)
 
        def close(self):
            pass
 
 
def run_predictions(
    df_main,
    agent,
    prompt_number,
    output_name,
    condense=False,
    get_probs=True,
    checkpoint_every=10,
    output_dir=".",
):
    """
    Run predict_from_text() over every row of df_main and save the results.
 
    Parameters
    ----------
    df_main : pd.DataFrame
        Source dataframe. Must contain 'filepath', 'resolved_ticker', and
        'earnings_call_date' (string, 'YYYY-MM-DD...') columns.
    agent : object
        The agent object passed straight through to predict_from_text.
    prompt_number : int
        Which prompt variant to use (0-4). Passed through to
        predict_from_text and also stored in the output for bookkeeping.
    output_name : str
        Base filename (no extension) for the checkpoint/output files.
        Results are written to f"{output_dir}/{output_name}.pkl" (full
        fidelity, since some result fields are DataFrames/dicts) and
        mirrored to f"{output_dir}/{output_name}.csv" (stringified,
        convenient for quick inspection).
    condense : bool
        Passed through to predict_from_text.
    get_probs : bool
        Passed through to predict_from_text.
    checkpoint_every : int
        Save progress to disk after every N newly-processed rows.
    output_dir : str
        Directory the checkpoint/output files live in. Created if needed.
 
    Returns
    -------
    pd.DataFrame
        The accumulated results (one row per df_main row processed so far).
        Columns: row_idx, ticker, filepath, year, month, day,
        prompt_number, reaction, price_dif, analyst_recs, result, probs,
        error.
 
    Notes
    -----
    - If a checkpoint file already exists for output_name, rows already
      recorded there are skipped, so re-running with the same output_name
      resumes rather than restarts.
    - If predict_from_text raises for a given row, the error is caught,
      logged to the 'error' column, and the loop continues on to the next
      row rather than aborting the whole run.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{output_name}.pkl")
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
 
    result_columns = [
        "row_idx", "ticker", "filepath", "year", "month", "day",
        "prompt_number", "reaction", "price_dif", "analyst_recs",
        "result", "probs", "error",
    ]
 
    # Resume from an existing checkpoint if present.
    if os.path.exists(checkpoint_path):
        results_df = pd.read_pickle(checkpoint_path)
        done_indices = set(results_df["row_idx"].tolist())
        print(f"Resuming '{output_name}': {len(done_indices)} row(s) already done.")
    else:
        results_df = pd.DataFrame(columns=result_columns)
        done_indices = set()
 
    n_rows = len(df_main)
    new_rows_since_checkpoint = 0
 
    # Iterating over the full range keeps the bar's total accurate; rows
    # already present in the checkpoint are skipped (fast) rather than
    # re-processed, so on a resume the bar zips quickly through the
    # already-done prefix before slowing down at the real work.
    pbar = tqdm(range(n_rows), desc=output_name, total=n_rows)
 
    for idx in pbar:
        if idx in done_indices:
            continue
 
        path = df_main.iloc[idx]["filepath"]
        ticker = df_main.iloc[idx]["resolved_ticker"]
        date_str = str(df_main.iloc[idx]["earnings_call_date"])
        year = date_str[:4]
        month = date_str[5:7]
        day = date_str[8:]
 
        pbar.set_postfix(ticker=ticker, refresh=False)
 
        row_result = {
            "row_idx": idx,
            "ticker": ticker,
            "filepath": path,
            "year": year,
            "month": month,
            "day": day,
            "prompt_number": prompt_number,
            "reaction": None,
            "price_dif": None,
            "analyst_recs": None,
            "result": None,
            "probs": None,
            "error": None,
        }
 
        try:
            reaction, price_dif, analyst_recs, result, probs = predict_from_text(
                agent,
                ticker,
                path,
                prompt_number=prompt_number,
                day=int(day),
                month=int(month),
                year=int(year),
                condense=condense,
                get_probs=get_probs,
            )
            row_result.update(
                {
                    "reaction": reaction,
                    "price_dif": price_dif,
                    "analyst_recs": analyst_recs,
                    "result": result,
                    "probs": probs,
                }
            )
        except Exception as e:
            row_result["error"] = str(e)
            pbar.set_postfix(ticker=ticker, error=True, refresh=False)
            pbar.write(f"[{idx + 1}/{n_rows}] ERROR for {ticker} ({path}): {e}")
 
        results_df = pd.concat(
            [results_df, pd.DataFrame([row_result])], ignore_index=True
        )
        new_rows_since_checkpoint += 1
 
        if new_rows_since_checkpoint >= checkpoint_every:
            results_df.to_pickle(checkpoint_path)
            results_df.to_csv(csv_path, index=False)
            new_rows_since_checkpoint = 0
            pbar.write(f"Checkpoint saved at row {idx + 1}/{n_rows} -> {checkpoint_path}")
 
    pbar.close()
 
    # Final save, in case the last batch was smaller than checkpoint_every.
    results_df.to_pickle(checkpoint_path)
    results_df.to_csv(csv_path, index=False)
    print(f"Done. {len(results_df)} row(s) saved to {checkpoint_path} (and {csv_path}).")
 
    return results_df
 
# Search for and load the .env file
load_dotenv()
model = input("Enter the model name (e.g., deepseek-ai/DeepSeek-V4-Flash-0731): ")
provider = input("Enter the provider (e.g., huggingface, openrouter): ")
agent = Agent(provider=provider, model=model)
idx = 1
path = df_main.iloc[idx]['filepath']
ticker = df_main.iloc[idx]['resolved_ticker']
year=df_main.iloc[idx]['earnings_call_date'][:4]
month=df_main.iloc[idx]['earnings_call_date'][5:7]
day=df_main.iloc[idx]['earnings_call_date'][8:]


res = predict_from_text(agent, ticker,path, day=day, month=month, year=year, prompt_number=0, get_probs=False)
print(res)

for prompt_number in range(5):
    run_predictions(df_main, agent, prompt_number=prompt_number, output_name=f"results/deep_seek_prompt_{prompt_number}", condense=False, get_probs=True, checkpoint_every=10, output_dir=".")