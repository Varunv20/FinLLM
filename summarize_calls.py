"""Condense each transcript once into a ~1,000-1,500 token factual summary, so every prompt
in main.py (--summaries) reads the short version instead of the full ~10.5k-token call.

    python summarize_calls.py                         # usable 2018 calls (main.py's default)
    python summarize_calls.py --year 0                # every usable call
    python summarize_calls.py --calls-file runs/anon_openai_gpt-oss-20b_2018_calls.csv
    python summarize_calls.py --preview 5             # print 5 summaries next to their inputs

Summarizer: a light mixture-of-experts model (Qwen3-30B-A3B-Instruct-2507-FP8: 30B weights,
3B active per token, ~31 GB in FP8, no reasoning tokens), run with vLLM in big batches.
It reads the anonymized transcripts by default, so names don't come back in the summary,
and is told to report only what was said, neutrally, without any buy / sell view.

Output: Earnings_call_transcripts/sp1500_summaries/part-*.parquet (one file per batch;
read them all with llms.load_summaries()). Columns: call_id, summary, input_tokens,
output_tokens, input_truncated, hit_max_tokens, leaks (company / ticker mentions in the
summary), anonymized, model, prompt_version. Rerunning skips calls already summarized,
so a job that hit its time limit just continues.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import llms

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
PROMPT_VERSION = 1
SUMMARY_PROMPT = """You are preparing a factual brief of an earnings conference call for a reader who will not see the transcript.

Write the brief from the transcript below, in about 700-1,000 words, under these headings:
1. Reported results: revenue, earnings per share, margins, cash flow and other figures, with the comparisons management gave (year over year, versus guidance, versus expectations).
2. Guidance and outlook: any targets for coming periods, and whether they were raised, lowered, maintained or newly introduced.
3. Business drivers: segments, products, pricing, volumes, costs and other reasons management gave for the results.
4. Capital allocation and balance sheet: buybacks, dividends, debt, acquisitions, investment plans.
5. Risks and headwinds: problems, uncertainties and one-off items management or analysts raised.
6. Q&A: the main topics analysts pressed on and how management answered, including questions management avoided or answered vaguely.
7. Tone: how confident or cautious management sounded, with one or two short quotes as evidence.

Rules:
- Use only what is said in the transcript. Keep every number exactly as stated.
- Be neutral. Do not give your own opinion of the company or the stock, and do not recommend buying, holding or selling.
- Keep placeholders such as [COMPANY], [TICKER], Executive 1, [YEAR+1] exactly as written. Do not guess or add the real names of companies, people, products' owners or dates.
- If a heading has nothing in the transcript, write "Not discussed."

Transcript:
"""
MAX_INPUT_TOKENS = 30000
MAX_NEW_TOKENS = 2048
OUT_DIR = llms.SUMMARIES_PATH


def _final(text):
    """The reply without any reasoning a thinking model put first; "" if the reasoning never
    finished (main.py --summaries skips calls whose summary is empty)."""
    text = text.rsplit("assistantfinal", 1)[-1].rsplit("</think>", 1)[-1].strip()
    return "" if text.startswith("<think>") else text


def select(year, calls_file):
    cat = llms.load_catalogue()
    if calls_file:
        ids = pd.read_csv(calls_file)["call_id"]
        return cat[cat["call_id"].isin(ids)]
    usable = cat[cat["usable"] == True]  # noqa: E712
    return usable[usable["call_date"].dt.year == year] if year else usable


def _leaks(summary, call):
    from data_collection.anonymize_calls import _company_variants, count_leaks
    _, core = _company_variants(call["company"])
    return count_leaks(summary, call["ticker"], core, [])


def run(calls, anonymized=True, batch_size=256, model=MODEL, preview=0, max_input_tokens=MAX_INPUT_TOKENS):
    from agent import Agent
    if anonymized and not os.path.exists(llms.ANON_CALLS_PATH):
        raise SystemExit(f"{llms.ANON_CALLS_PATH} not found: run python data_collection/anonymize_calls.py "
                         "first, or pass --original")
    done = set(llms.load_summaries(columns=["call_id"])["call_id"]) if not preview else set()
    todo = calls[~calls["call_id"].isin(done)]
    if preview:
        todo = todo.head(preview)
    print(f"{len(calls)} calls; {len(calls) - len(todo)} already summarized, {len(todo)} to go", flush=True)
    if todo.empty:
        return
    agent = Agent(provider="huggingface", llm=model, max_model_len=32768)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        texts = [agent.truncate(llms.call_text(c, anonymized), max_input_tokens) for _, c in batch.iterrows()]
        replies = agent.run_batch([SUMMARY_PROMPT + t for t, _ in texts], max_tokens=MAX_NEW_TOKENS)
        rows = [dict(call_id=c["call_id"], summary=_final(r["text"]), input_tokens=r["input_tokens"],
                     output_tokens=r["output_tokens"], input_truncated=cut, hit_max_tokens=r["hit_max_tokens"],
                     leaks=_leaks(_final(r["text"]), c), anonymized=anonymized, model=model, prompt_version=PROMPT_VERSION)
                for (_, c), (_, cut), r in zip(batch.iterrows(), texts, replies)]
        out = pd.DataFrame(rows)
        if preview:
            for (_, c), (t, _), r in zip(batch.iterrows(), texts, rows):
                print(f"\n{'=' * 30} call {c['call_id']} ({c['ticker']}, {c['call_date'].date()}): "
                      f"{r['input_tokens']} -> {r['output_tokens']} tokens, {r['leaks']} leaks {'=' * 30}")
                print(r["summary"])
            return out
        # one file per batch, written whole and then renamed, so a killed job loses at most a batch
        path = os.path.join(OUT_DIR, f"part-{int(time.time() * 1000)}.parquet")
        out.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        n = start + len(batch)
        rate = n / (time.time() - t0)
        print(f"[{len(calls) - len(todo) + n}/{len(calls)}] {rate * 3600:.0f} calls/h, "
              f"~{(len(todo) - n) / rate / 3600:.1f} h left; summary tokens median "
              f"{int(out['output_tokens'].median())}, {int(out['hit_max_tokens'].sum())} cut off, "
              f"{int((out['leaks'] > 0).sum())} with leaks, {int((out['summary'] == '').sum())} empty", flush=True)
    s = llms.load_summaries(columns=["call_id", "output_tokens", "leaks"])
    print(f"done: {len(s)} summaries in {OUT_DIR}; tokens per summary: "
          f"{s['output_tokens'].describe()[['mean', '50%', 'max']].round(0).to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, default=2018, help="usable calls from this year (0 = all years)")
    ap.add_argument("--calls-file", help="summarize exactly the calls in this list (main.py's runs/*.csv)")
    ap.add_argument("--original", action="store_true", help="summarize the original, not anonymized, text")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS)
    ap.add_argument("--preview", type=int, default=0, metavar="N", help="print N summaries, save nothing")
    a = ap.parse_args()
    run(select(a.year, a.calls_file), anonymized=not a.original, batch_size=a.batch_size, model=a.model,
        preview=a.preview, max_input_tokens=a.max_input_tokens)
