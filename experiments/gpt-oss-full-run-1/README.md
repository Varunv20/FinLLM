# gpt-oss-full-run-1

gpt-oss-20b rating 2018 S&P 1500 earnings calls (buy / hold / sell) from anonymized ~1,000-word
summaries, under the neutral prompt and 11 emotion prompts. Snapshot taken 2026-11-02, before
the `--retry-cut` / forced-decision rerun.

| | |
|---|---|
| Calls | 4,387 usable 2018 calls (`data/calls.csv`) |
| Summaries | gpt-oss-20b, low reasoning effort, `summarize_calls.py` (summaries stay on the cluster, `Earnings_call_transcripts/sp1500_summaries/`) |
| Ratings | gpt-oss-20b via vLLM, default reasoning effort, greedy, 2,048 output tokens, `main.py --summaries` |
| Prompts | `neutral` and emotions 0-10: amusement, anger, anxiety, belief, confusion, depression, disgust, excitement, optimism, panic, surprise |
| Outcome | `scar5`: market-model standardized cumulative abnormal return, trading days 0-4 after the call |
| Known issue | 8-31% of replies per prompt were cut off at 2,048 tokens before a decision (no forcing yet); only 1,416 calls have a decision under all 12 prompts |

## Contents

- `results/` - the raw result pickles, one per prompt (`gpt-oss-20b_neutral.pkl`, `gpt-oss-20b_0.pkl` ... `_10.pkl`).
  Columns: call_id, ticker, year, month, day, decision (as stored at run time), score (raw 5-day
  return), car5, scar5, anonymized, input_tokens, output_tokens, input_truncated, hit_max_tokens,
  summary (the model's full reply).
- `data/`
  - `calls.csv` - the fixed call list of the run.
  - `call_info.csv` - one row per call: ticker, company, index, sector, date, LIWC Posemo / Negemo,
    the event-study columns (timing, event_day0, beta, alpha, sigma, n_est, car5, scar5, raw_ret5,
    mkt_ret5, mktadj5, usable), `eps_beat` / `rev_beat` (+1 beat, -1 miss, from the transcript
    header; 1,453 calls have it) and `analyst_score` (consensus, +2 strong buy .. -2 strong sell; 4,003 calls).
  - `decisions_long.csv` - every reply (52,644 = 12 x 4,387) with the decision parsed the way the
    analysis does it (`decision`; `decision_raw` = stored value), tokens and cut-off flag; no reply text.
- `analysis/`
  - `movement/` - Kendall tau vs SCAR per prompt on all decided calls (`movement_metrics.py`):
    `tau_by_prompt.png`, `scar_by_decision.png`, `movement_metrics.csv`, `calls_with_movement.csv`.
  - `checks/` - baselines (EPS / revenue beat, LIWC tone), beat-controlled regressions, cut-off
    check, run comparison (`summary_run_checks.py`): `checks.txt`, `checks_by_prompt.csv`, PNGs.
  - `emotions/` - the 12 prompts on the same 1,416 calls (`emotion_compare.py`): `tau_scar.png`,
    `decision_mix.png`, `tau_analyst.png`, `pairwise.png`, `emotion_compare.csv`, `emotion_compare.txt`.
- `code/` - the scripts as they were when the analysis ran.

## Reproducing the graphs

From the repository root (they read the catalogue, transcripts and analyst cache in the repo;
`data/call_info.csv` holds every value they take from those):

```bash
python movement_metrics.py "experiments/gpt-oss-full-run-1/results/*.pkl" --out experiments/gpt-oss-full-run-1/analysis/movement --perm 5000
python summary_run_checks.py "experiments/gpt-oss-full-run-1/results/*.pkl" --out experiments/gpt-oss-full-run-1/analysis/checks
python emotion_compare.py "experiments/gpt-oss-full-run-1/results/*.pkl" --out experiments/gpt-oss-full-run-1/analysis/emotions --boot 2000
```

Seeds are fixed, so tau, intervals and p-values come out the same (verified for movement_metrics).

## Headline results

- All decided calls: tau(decision, SCAR) about +0.09 for every prompt (neutral +0.098, n about 3,700 per prompt), p < 0.001.
- Same 1,416 calls for every prompt, alpha = 0.05 with Holm correction:
  - tau with SCAR: no difference between prompts (all equal p = 0.17; 0 of 66 pairs).
  - buy / hold / sell mix: differs (Cochran's Q p < 0.0001; 35 of 66 pairs in buy share). Neutral buys 85.5%; confusion -15 pts, anger -5, disgust and depression -3, excitement +3.
  - tau with analyst consensus: no difference (all equal p = 0.095; 0 of 66 pairs).
- Baselines on the 1,453 calls with a beat / miss line: EPS beat tau +0.178, revenue +0.138; LIWC tone +0.034 on all calls. Controlling for EPS and revenue beat, the model adds little (coefficient +0.17, p = 0.21).

Slides: https://claude.ai/artifact/WVSz4KfuhkoaEUX7n1fHGB (full run) and
https://claude.ai/artifact/Xc78GG7Pu5NA5kzPjWAGWd (emotions on the same calls).
