"""Abnormal returns over the 5 trading days after every earnings call.

For each call:
  day 0      first trading day the market can react: the call day if the call starts
             before 4:00 pm ET (and the call day traded), otherwise the next trading day
  estimation trading days -250 .. -30 relative to day 0 (at least MIN_EST returns)
             market model  r_t = alpha + beta * m_t + e_t,  sigma = std of e_t
  event      AR_t = r_t - (alpha + beta * m_t), t = 0 .. 4
             car5  = sum of AR_t
             scar5 = car5 / sd(car5), with sd from the market model's prediction error
                     (sigma^2 * [L + L^2/T + (sum m_t - L * mean m)^2 / sum (m - mean m)^2])
  also       raw_ret5 (stock), mkt_ret5 (S&P 500), mktadj5 = raw_ret5 - mkt_ret5 (beta = 1)
             price_check: |raw_ret5 from Yahoo - from the dataset's own price window| < 2%
             (False flags a ticker that Yahoo maps to a different company or bad data)
  usable     car5 exists, price_check did not fail, and fewer than MAX_TICKER_FAIL of the
             ticker's checked calls failed (a reused ticker: Yahoo has another company's prices)

r_t and m_t are daily log returns of adjusted closes (data_collection/fetch_prices.py),
so the 5-day abnormal returns add up. Writes, keyed by call_id:
    Earnings_call_transcripts/sp1500_event_returns.parquet
which llms.load_catalogue() / load_calls() merge in, and build_catalogue.py bakes in.

Run from the FinLLM project root, after fetch_prices.py:
    python data_collection/build_event_returns.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(os.environ.get("FINLLM_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "Earnings_call_transcripts"))
PRICES = HERE / "daily_adj_close.parquet"
CATALOGUE = HERE / "sp1500_catalogue.parquet"
OUT = HERE / "sp1500_event_returns.parquet"
MARKET = "^GSPC"
EST_START, EST_END = -250, -30   # estimation window, trading days relative to day 0
EVENT_DAYS = 5                   # days 0 .. 4
MIN_EST = 120                    # minimum valid estimation-window returns
CHECK_TOL = 0.02
MAX_TICKER_FAIL = 0.3

EVENT_COLS = ["timing", "event_day0", "beta", "alpha", "sigma", "n_est", "car5", "scar5",
              "raw_ret5", "mkt_ret5", "mktadj5", "price_check", "usable"]


def call_hour(raw):
    """Hour of day (ET) from strings like "May 15, 2006, 4:30 p.m. EST", or NaN."""
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m", raw, re.I) if isinstance(raw, str) else None
    if not m:
        return np.nan
    return int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0) + int(m.group(2) or 0) / 60


def timing_of(hour):
    if np.isnan(hour):
        return "unknown"
    return "after close" if hour >= 16 else ("before open" if hour < 9.5 else "during market")


def _dataset_ret5(row, day0_offset):
    """Raw 5-day return from the catalogue's own (unadjusted) price window, for a sanity check."""
    days = range(-10, 11)
    traded = [(d, row[f"price_day_{d}"]) for d in days if pd.notna(row[f"price_day_{d}"])]
    before = [p for d, p in traded if d < day0_offset]
    after = [p for d, p in traded if d >= day0_offset]
    if not before or len(after) < EVENT_DAYS:
        return np.nan
    return after[EVENT_DAYS - 1] / before[-1] - 1


def event_returns(cat, prices):
    """One row of EVENT_COLS per catalogue row (same index)."""
    logp = np.log(prices)
    rets = logp.diff()
    m = rets[MARKET].to_numpy()
    dates = rets.index
    out = []
    for _, row in cat.iterrows():
        rec = dict.fromkeys(EVENT_COLS, np.nan)
        timing = timing_of(call_hour(row["call_datetime_raw"]))
        rec["timing"] = timing
        out.append(rec)
        t = row["ticker"]
        if t not in rets.columns or pd.isna(row["call_date"]):
            continue
        # day 0: first trading day on/after the call date, pushed one day for after-close calls
        i0 = dates.searchsorted(row["call_date"])
        if i0 < len(dates) and dates[i0] == row["call_date"] and timing == "after close":
            i0 += 1
        if i0 + EVENT_DAYS > len(dates) or i0 + EST_START < 1:
            continue
        rec["event_day0"] = dates[i0]
        r = rets[t].to_numpy()
        est = slice(i0 + EST_START, i0 + EST_END + 1)
        ev = slice(i0, i0 + EVENT_DAYS)
        re_, me = r[est], m[est]
        ok = ~(np.isnan(re_) | np.isnan(me))
        if ok.sum() < MIN_EST or np.isnan(r[ev]).any() or np.isnan(m[ev]).any():
            continue
        re_, me = re_[ok], me[ok]
        T = len(me)
        mbar = me.mean()
        sxx = ((me - mbar) ** 2).sum()
        beta = ((me - mbar) * (re_ - re_.mean())).sum() / sxx
        alpha = re_.mean() - beta * mbar
        sigma = np.sqrt(((re_ - alpha - beta * me) ** 2).sum() / (T - 2))
        ar = r[ev] - (alpha + beta * m[ev])
        car = ar.sum()
        L = EVENT_DAYS
        var = sigma ** 2 * (L + L ** 2 / T + (m[ev].sum() - L * mbar) ** 2 / sxx)
        raw5, mkt5 = np.expm1(r[ev].sum()), np.expm1(m[ev].sum())
        # the dataset's price_day_k are calendar days from the call date; day 0 sits at this offset
        check = _dataset_ret5(row, (dates[i0] - row["call_date"]).days)
        rec.update(beta=beta, alpha=alpha, sigma=sigma, n_est=T, car5=car, scar5=car / np.sqrt(var),
                   raw_ret5=raw5, mkt_ret5=mkt5, mktadj5=raw5 - mkt5,
                   price_check=bool(abs(raw5 - check) < CHECK_TOL) if pd.notna(check) else np.nan)
    return pd.DataFrame(out, index=cat.index)


def build_event_returns(save=True):
    cols = ["call_id", "ticker", "call_date", "call_datetime_raw"] + [f"price_day_{d}" for d in range(-10, 11)]
    cat = pd.read_parquet(CATALOGUE, columns=cols)
    prices = pd.read_parquet(PRICES)
    ev = pd.concat([cat[["call_id"]], event_returns(cat, prices)], axis=1)
    failed = (ev["price_check"] == False).groupby(cat["ticker"]).sum()  # noqa: E712
    checked = ev["price_check"].notna().groupby(cat["ticker"]).sum()
    bad_tickers = set((failed / checked.clip(lower=1))[lambda x: x >= MAX_TICKER_FAIL].index)
    ev["usable"] = (ev["car5"].notna() & (ev["price_check"] != False)  # noqa: E712
                    & ~cat["ticker"].isin(bad_tickers))
    if save:
        ev.to_parquet(OUT, index=False)
    return ev


if __name__ == "__main__":
    ev = build_event_returns()
    n = len(ev)
    print(f"{n} calls -> {OUT}")
    print(f"  with car5: {ev['car5'].notna().sum()} ({ev['car5'].notna().mean():.0%})")
    print(f"  price_check failed: {(ev['price_check'] == False).sum()}")  # noqa: E712
    print(f"  usable: {ev['usable'].sum()} ({ev['usable'].mean():.0%})")
    print(ev["timing"].value_counts().to_string())
    print(ev[["beta", "sigma", "car5", "scar5", "raw_ret5"]].describe().round(4).to_string())
