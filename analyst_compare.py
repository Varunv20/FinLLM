"""Compare the model's buy/hold/sell calls with Wall Street analysts, and score both with Sharpe.

    python analyst_compare.py "sample_openai/gpt-oss-120b_*.pkl" --out results2/analyst

Analyst view of a call: every firm's latest rating dated from a year before the call up to a
week after it (so a rating changed in reaction to the call counts), from yfinance's
upgrade/downgrade history (cached by fetch_analyst_ratings.py). Each rating is bucketed
strong buy / buy / hold / sell / strong sell; the call's consensus is the average bucket
(+2 .. -2) rounded to the nearest one. Yahoo's history only starts around 2012, so older
calls have no analyst view.

Return of a call: market-adjusted, from the last close before the market could react to the
call (the call-day close for calls after 4 pm, the previous trading day's close otherwise) to
the close 5 trading days later, minus the S&P 500's return over the same dates. Stock prices
come from the catalogue's price window, the index from yfinance (cached in cache/sp500.csv).
Moves over 50% are treated as bad price data and dropped. A strategy holds +1 (buy), -1 (sell)
or 0 (hold) of that return.

Sharpe = mean / std of the per-call strategy returns x sqrt(52) (each call is a one-week
bet), with a 95% bootstrap interval. Two comparisons make it meaningful:
  - random: the same prompt's decisions shuffled across calls (same buy/hold/sell mix, no
    information). p = share of shuffles with a Sharpe at least as high.
  - paired: Sharpe differences between strategies, bootstrapped on the same resampled calls.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import visualize as v
from visualize import plt, matplotlib, SURFACE, INK, INK_2, MUTED, AXIS, SERIES, DECISION_COLORS, BLUES

LOOKBACK_DAYS, AFTER_DAYS = 365, 7
HOLD_DAYS = 5          # trading days held after the entry close
MAX_MOVE = 0.5         # |raw return| above this is bad price data
SP500_CACHE = os.path.join("cache", "sp500.csv")
ANALYST_LEVELS = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
LEVEL_SCORE = {"strong_buy": 2, "buy": 1, "hold": 0, "sell": -1, "strong_sell": -2}
LEVEL_LABEL = {"strong_buy": "strong buy", "buy": "buy", "hold": "hold", "sell": "sell", "strong_sell": "strong sell"}
TO_3 = {"strong_buy": "buy", "buy": "buy", "hold": "hold", "sell": "sell", "strong_sell": "sell"}
POSITION = {"buy": 1, "hold": 0, "sell": -1}


# ------------------------------------------------------------------ data

def _call_hour(raw):
    """Hour of day (ET) from strings like "May 15, 2006, 4:30 p.m. EST", or NaN."""
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m", raw, re.I) if isinstance(raw, str) else None
    if not m:
        return np.nan
    return int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0) + int(m.group(2) or 0) / 60


def sp500_closes():
    """Daily S&P 500 closes, downloaded once and cached."""
    if os.path.exists(SP500_CACHE):
        s = pd.read_csv(SP500_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        s = yf.download("^GSPC", start="1999-01-01", end="2022-01-01", auto_adjust=True,
                        progress=False)["Close"].squeeze()
        os.makedirs(os.path.dirname(SP500_CACHE), exist_ok=True)
        s.to_csv(SP500_CACHE)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def call_returns(calls):
    """Raw, market and market-adjusted return of each call, plus when the call happened."""
    from llms import load_catalogue
    cat = load_catalogue()
    days = list(range(-10, 11))
    cols = [f"price_day_{i}" for i in days]
    p = calls[["ticker", "call_date"]].merge(
        cat[["ticker", "call_date", "call_datetime_raw"] + cols].drop_duplicates(["ticker", "call_date"]),
        on=["ticker", "call_date"], how="left")
    hour = p["call_datetime_raw"].map(_call_hour)
    timing = np.where(hour >= 16, "after close", np.where(hour < 9.5, "before open",
                      np.where(hour.notna(), "during market", "unknown")))
    spx = sp500_closes()
    raw, mkt = np.full(len(p), np.nan), np.full(len(p), np.nan)
    for i, row in enumerate(p[cols].to_numpy()):
        traded = [d for d, x in zip(days, row) if not np.isnan(x)]  # NaN = no trading that day
        last_before = 0 if timing[i] == "after close" else -1
        before = [d for d in traded if d <= last_before]
        after = [d for d in traded if d > last_before]
        if not before or len(after) < HOLD_DAYS:
            continue
        d0, d1 = before[-1], after[HOLD_DAYS - 1]
        raw[i] = row[days.index(d1)] / row[days.index(d0)] - 1
        t0, t1 = p["call_date"].iloc[i] + pd.Timedelta(days=d0), p["call_date"].iloc[i] + pd.Timedelta(days=d1)
        i0, i1 = spx.index.searchsorted(t0, "right") - 1, spx.index.searchsorted(t1, "right") - 1
        if i0 >= 0 and i1 > i0:
            mkt[i] = spx.iloc[i1] / spx.iloc[i0] - 1
    raw[np.abs(raw) > MAX_MOVE] = np.nan
    return pd.DataFrame({"raw_ret": raw, "mkt_ret": mkt, "ret": raw - mkt, "timing": timing})


def analyst_view(ticker, call_date):
    """Counts per rating level and the consensus level for one call, or None without coverage."""
    from fetch_analyst_ratings import fetch_upgrades_downgrades
    from llms import normalize_grade
    h = fetch_upgrades_downgrades(ticker)
    if h is None or h.empty or "GradeDate" not in h:
        return None
    dates = pd.to_datetime(h["GradeDate"]).dt.tz_localize(None)
    win = h[(dates >= call_date - pd.Timedelta(days=LOOKBACK_DAYS))
            & (dates <= call_date + pd.Timedelta(days=AFTER_DAYS))].assign(_d=dates)
    if win.empty:
        return None
    latest = win.sort_values("_d").groupby("Firm").tail(1)
    levels = [normalize_grade(g) for g in latest["ToGrade"]]
    levels = [l for l in levels if l in LEVEL_SCORE]
    if not levels:
        return None
    mean = float(np.mean([LEVEL_SCORE[l] for l in levels]))
    consensus = min(LEVEL_SCORE, key=lambda l: abs(LEVEL_SCORE[l] - mean))
    out = {f"n_{l}": levels.count(l) for l in ANALYST_LEVELS}
    out.update(n_firms=len(levels), analyst_score=mean, analyst=consensus, analyst3=TO_3[consensus])
    return out


def build(paths):
    """Model results joined with the analyst view and week-after return of each call."""
    res = v.load_results(paths, with_returns=False)
    calls = res.drop_duplicates(["ticker", "call_date"])[["ticker", "call_date"]].reset_index(drop=True)
    calls = pd.concat([calls, call_returns(calls)], axis=1)
    views = [analyst_view(t, d) for t, d in zip(calls["ticker"], calls["call_date"])]
    calls = pd.concat([calls, pd.DataFrame([x or {} for x in views])], axis=1)
    return res.merge(calls, on=["ticker", "call_date"], how="left")


# ------------------------------------------------------------------ metrics

ANNUAL = np.sqrt(52)


def _sharpe_cols(x):
    """Annualised Sharpe of each column of x along axis -2 (NaN-aware)."""
    m, sd = np.nanmean(x, axis=-2), np.nanstd(x, axis=-2, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, m / sd * ANNUAL, np.nan)


def sharpe(r, n_boot=2000, seed=0):
    """Annualised Sharpe of one-week bets, with a 95% bootstrap interval."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = _sharpe_cols(r[rng.integers(0, len(r), (n_boot, len(r)))][..., None])[:, 0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return _sharpe_cols(r[:, None])[0], lo, hi


def _strategy_matrix(df, covered_only=False):
    """Per-call positions (calls x strategies) and the market-adjusted return of each call."""
    d = df.dropna(subset=["ret"])
    if covered_only:
        d = d.dropna(subset=["analyst3"])
    emotions = [e for e in v._ordered(d["emotion"]) if (d["emotion"] == e).sum() >= 10]  # skip unfinished runs
    pos = (d[d["emotion"].isin(emotions)].assign(p=lambda x: x["decision"].map(POSITION).fillna(0))
           .pivot_table(index=["ticker", "call_date"], columns="emotion", values="p")[emotions])
    calls = d.drop_duplicates(["ticker", "call_date"]).set_index(["ticker", "call_date"]).loc[pos.index]
    pos["always long"] = 1.0
    pos["analyst consensus"] = calls["analyst3"].map(POSITION)  # NaN where there is no coverage
    return pos, calls["ret"].to_numpy()


def _shuffle_test(pos, r, observed, rng, n_boot):
    """95% range and p-value of Sharpes from the positions shuffled across calls."""
    shuf = np.stack([pos[rng.permutation(len(pos))] for _ in range(n_boot)]) * r
    null = _sharpe_cols(shuf[..., None])[:, 0]
    return np.nanpercentile(null, [2.5, 97.5]), np.mean(null >= observed)


def strategy_table(df, covered_only=False, n_boot=5000, seed=0, drop_top=3):
    """Sharpe of each strategy with four yardsticks.

    lo/hi            95% bootstrap interval of the Sharpe itself
    rand_lo/rand_hi  95% range of Sharpes from the same decisions shuffled across calls
    p_random         share of those shuffles scoring at least as high (low = beats chance)
    vs_long, lo, hi  paired bootstrap difference from always long, on the same calls
    *_drop           Sharpe and p_random again without the `drop_top` calls that earn the
                     model prompts the most in total (is the result a few lucky calls?)
    """
    pos, r = _strategy_matrix(df, covered_only)
    names = list(pos.columns)
    n_model = len(names) - 2
    P = pos.to_numpy()
    S = P * r[:, None]
    n = len(r)
    rng = np.random.default_rng(seed)
    boots = np.stack([_sharpe_cols(S[i]) for i in rng.integers(0, n, (n_boot, n))])
    obs = _sharpe_cols(S)
    keep = np.ones(n, bool)
    keep[np.argsort(np.nansum(S[:, :n_model], axis=1))[-drop_top:]] = False
    obs_drop = _sharpe_cols(S[keep])
    rows = []
    for j, name in enumerate(names):
        ok = ~np.isnan(P[:, j])
        rand, rand_drop = (np.full(2, np.nan), np.nan), (None, np.nan)
        if name != "always long":
            rand = _shuffle_test(P[ok, j], r[ok], obs[j], rng, n_boot)
            rand_drop = _shuffle_test(P[ok & keep, j], r[ok & keep], obs_drop[j], rng, n_boot)
        diff = boots[:, j] - boots[:, names.index("always long")]
        rows.append(dict(kind="model" if j < n_model else "baseline", strategy=name,
                         sharpe=obs[j], lo=np.nanpercentile(boots[:, j], 2.5),
                         hi=np.nanpercentile(boots[:, j], 97.5), rand_lo=rand[0][0], rand_hi=rand[0][1],
                         p_random=rand[1], vs_long=obs[j] - obs[names.index("always long")],
                         vs_long_lo=np.nanpercentile(diff, 2.5), vs_long_hi=np.nanpercentile(diff, 97.5),
                         sharpe_drop=obs_drop[j], p_random_drop=rand_drop[1],
                         mean_ret=np.nanmean(S[:, j]), calls=int(ok.sum())))
    return pd.DataFrame(rows)


def pairwise_table(df, covered_only=False, n_boot=5000, seed=0):
    """Paired bootstrap of the Sharpe difference between every two emotion prompts."""
    pos, r = _strategy_matrix(df, covered_only)
    emotions = list(pos.columns[:-2])
    S = pos[emotions].to_numpy() * r[:, None]
    rng = np.random.default_rng(seed)
    boots = np.stack([_sharpe_cols(S[i]) for i in rng.integers(0, len(r), (n_boot, len(r)))])
    obs = _sharpe_cols(S)
    rows = []
    for i, a in enumerate(emotions):
        for j, b in enumerate(emotions):
            if i < j:
                lo, hi = np.nanpercentile(boots[:, i] - boots[:, j], [2.5, 97.5])
                rows.append((a, b, obs[i] - obs[j], lo, hi, lo > 0 or hi < 0))
    return pd.DataFrame(rows, columns=["a", "b", "diff", "lo", "hi", "significant"])


def agreement_table(df):
    """How often the model's buy/hold/sell matches the analyst consensus, per emotion."""
    d = df.dropna(subset=["analyst3"])
    rows = []
    for e in v._ordered(d["emotion"]):
        s = d[d["emotion"] == e]
        if len(s) < 10:
            continue
        rows.append((e, (s["decision"] == s["analyst3"]).mean(), len(s)))
    return pd.DataFrame(rows, columns=["emotion", "agreement", "calls"])


# ------------------------------------------------------------------ charts

def plot_analyst_mix(df, path=None):
    """Every analyst firm's rating behind the covered calls, by level."""
    calls = df.drop_duplicates(["ticker", "call_date"]).dropna(subset=["analyst"])
    counts = calls[[f"n_{l}" for l in ANALYST_LEVELS]].sum().astype(int).to_numpy()
    colors = [BLUES[6], BLUES[4], "#a3a29c", "#e66767", "#c43c3c"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar([LEVEL_LABEL[l] for l in ANALYST_LEVELS], counts, color=colors, width=0.7, zorder=2)
    ax.bar_label(bars, labels=[f"{n}" for n in counts], padding=4, color=INK_2, fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("firm ratings")
    ax.set_title(f"Analyst ratings behind the {len(calls)} covered calls ({counts.sum()} firm ratings)")
    return v._finish(fig, path)


def plot_model_vs_analyst(df, path=None):
    """Pooled over emotions: model decision (rows) vs analyst consensus (columns that occur)."""
    d = df.dropna(subset=["analyst"])
    cols = [l for l in ANALYST_LEVELS if (d["analyst"] == l).any()]
    cm = pd.crosstab(d["decision"], d["analyst"]).reindex(index=v.DECISIONS, columns=cols, fill_value=0)
    share = cm / cm.sum(axis=0)  # of the calls analysts rated X, how the model split
    fig, ax = plt.subplots(figsize=(3 + 2.2 * len(cols), 4.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(share.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = share.iat[i, j]
            ax.text(j, i, f"{val:.0%}\n({cm.iat[i, j]})", ha="center", va="center", fontsize=12,
                    color="white" if val > 0.5 else INK)
    ax.set_xticks(range(len(cols)), [f"analysts: {LEVEL_LABEL[l]}\n({cm[l].sum()} model calls)" for l in cols])
    ax.set_yticks(range(3), [f"model: {d}" for d in v.DECISIONS])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("What the model said, given the analyst consensus")
    return v._finish(fig, path)


def plot_agreement(df, path=None):
    """Share of calls where the model's buy/hold/sell equals the analysts' (strong buy = buy)."""
    t = agreement_table(df)
    d = df.dropna(subset=["analyst3"]).drop_duplicates(["ticker", "call_date"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bars = ax.bar(t["emotion"], t["agreement"], color=SERIES[0], width=0.72, zorder=2)
    ax.bar_label(bars, labels=[f"{a:.0%}" for a in t["agreement"]], label_type="center", color="white", fontsize=11)
    # the agreement you'd get by always answering what analysts most often say
    top = d["analyst3"].value_counts(normalize=True)
    v._baseline(ax, top.iloc[0], label=f"always '{top.index[0]}'")
    ax.set_ylim(0, 1)
    v._pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("agrees with analysts")
    ax.set_title(f"How often the model agrees with analyst consensus (n={int(t['calls'].max())} calls each)")
    return v._finish(fig, path)


def plot_sharpe(table, path=None, title="Sharpe ratio of acting on each call"):
    """Sharpe per strategy with its 95% interval, over the range chance alone gives that decision mix."""
    t = table.copy()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = np.arange(len(t))
    color = [SERIES[0] if k == "model" else (MUTED if s == "always long" else SERIES[1])
             for k, s in zip(t["kind"], t["strategy"])]
    for yi, lo, hi in zip(y, t["rand_lo"], t["rand_hi"]):
        if not np.isnan(lo):
            ax.add_patch(matplotlib.patches.Rectangle((lo, yi - 0.32), hi - lo, 0.64, color="#e4e3dd",
                                                      linewidth=0, zorder=1))
    ax.hlines(y, t["lo"], t["hi"], color=color, linewidth=2, alpha=0.5, zorder=2)
    ax.scatter(t["sharpe_drop"], y, facecolor=SURFACE, edgecolor=color, s=46, linewidth=1.6, zorder=3)
    ax.scatter(t["sharpe"], y, color=color, s=70, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    for yi, s in zip(y, t["sharpe"]):
        ax.annotate(f"{s:+.2f}", (s, yi), xytext=(0, 9), textcoords="offset points", ha="center",
                    color=INK_2, fontsize=9.5)
    x_right = np.nanmax(t[["hi", "rand_hi"]].to_numpy()) + 0.3
    for yi, pv in zip(y, t["p_random"]):
        if not np.isnan(pv):
            ax.text(x_right, yi, f"p = {pv:.3f}", va="center", ha="left", color=INK_2 if pv < 0.05 else MUTED,
                    fontsize=9.5, fontweight="bold" if pv < 0.05 else "normal")
    ax.set_xlim(right=x_right + 1.0)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, [f"{s}  (n={n})" for s, n in zip(t["strategy"], t["calls"])])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("annualised Sharpe of market-adjusted 5-day bets · line = 95% bootstrap interval")
    ax.set_title(title)
    handles = [matplotlib.lines.Line2D([], [], marker="o", linestyle="", color=c, label=l, markersize=8)
               for c, l in [(SERIES[0], "model, by emotion prompt"), (SERIES[1], "analyst consensus"),
                            (MUTED, "always long")]]
    handles.append(matplotlib.lines.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
                                           markeredgecolor=INK_2, label="without the 3 best calls", markersize=7))
    handles.append(matplotlib.patches.Patch(color="#e4e3dd", label="same decisions at random (95%)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    ax.text(1.0, -0.24, "p = share of random shuffles scoring at least as high", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=9)
    return v._finish(fig, path)


def plot_contribution(df, path=None):
    """Total market-adjusted return each prompt earned from its buys and from its sells."""
    d = df.dropna(subset=["ret"])
    emotions = [e for e in v._ordered(d["emotion"]) if (d["emotion"] == e).sum() >= 10]
    d = d[d["emotion"].isin(emotions)].assign(s=lambda x: x["decision"].map(POSITION).fillna(0) * x["ret"] * 100)
    t = d.pivot_table(index="emotion", columns="decision", values="s", aggfunc="sum").reindex(emotions).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(t))
    for col, off in [("buy", -0.2), ("sell", 0.2)]:
        vals = t.get(col, pd.Series(0, index=t.index))
        ax.bar(x + off, vals, width=0.38, color=DECISION_COLORS[col], label=f"from {col}s", zorder=2)
        for xi, val in zip(x + off, vals):
            ax.annotate(f"{val:+.0f}", (xi, val), xytext=(0, 3 if val >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, t.index, rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("summed market-adjusted return, %")
    ax.set_title("Where each prompt's return comes from")
    ax.legend(loc="upper left", ncol=2)
    return v._finish(fig, path)


def plot_cumulative(df, path=None):
    """Growth of the per-call strategy returns, in call-date order."""
    d = df.dropna(subset=["ret", "call_date"]).sort_values("call_date")
    fig, ax = plt.subplots(figsize=(11, 5))
    emotions = [e for e in v._ordered(d["emotion"]) if (d["emotion"] == e).sum() >= 10]
    for e in emotions:
        s = d[d["emotion"] == e]
        ax.plot(s["call_date"], (s["decision"].map(POSITION).fillna(0) * s["ret"]).cumsum() * 100,
                color=SERIES[0], alpha=0.35, linewidth=1.4)
    calls = d.drop_duplicates(["ticker", "call_date"])
    ax.plot(calls["call_date"], calls["ret"].cumsum() * 100, color=INK_2, linestyle=(0, (4, 3)),
            linewidth=2, label="always long")
    a = calls.dropna(subset=["analyst3"])
    ax.plot(a["call_date"], (a["analyst3"].map(POSITION) * a["ret"]).cumsum() * 100, color=SERIES[1],
            linewidth=2.2, label="analyst consensus")
    ax.plot([], [], color=SERIES[0], alpha=0.6, label="model (one line per emotion)")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("cumulative market-adjusted return, %\n(sum of 5-day bets)")
    ax.set_title("Acting on every call, in date order")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    return v._finish(fig, path)


def make_report(paths, out_dir):
    df = build(paths)
    os.makedirs(out_dir, exist_ok=True)
    df.drop(columns="summary", errors="ignore").to_csv(os.path.join(out_dir, "calls_with_analysts.csv"), index=False)
    s_all = strategy_table(df)
    s_cov = strategy_table(df, covered_only=True)
    agree = agreement_table(df)
    pairwise_table(df).to_csv(os.path.join(out_dir, "sharpe_pairwise.csv"), index=False)
    s_all.to_csv(os.path.join(out_dir, "sharpe_all_calls.csv"), index=False)
    s_cov.to_csv(os.path.join(out_dir, "sharpe_analyst_covered.csv"), index=False)
    agree.to_csv(os.path.join(out_dir, "agreement.csv"), index=False)
    for name, fig in [
        ("decision_mix", v.plot_decision_mix(df[df.groupby("emotion")["ticker"].transform("size") >= 10])),
        ("analyst_mix", plot_analyst_mix(df)),
        ("model_vs_analyst", plot_model_vs_analyst(df)),
        ("agreement", plot_agreement(df)),
        ("sharpe_all", plot_sharpe(s_all, title="Sharpe ratio of acting on each call (all calls)")),
        ("sharpe_covered", plot_sharpe(s_cov, title="Sharpe ratio, calls with analyst coverage only")),
        ("contribution", plot_contribution(df)),
        ("cumulative", plot_cumulative(df)),
    ]:
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(s_all.to_string(index=False), "\n")
        print(s_cov.to_string(index=False), "\n")
        print(agree.to_string(index=False))
    calls = df.drop_duplicates(["ticker", "call_date"])
    print(f"\n{len(calls)} calls, {calls['analyst'].notna().sum()} with analyst coverage, "
          f"{calls['ret'].notna().sum()} with a price window -> {out_dir}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results/analyst")
    a = ap.parse_args()
    make_report(a.paths, a.out)
