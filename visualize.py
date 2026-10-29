"""Matplotlib summaries of run_all results (the {model}_{i}.pkl files main.py writes).

    from visualize import load_results, make_report
    df = load_results("openai/gpt-oss-120b_*.pkl")
    make_report(df, "results/images/gpt-oss-120b")

or from the shell:
    python visualize.py "openai/gpt-oss-120b_*.pkl" --out results/images/gpt-oss-120b

A call counts as correct when a buy is followed by a price rise over the week after the
call (score > 0) or a sell by a fall (score < 0). Holds and flat weeks are left out of
accuracy. Every accuracy chart shows the "always buy" baseline, since most weeks go up.
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prompts import emotion_stories

# emotion order main.py runs in: file suffix _i is emotions[i]
EMOTIONS = list(emotion_stories)
NEUTRAL = "neutral"  # the prompt with no emotion story (main.py runs it as the control)

# ---- palette (validated: categorical slots pass CVD/normal-vision checks on #fcfcfb) ----
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# buy / hold / sell is a polarity, so it uses the diverging pair with a gray midpoint
DECISION_COLORS = {"buy": "#2a78d6", "hold": "#a3a29c", "sell": "#e34948"}
DECISIONS = ["buy", "hold", "sell"]
BLUES = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 7,
})


# ------------------------------------------------------------------ loading

# parse_decision lives in llms (main.py uses it without matplotlib); re-exported here
from llms import parse_decision  # noqa: E402,F401


def _label_from_path(path):
    """('gpt-oss-120b', 'panic') from '.../gpt-oss-120b_9.pkl'; '..._neutral.pkl' is the
    no-emotion control."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("_" + NEUTRAL):
        return stem[:-len(NEUTRAL) - 1], NEUTRAL
    m = re.match(r"(.+)_(\d+)$", stem)
    if not m:
        return stem, "unknown"
    i = int(m.group(2))
    return m.group(1), EMOTIONS[i] if i < len(EMOTIONS) else f"emotion {i}"


def load_results(paths, with_returns=True):
    """One df from one or more result pickles (a path, a glob, or a list of either).

    Adds: model, emotion (from the file name), decision_raw (the stored column),
    decision (parsed), direction (up/down/flat), correct (bool, NaN for hold/flat), and,
    if with_returns, ret_pct: score as % of the call-day price, plus sector / index from
    the catalogue."""
    if isinstance(paths, str):
        paths = [paths]
    files = sorted({f for p in paths for f in (glob.glob(p) or [p])})
    if not files:
        raise FileNotFoundError(f"No result pickles match {paths}")
    frames = []
    for f in files:
        d = pd.read_pickle(f)
        d["model"], d["emotion"] = _label_from_path(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["decision_raw"] = df["decision"]
    df["decision"] = [parse_decision(s, r) for s, r in zip(df["summary"], df["decision_raw"])]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["direction"] = np.select([df["score"] > 0, df["score"] < 0], ["up", "down"], "flat")
    df["call_date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str),
        format="%Y-%b-%d", errors="coerce")
    scored = df["decision"].isin(["buy", "sell"]) & df["direction"].isin(["up", "down"])
    df["correct"] = pd.Series(np.where(
        (df["decision"] == "buy") & (df["direction"] == "up")
        | (df["decision"] == "sell") & (df["direction"] == "down"), True, False)).where(scored)

    if with_returns:
        df = _add_catalogue_fields(df)
    return df


def _add_catalogue_fields(df):
    try:
        from llms import load_catalogue
        cat = load_catalogue()
    except Exception as e:  # catalogue not built on this machine: charts that need it are skipped
        print(f"[visualize] no catalogue ({e}); skipping % returns and sector charts")
        return df
    cols = ["ticker", "call_date", "price_day_0", "gics_sector", "sp_index"]
    if "car5" in cat.columns:
        cols += ["car5", "raw_ret5", "usable"]
    info = cat[cols].drop_duplicates(["ticker", "call_date"]).reset_index(drop=True)
    df = df.drop(columns=[c for c in ("car5",) if c in df.columns]).merge(info, on=["ticker", "call_date"], how="left")
    if "car5" in df.columns:
        # movement = 5-day abnormal return after the call (data_collection/build_event_returns.py),
        # not the stored score
        df["ret_pct"] = 100 * df["raw_ret5"]
        move = df["car5"].where(df["usable"] == True)  # noqa: E712
        df["direction"] = np.select([move > 0, move < 0], ["up", "down"], "flat")
        scored = df["decision"].isin(["buy", "sell"]) & df["direction"].isin(["up", "down"])
        df["correct"] = pd.Series(np.where(
            (df["decision"] == "buy") & (df["direction"] == "up")
            | (df["decision"] == "sell") & (df["direction"] == "down"), True, False)).where(scored)
    else:
        df["ret_pct"] = 100 * df["score"] / df["price_day_0"]
    return df


# ------------------------------------------------------------------ tables

def summary_table(df, by=("model", "emotion")):
    """Calls, accuracy, always-buy baseline, and decision mix per group."""
    by = list(by)
    g = df.groupby(by, sort=False)
    out = pd.DataFrame({
        "calls": g.size(),
        "scored": g["correct"].count(),
        "accuracy": g["correct"].mean(),
        "always_buy": g["direction"].apply(lambda s: (s == "up").sum() / s.isin(["up", "down"]).sum()),
    })
    for d in DECISIONS:
        out[f"{d}_share"] = g["decision"].apply(lambda s, d=d: (s == d).mean())
    out["edge_vs_always_buy"] = out["accuracy"] - out["always_buy"]
    return out.reset_index()


# ------------------------------------------------------------------ helpers

def _ordered(values):
    """The neutral control first, then emotions in main.py's order, anything else after."""
    vals = list(pd.unique(values))
    return sorted(vals, key=lambda v: (-1 if v == NEUTRAL else EMOTIONS.index(v) if v in EMOTIONS
                                       else len(EMOTIONS), str(v)))


def _pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _finish(fig, path):
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _baseline(ax, value, label="always buy", axis="y"):
    line = ax.axhline if axis == "y" else ax.axvline
    line(value, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    if axis == "y":
        # outside the plot on the right, so it never sits on a bar
        ax.annotate(f"{label}\n{value:.0%}", (1, value), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    color=INK_2, fontsize=9)
    else:
        ax.annotate(f"{label} {value:.0%}", (value, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                    color=INK_2, fontsize=9)


def _base_rate(df):
    d = df["direction"]
    return (d == "up").sum() / max(d.isin(["up", "down"]).sum(), 1)


# ------------------------------------------------------------------ charts

def plot_accuracy_by_emotion(df, path=None):
    """Accuracy per emotion prompt (one bar per model when several are loaded)."""
    emotions = _ordered(df["emotion"])
    models = list(pd.unique(df["model"]))
    acc = df.groupby(["emotion", "model"])["correct"].mean().unstack("model").reindex(emotions)
    n = df.groupby("emotion")["correct"].count().reindex(emotions)

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(emotions) * max(1, len(models) / 1.5)), 4.5))
    x = np.arange(len(emotions))
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        xs = x - 0.4 + width * (j + 0.5)
        bars = ax.bar(xs, acc[m], width=width - 0.03, color=SERIES[j % len(SERIES)], label=m, zorder=2)
        if len(models) == 1:
            # mid-bar, well clear of the baseline that runs near the bar tops
            ax.bar_label(bars, labels=[f"{v:.0%}" if pd.notna(v) and v >= 0.08 else "" for v in acc[m]],
                         label_type="center", color="white", fontsize=9.5)
    _baseline(ax, _base_rate(df))
    ax.set_xticks(x, [f"{e}\n(n={int(k)})" for e, k in zip(emotions, n.fillna(0))], fontsize=9)
    ax.set_ylim(0, 1)
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy (buy→up, sell→down)")
    title = "Accuracy by emotion prompt"
    ax.set_title(title if len(models) > 1 else f"{title} · {models[0]}")
    if len(models) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=min(len(models), 4))
    return _finish(fig, path)


def plot_decision_mix(df, path=None):
    """Share of buy / hold / sell per emotion: shows whether a prompt just shifts the bias."""
    emotions = _ordered(df["emotion"])
    mix = (pd.crosstab(df["emotion"], df["decision"], normalize="index")
           .reindex(index=emotions, columns=DECISIONS, fill_value=0))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(emotions) + 1.6))
    left = np.zeros(len(emotions))
    for d in DECISIONS:
        vals = mix[d].to_numpy()
        ax.barh(emotions, vals, left=left, color=DECISION_COLORS[d], label=d,
                edgecolor=SURFACE, linewidth=2, height=0.7, zorder=2)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 0.08:  # label only segments wide enough to hold text
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if d != "hold" else INK)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _pct_axis(ax, "x")
    ax.grid(False)
    ax.set_title("Decision mix by emotion prompt")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3)
    return _finish(fig, path)


def plot_confusion(df, path=None, emotion=None):
    """Counts of decision vs the price direction that followed."""
    d = df if emotion is None else df[df["emotion"] == emotion]
    d = d[d["direction"].isin(["up", "down"])]
    cm = pd.crosstab(d["decision"], d["direction"]).reindex(index=DECISIONS, columns=["up", "down"], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    vmax = max(cm.to_numpy().max(), 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(cm.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            ax.text(j, i, f"{v}\n{v / max(cm.to_numpy().sum(), 1):.0%}", ha="center", va="center",
                    fontsize=10, color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks([0, 1], ["price went up", "price went down"])
    ax.set_yticks(range(3), DECISIONS)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("week after the call")
    ax.set_ylabel("model decision")
    ax.set_title("Decision vs outcome" + (f" · {emotion}" if emotion else ""))
    return _finish(fig, path)


def plot_accuracy_over_time(df, path=None, min_calls=5):
    """Accuracy by call year, against the always-buy rate for the same year."""
    d = df.dropna(subset=["call_date"])
    g = d.groupby(d["call_date"].dt.year)
    t = pd.DataFrame({"accuracy": g["correct"].mean(), "n": g["correct"].count(),
                      "always_buy": g["direction"].apply(lambda s: (s == "up").sum() / max(s.isin(["up", "down"]).sum(), 1))})
    t = t[t["n"] >= min_calls]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if t.empty:
        ax.text(0.5, 0.5, f"no year has {min_calls}+ scored calls yet", ha="center", transform=ax.transAxes, color=INK_2)
    else:
        ax.plot(t.index, t["accuracy"], color=SERIES[0], marker="o", label="model", zorder=3)
        ax.plot(t.index, t["always_buy"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.5, label="always buy")
        ax.annotate("model", (t.index[-1], t["accuracy"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
        ax.annotate("always buy", (t.index[-1], t["always_buy"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax.set_ylim(0, 1)
    _pct_axis(ax)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by year of the call")
    return _finish(fig, path)


def plot_accuracy_by_group(df, group="ticker", path=None, top_n=20, min_calls=3):
    """Accuracy per ticker / gics_sector / sp_index: the best and worst `top_n`, with n."""
    if group not in df:
        return None
    g = df.groupby(group)["correct"]
    t = pd.DataFrame({"accuracy": g.mean(), "n": g.count()})
    t = t[t["n"] >= min_calls].sort_values("accuracy")
    if len(t) > top_n:
        t = pd.concat([t.head(top_n // 2), t.tail(top_n - top_n // 2)])
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.6))
    bars = ax.barh([f"{i}  (n={n})" for i, n in zip(t.index, t["n"])], t["accuracy"],
                   color=SERIES[0], height=0.7, zorder=2)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in t["accuracy"]], padding=3, color=INK_2, fontsize=8.5)
    _baseline(ax, _base_rate(df), axis="x")
    ax.set_xlim(0, 1.08)
    _pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    label = {"ticker": "ticker", "gics_sector": "sector", "sp_index": "index"}.get(group, group)
    ax.set_title(f"Accuracy by {label}" + (f" (best and worst {top_n})" if len(g.size()) > top_n else ""))
    return _finish(fig, path)


def plot_returns_by_decision(df, path=None):
    """Distribution of the week-after return for each decision."""
    col, unit = ("ret_pct", "%") if "ret_pct" in df and df["ret_pct"].notna().any() else ("score", "$")
    data = [df.loc[df["decision"] == d, col].dropna() for d in DECISIONS]
    keep = [i for i, v in enumerate(data) if len(v)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if keep:
        bp = ax.boxplot([data[i] for i in keep], vert=False, widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS))
        for patch, i in zip(bp["boxes"], keep):
            patch.set(facecolor=DECISION_COLORS[DECISIONS[i]], edgecolor=SURFACE, alpha=0.9)
        ax.set_yticks(range(1, len(keep) + 1), [f"{DECISIONS[i]}  (n={len(data[i])})" for i in keep])
        for k, i in enumerate(keep, start=1):
            ax.annotate(f"mean {data[i].mean():+.2f}{unit}", (1, k), xycoords=("axes fraction", "data"),
                        xytext=(-4, 10), textcoords="offset points", ha="right", color=INK_2, fontsize=8.5)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("return over the week after the call" + (" (% of call-day price)" if unit == "%" else " ($ per share)"))
    ax.set_title("Week-after return by decision")
    return _finish(fig, path)


def plot_strategy_returns(df, path=None):
    """Cumulative return of following the model (long on buy, short on sell, out on hold)
    against buying every call, per emotion. Equal weight per call, in call-date order."""
    col, unit = ("ret_pct", "%") if "ret_pct" in df and df["ret_pct"].notna().any() else ("score", "$")
    d = df.dropna(subset=["call_date", col]).sort_values("call_date")
    sign = d["decision"].map({"buy": 1, "sell": -1, "hold": 0}).fillna(0)
    d = d.assign(strategy=sign * d[col])
    emotions = _ordered(d["emotion"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, e in enumerate(emotions[:len(SERIES)]):
        s = d[d["emotion"] == e]
        ax.plot(s["call_date"], s["strategy"].cumsum(), color=SERIES[j], linewidth=1.8, label=e)
    base = d.drop_duplicates(["ticker", "call_date"])
    ax.plot(base["call_date"], base[col].cumsum(), color=INK_2, linestyle=(0, (4, 3)), linewidth=1.5,
            label="buy every call")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel(f"cumulative return ({unit}, summed per call)")
    ax.set_title("Following the model vs buying every call")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return _finish(fig, path)


def make_report(df, out_dir="results/images", show_table=True):
    """Save every chart to out_dir and return the summary table."""
    if isinstance(df, (str, list)):
        df = load_results(df)
    os.makedirs(out_dir, exist_ok=True)
    charts = {
        "accuracy_by_emotion": plot_accuracy_by_emotion,
        "decision_mix": plot_decision_mix,
        "decision_vs_outcome": plot_confusion,
        "accuracy_by_year": plot_accuracy_over_time,
        "accuracy_by_ticker": lambda d, path: plot_accuracy_by_group(d, "ticker", path),
        "accuracy_by_sector": lambda d, path: plot_accuracy_by_group(d, "gics_sector", path, min_calls=5),
        "returns_by_decision": plot_returns_by_decision,
        "strategy_returns": plot_strategy_returns,
    }
    for name, fn in charts.items():
        fig = fn(df, path=os.path.join(out_dir, f"{name}.png"))
        if fig is not None:
            plt.close(fig)
    table = summary_table(df)
    table.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    if show_table:
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
            print(table.to_string(index=False))
        changed = (df["decision"] != df["decision_raw"]).mean()
        print(f"\nparsed decision differs from the stored buy/sell word count on {changed:.0%} of calls")
    print(f"charts and summary.csv saved to {out_dir}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="result pickles or globs, e.g. 'openai/gpt-oss-120b_*.pkl'")
    ap.add_argument("--out", default="results/images", help="folder for the PNGs and summary.csv")
    args = ap.parse_args()
    make_report(load_results(args.paths), args.out)
