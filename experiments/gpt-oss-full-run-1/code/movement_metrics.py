"""Does the model's buy / hold / sell line up with how the stock moved after the call?

    python movement_metrics.py "sample_openai/gpt-oss-120b_*.pkl" --out results3/movement
    python movement_metrics.py "anon_openai/gpt-oss-120b_*.pkl" --out results3/movement_anon --analysts

Movement of a call: scar5, the standardised cumulative abnormal return over the 5 trading
days after the call (market model, data_collection/build_event_returns.py). Only calls
flagged `usable` are scored.

Per prompt, with d = +1 buy / 0 hold / -1 sell:
  tau        Kendall tau-b between d and scar5 (the information coefficient; tie-aware and
             rank-based, so one extreme move can't dominate). 95% bootstrap interval.
  p          two-sided permutation p-value: d shuffled across calls; p_holm corrects it for
             testing every prompt
  hit_rate   share of buys followed by scar5 > 0 and sells by scar5 < 0, with a binomial
             test against 50% (abnormal returns are centred on zero)
  mean SCAR  of the calls the prompt rated buy, hold and sell
Across prompts, OLS with standard errors clustered by call (every prompt sees the same calls):
  pooled     scar5 ~ d + prompt                 slope of d = average predictive power
  slopes     scar5 ~ prompt + d x prompt        Wald test: are the prompts' slopes equal?
--analysts adds the analyst consensus (analyst_compare.py) as a benchmark row.
The "neutral" prompt (no emotion story, main.py) is the control the emotions are compared to.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import visualize as v
from visualize import plt, SURFACE, INK_2, MUTED, AXIS, SERIES, DECISION_COLORS

POSITION = {"buy": 1, "hold": 0, "sell": -1}
MOVE = "scar5"


# ------------------------------------------------------------------ data

def load(paths):
    """Model results joined with each call's 5-day abnormal return; usable calls only."""
    from llms import load_catalogue, EVENT_COLS
    res = v.load_results(paths, with_returns=False)
    cat = load_catalogue()
    if "car5" not in cat.columns:
        raise SystemExit("catalogue has no event returns: run data_collection/fetch_prices.py, "
                         "build_event_returns.py and build_catalogue.py")
    cols = ["call_id", "ticker", "call_date", "gics_sector"] + EVENT_COLS
    info = cat[cols].drop_duplicates(["ticker", "call_date"])
    res = res.drop(columns=[c for c in ("call_id", "car5", "scar5") if c in res.columns])
    df = res.merge(info, on=["ticker", "call_date"], how="left")
    df["d"] = df["decision"].map(POSITION)
    n0 = df["call_id"].nunique()
    df = df[(df["usable"] == True) & df["d"].notna()].copy()  # noqa: E712
    print(f"{df['call_id'].nunique()} of {n0} calls have a usable 5-day abnormal return")
    return df


def _prompts(df, min_calls=10):
    return [e for e in v._ordered(df["emotion"]) if (df["emotion"] == e).sum() >= min_calls]


# ------------------------------------------------------------------ per-prompt statistics

def tau_b(x, y):
    t = stats.kendalltau(x, y, variant="b").statistic
    return 0.0 if np.isnan(t) else t


def tau_test(d, y, n_perm=10000, n_boot=2000, rng=None):
    """tau-b, two-sided permutation p-value, and 95% bootstrap interval."""
    rng = rng or np.random.default_rng(0)
    d, y = np.asarray(d, float), np.asarray(y, float)
    t = tau_b(d, y)
    perm = np.array([tau_b(rng.permutation(d), y) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm) >= abs(t) - 1e-12) + 1) / (n_perm + 1)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = np.array([tau_b(d[i], y[i]) for i in idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return t, p, lo, hi


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def prompt_table(df, n_perm=10000, analysts=None):
    rng = np.random.default_rng(0)
    rows = []
    for e in _prompts(df):
        s = df[df["emotion"] == e]
        t, p, lo, hi = tau_test(s["d"], s[MOVE], n_perm=n_perm, rng=rng)
        bets = s[s["d"] != 0]
        hits = int(((bets["d"] > 0) & (bets[MOVE] > 0) | (bets["d"] < 0) & (bets[MOVE] < 0)).sum())
        rows.append(dict(
            strategy=e, kind="control" if e == v.NEUTRAL else "model", calls=len(s), tau=t, lo=lo, hi=hi, p=p,
            spearman=stats.spearmanr(s["d"], s[MOVE]).statistic,
            n_buy=int((s["d"] > 0).sum()), n_hold=int((s["d"] == 0).sum()), n_sell=int((s["d"] < 0).sum()),
            hit_rate=hits / len(bets) if len(bets) else np.nan,
            hit_p=stats.binomtest(hits, len(bets), 0.5).pvalue if len(bets) else np.nan,
            scar_buy=s.loc[s["d"] > 0, MOVE].mean(), scar_hold=s.loc[s["d"] == 0, MOVE].mean(),
            scar_sell=s.loc[s["d"] < 0, MOVE].mean()))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    if analysts is not None and len(analysts):
        a = analysts
        t, p, lo, hi = tau_test(a["analyst_score"], a[MOVE], n_perm=n_perm, rng=rng)
        out = pd.concat([out, pd.DataFrame([dict(
            strategy="analyst consensus", kind="benchmark", calls=len(a), tau=t, lo=lo, hi=hi, p=p,
            spearman=stats.spearmanr(a["analyst_score"], a[MOVE]).statistic)])], ignore_index=True)
    return out


# ------------------------------------------------------------------ pooled regression

def cluster_ols(y, X, groups):
    """OLS coefficients and cluster-robust covariance (CR1 small-sample correction)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    codes, uniq = pd.factorize(groups)
    for g in range(len(uniq)):
        sg = X[codes == g].T @ u[codes == g]
        meat += np.outer(sg, sg)
    n, k, G = len(y), X.shape[1], len(uniq)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, V, G


def regressions(df):
    """Pooled slope of scar5 on d, and a test of whether the slope differs by prompt."""
    prompts = _prompts(df)
    d = df[df["emotion"].isin(prompts)]
    dummies = pd.get_dummies(d["emotion"]).reindex(columns=prompts).astype(float).to_numpy()
    y, x, groups = d[MOVE].to_numpy(), d["d"].to_numpy(float), d["call_id"].to_numpy()

    # pooled: scar5 = prompt intercepts + b * d
    b, V, G = cluster_ols(y, np.column_stack([dummies, x]), groups)
    se = np.sqrt(V[-1, -1])
    pooled = dict(slope=b[-1], se=se, t=b[-1] / se, p=2 * stats.t.sf(abs(b[-1] / se), G - 1), clusters=G)

    # separate slopes: scar5 = prompt intercepts + b_k * d * prompt_k; H0: all b_k equal
    k = len(prompts)
    b, V, G = cluster_ols(y, np.column_stack([dummies, dummies * x[:, None]]), groups)
    slopes, Vs = b[k:], V[k:, k:]
    R = np.column_stack([np.ones(k - 1), -np.eye(k - 1)])  # b_1 - b_j = 0
    diff = R @ slopes
    W = float(diff @ np.linalg.pinv(R @ Vs @ R.T) @ diff)
    F = W / (k - 1)
    equal = dict(F=F, df1=k - 1, df2=G - 1, p=stats.f.sf(F, k - 1, G - 1))
    per = pd.DataFrame({"strategy": prompts, "slope": slopes, "se": np.sqrt(np.diag(Vs))})
    return pooled, equal, per


# ------------------------------------------------------------------ analyst benchmark

def analyst_scores(df):
    """Analyst consensus score (+2 strong buy .. -2 strong sell) per usable call, where covered."""
    from analyst_compare import analyst_view
    calls = df.drop_duplicates("call_id")[["call_id", "ticker", "call_date", MOVE]]
    views = [analyst_view(t, d) for t, d in zip(calls["ticker"], calls["call_date"])]
    calls["analyst_score"] = [x["analyst_score"] if x else np.nan for x in views]
    return calls.dropna(subset=["analyst_score"])


# ------------------------------------------------------------------ charts

def plot_tau(table, path=None):
    """Kendall tau-b of each prompt's decisions with the 5-day abnormal return."""
    t = table.copy()
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(t) + 1.8))
    y = np.arange(len(t))
    color = [SERIES[0] if k == "model" else INK_2 if k == "control" else SERIES[1] for k in t["kind"]]
    ax.hlines(y, t["lo"], t["hi"], color=color, linewidth=2, alpha=0.5, zorder=2)
    ax.scatter(t["tau"], y, color=color, s=64, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    x_right = np.nanmax(t["hi"]) + 0.04
    for yi, (tau, p, ph) in enumerate(zip(t["tau"], t["p"], t.get("p_holm", t["p"]))):
        ax.annotate(f"{tau:+.2f}", (tau, yi), xytext=(0, 8), textcoords="offset points", ha="center",
                    color=INK_2, fontsize=9)
        label = f"p = {p:.3f}" + (f" · Holm {ph:.2f}" if pd.notna(ph) and ph != p else "")
        ax.text(x_right, yi, label, va="center", ha="left", fontsize=9,
                color=INK_2 if (pd.notna(ph) and ph < 0.05) else MUTED)
    ax.set_xlim(right=x_right + 0.28)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, [f"{s}  (n={n})" for s, n in zip(t["strategy"], t["calls"])])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Kendall's tau-b: decision (sell < hold < buy) vs 5-day standardised abnormal return\n"
                  "line = 95% bootstrap interval · p = permutation test")
    ax.set_title("Do the model's calls line up with how the stock moved?")
    return v._finish(fig, path)


def plot_scar_by_decision(df, path=None):
    """Mean 5-day SCAR of the calls each prompt rated buy, hold and sell, with standard errors."""
    prompts = _prompts(df)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(prompts))
    for off, dec in [(-0.22, "buy"), (0, "hold"), (0.22, "sell")]:
        g = df[df["decision"] == dec].groupby("emotion")[MOVE]
        m, se, n = g.mean().reindex(prompts), g.sem().reindex(prompts), g.size().reindex(prompts)
        ax.errorbar(x + off, m, yerr=se, fmt="o", color=DECISION_COLORS[dec], markersize=7, capsize=0,
                    elinewidth=1.6, label=dec, markeredgecolor=SURFACE)
        for xi, mi, ni in zip(x + off, m, n):
            if pd.notna(mi):
                ax.annotate(f"{int(ni)}", (xi, mi), xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, prompts, rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean 5-day SCAR (± 1 s.e.)")
    ax.set_title("Abnormal return after the calls each prompt rated buy, hold and sell (number = calls)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return v._finish(fig, path)


# ------------------------------------------------------------------ report

def make_report(paths, out_dir, with_analysts=False, n_perm=10000):
    df = load(paths)
    os.makedirs(out_dir, exist_ok=True)
    analysts = analyst_scores(df) if with_analysts else None
    table = prompt_table(df, n_perm=n_perm, analysts=analysts)
    pooled, equal, per = regressions(df)
    table = table.merge(per, on="strategy", how="left")
    table.to_csv(os.path.join(out_dir, "movement_metrics.csv"), index=False)
    df.drop(columns="summary", errors="ignore").to_csv(os.path.join(out_dir, "calls_with_movement.csv"), index=False)
    for name, fig in [("tau_by_prompt", plot_tau(table)), ("scar_by_decision", plot_scar_by_decision(df))]:
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(table[["strategy", "calls", "tau", "lo", "hi", "p", "p_holm", "hit_rate", "hit_p",
                     "scar_buy", "scar_hold", "scar_sell", "slope", "se"]].to_string(index=False))
    print(f"\npooled slope of SCAR on decision: {pooled['slope']:+.3f} (s.e. {pooled['se']:.3f}, "
          f"t = {pooled['t']:.2f}, p = {pooled['p']:.3f}, {pooled['clusters']} calls)")
    print(f"prompts' slopes equal? F({equal['df1']}, {equal['df2']}) = {equal['F']:.2f}, p = {equal['p']:.3f}")
    print(f"-> {out_dir}")
    return df, table, pooled, equal


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results/movement")
    ap.add_argument("--analysts", action="store_true", help="add the analyst consensus as a benchmark")
    ap.add_argument("--perm", type=int, default=10000, help="permutations per test")
    a = ap.parse_args()
    make_report(a.paths, a.out, with_analysts=a.analysts, n_perm=a.perm)
