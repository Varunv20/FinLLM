"""Do the emotion prompts differ from each other? Every prompt is scored on the SAME calls.

    python emotion_compare.py "anon_sum_openai/gpt-oss-20b_*.pkl" --out results_sum/emotions

Calls: only those where every prompt gave a decision (a reply cut off before its decision in
any prompt drops the call from all of them), and with a usable 5-day abnormal return. So n is
the same for every prompt and every difference is a paired comparison.

Per prompt, with d = +1 buy / 0 hold / -1 sell:
  tau_scar     Kendall tau-b of d with scar5 (5-day standardised abnormal return)
  buy / hold / sell   share of calls with each decision
  tau_analyst  Kendall tau-b of d with the analyst consensus score (+2 strong buy .. -2 strong
               sell; analyst_compare.analyst_view), on the calls analysts cover
Error bars: 95% bootstrap intervals, resampling calls (the same resample for every prompt).
Differences: for each pair of prompts, the difference over its bootstrap standard error (same
resampled calls for both) gives a two-sided z-test p; Holm-corrected over all 66 pairs (heatmaps) and
over the 11 comparisons with neutral (stars on the charts). A Wald test on the bootstrap
covariance asks whether all 12 prompts are equal at once; for the buy share, Cochran's Q too.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import visualize as v
from visualize import plt, SURFACE, INK, INK_2, MUTED, AXIS, SERIES, DECISION_COLORS
from movement_metrics import holm, POSITION, MOVE

NEUTRAL = "neutral"


# ------------------------------------------------------------------ data

def load(paths):
    """decisions (calls x prompts, +1/0/-1), scar5 and analyst score, on the common calls."""
    from llms import load_catalogue
    res = v.load_results(paths, with_returns=False)
    cat = load_catalogue()
    info = cat[["call_id", "ticker", "call_date", "usable", MOVE]].drop_duplicates(["ticker", "call_date"])
    df = res.drop(columns=[c for c in ("call_id", "car5", "scar5") if c in res.columns]).merge(
        info, on=["ticker", "call_date"], how="left")
    df = df[df["usable"] == True]  # noqa: E712
    prompts = v._ordered(df["emotion"])
    wide = df.assign(d=df["decision"].map(POSITION)).pivot_table(
        index="call_id", columns="emotion", values="d", aggfunc="first")[prompts]
    n_all = len(wide)
    wide = wide.dropna()
    print(f"{len(wide)} of {n_all} calls have a decision under all {len(prompts)} prompts")
    calls = df.drop_duplicates("call_id").set_index("call_id").loc[wide.index, ["ticker", "call_date", MOVE]]
    return wide, calls, prompts


def analyst_scores(calls):
    from analyst_compare import analyst_view
    views = [analyst_view(t, d) for t, d in zip(calls["ticker"], calls["call_date"])]
    return pd.Series([x["analyst_score"] if x else np.nan for x in views], index=calls.index)


# ------------------------------------------------------------------ statistics

def tau_b(x, y):
    t = stats.kendalltau(x, y, variant="b").statistic
    return 0.0 if np.isnan(t) else t


def bootstrap(D, stat, n_boot, rng):
    """stat(D rows) -> vector over prompts; observed value and (n_boot x prompts) resamples."""
    obs = stat(np.arange(len(D)))
    idx = rng.integers(0, len(D), (n_boot, len(D)))
    return obs, np.array([stat(i) for i in idx])


def compare(obs, boot, prompts):
    """95% intervals, Holm-corrected pairwise and vs-neutral p-values, and the all-equal Wald test."""
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    k = len(prompts)
    pairs = list(itertools.combinations(range(k), 2))
    raw = []
    for i, j in pairs:
        # z test with the bootstrap standard error of the paired difference (unlike counting
        # resamples that cross zero, its p isn't floored at 1/n_boot, which Holm over 66 pairs needs)
        se = np.std(boot[:, i] - boot[:, j], ddof=1)
        raw.append(2 * stats.norm.sf(abs(obs[i] - obs[j]) / se) if se > 0 else 1.0)
    raw = np.array(raw)
    pair_p = np.full((k, k), np.nan)
    for (i, j), p in zip(pairs, holm(raw)):
        pair_p[i, j] = pair_p[j, i] = p
    n0 = prompts.index(NEUTRAL)
    others = [i for i in range(k) if i != n0]
    raw_n = [raw[pairs.index(tuple(sorted((i, n0))))] for i in others]
    vs_neutral = dict(zip([prompts[i] for i in others], holm(np.array(raw_n))))
    C = np.eye(k)[1:] - np.eye(k)[:-1]          # successive differences: all equal <=> C x = 0
    cov = np.cov(boot, rowvar=False)
    cx = C @ obs
    W = float(cx @ np.linalg.pinv(C @ cov @ C.T) @ cx)
    table = pd.DataFrame({"prompt": prompts, "value": obs, "lo": lo, "hi": hi,
                          "diff_vs_neutral": obs - obs[n0],
                          "p_vs_neutral_holm": [vs_neutral.get(p, np.nan) for p in prompts]})
    return table, pair_p, (W, k - 1, stats.chi2.sf(W, k - 1))


def cochran_q(X):
    """Cochran's Q for k paired binary outcomes (rows = calls)."""
    k = X.shape[1]
    col, row = X.sum(0), X.sum(1)
    q = (k - 1) * (k * np.sum(col ** 2) - col.sum() ** 2) / (k * row.sum() - np.sum(row ** 2))
    return q, k - 1, stats.chi2.sf(q, k - 1)


# ------------------------------------------------------------------ charts

def _stars(p):
    return "" if np.isnan(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def plot_dots(table, title, xlabel, path, n, ref=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(table))[::-1]
    for yy, r in zip(y, table.itertuples()):
        c = MUTED if r.prompt == NEUTRAL else SERIES[0]
        ax.plot([r.lo, r.hi], [yy, yy], color=c, lw=2.5, alpha=0.6, solid_capstyle="round")
        ax.plot(r.value, yy, "o", color=c, ms=8, markeredgecolor=SURFACE)
        ax.text(r.hi + 0.004, yy, f"{r.value:+.3f} {_stars(r.p_vs_neutral_holm)}", va="center",
                fontsize=10, color=INK_2)
    n0 = table.loc[table["prompt"] == NEUTRAL, "value"].iloc[0]
    ax.axvline(n0, color=MUTED, lw=1, ls="--")
    if ref is not None:
        ax.axvline(ref[0], color=SERIES[1], lw=1.2, ls=":")
        ax.text(ref[0], -0.55, " " + ref[1], color=SERIES[1], fontsize=10, ha="left", va="bottom")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, table["prompt"])
    ax.set_ylim(-0.7, len(table) - 0.1)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}  (same {n} calls for every prompt)", loc="left")
    fig.text(0.01, -0.02, "line = 95% bootstrap interval · dashed = neutral · stars = differs from neutral "
             "(paired bootstrap, Holm): * p<.05  ** p<.01  *** p<.001", fontsize=9, color=INK_2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_mix(tables, path, n):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    for ax, dec in zip(axes, ["buy", "hold", "sell"]):
        t = tables[dec]
        y = np.arange(len(t))[::-1]
        ax.barh(y, t["value"] * 100, color=DECISION_COLORS[dec], height=0.65, alpha=0.85)
        ax.errorbar(t["value"] * 100, y, xerr=[(t["value"] - t["lo"]) * 100, (t["hi"] - t["value"]) * 100],
                    fmt="none", color=INK, capsize=2, lw=1)
        pad = t["hi"].max() * 100 * 0.02
        for yy, r in zip(y, t.itertuples()):
            ax.text(r.hi * 100 + pad, yy, f"{r.value * 100:.1f}% {_stars(r.p_vs_neutral_holm)}",
                    va="center", fontsize=9, color=INK_2)
        n0 = t.loc[t["prompt"] == NEUTRAL, "value"].iloc[0] * 100
        ax.axvline(n0, color=MUTED, lw=1, ls="--")
        ax.set_title(dec, loc="left")
        ax.set_xlim(0, t["hi"].max() * 100 * 1.35)
        ax.set_xlabel("% of calls")
    axes[0].set_yticks(np.arange(len(t))[::-1], t["prompt"])
    fig.suptitle(f"Decision mix by prompt (same {n} calls for every prompt)", x=0.01, ha="left",
                 fontweight="bold")
    fig.text(0.01, -0.03, "bars = 95% bootstrap interval · dashed = neutral · stars = differs from neutral "
             "(paired bootstrap, Holm): * p<.05  ** p<.01  *** p<.001", fontsize=9, color=INK_2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_pairs(mats, prompts, path):
    fig, axes = plt.subplots(1, len(mats), figsize=(6.2 * len(mats), 5.8))
    axes = np.atleast_1d(axes)
    for ax, (title, diff, p) in zip(axes, mats):
        lim = np.nanmax(np.abs(diff)) or 1
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i, j in itertools.product(range(len(prompts)), repeat=2):
            if i != j and p[i, j] < 0.05:
                ax.text(j, i, _stars(p[i, j]), ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(prompts)), prompts, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(prompts)), prompts, fontsize=8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.text(0.01, -0.02, "cell = row prompt minus column prompt · stars = significant after Holm over all 66 pairs",
             fontsize=9, color=INK_2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ report

def main(paths, out, n_boot=2000, seed=0):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    wide, calls, prompts = load(paths)
    D, y = wide.to_numpy(), calls[MOVE].to_numpy()
    n = len(D)
    lines = [f"{n} calls with a decision under every prompt (of {len(prompts)} prompts)"]
    results, mats = {}, []

    def report(name, obs, boot, scale=1.0):
        table, pair_p, (W, df_, p) = compare(obs, boot, prompts)
        table.insert(1, "metric", name)
        results[name] = table
        diff = (obs[:, None] - obs[None, :]) * scale
        lines.append(f"\n{name}: all prompts equal? Wald chi2({df_}) = {W:.1f}, p = {p:.2g}")
        lines.append(table.round(4).to_string(index=False))
        sig = [(prompts[i], prompts[j]) for i, j in itertools.combinations(range(len(prompts)), 2)
               if pair_p[i, j] < 0.05]
        lines.append(f"  pairs that differ (Holm, 66 pairs): {len(sig)}" + (f": {sig}" if sig else ""))
        return diff, pair_p

    # 1. tau with SCAR
    obs, boot = bootstrap(D, lambda i: np.array([tau_b(D[i, j], y[i]) for j in range(len(prompts))]), n_boot, rng)
    d1, p1 = report("tau_scar", obs, boot)
    mats.append(("Kendall tau with SCAR: difference", d1, p1))

    # 2. decision shares
    for dec, val in [("buy", 1), ("hold", 0), ("sell", -1)]:
        X = (D == val).astype(float)
        obs, boot = bootstrap(X, lambda i, X=X: X[i].mean(0), n_boot, rng)
        dd, pp = report(dec, obs, boot, scale=100)
        if dec == "buy":
            mats.append(("Buy share: difference (points)", dd, pp))
            q, df_, p = cochran_q(X)
            lines.append(f"  Cochran's Q for the buy share: Q({df_}) = {q:.1f}, p = {p:.2g}")

    # 3. tau with the analyst consensus
    a = analyst_scores(calls).to_numpy()
    cov = ~np.isnan(a)
    Da, aa, ya = D[cov], a[cov], y[cov]
    lines.append(f"\nanalyst consensus available for {cov.sum()} of {n} calls")
    analyst_ref = None
    if cov.sum() > 30:
        obs, boot = bootstrap(Da, lambda i: np.array([tau_b(Da[i, j], aa[i]) for j in range(len(prompts))]),
                              n_boot, rng)
        d3, p3 = report("tau_analyst", obs, boot)
        mats.append(("Kendall tau with analysts: difference", d3, p3))
        t_as = tau_b(aa, ya)
        lines.append(f"  analysts' own tau with SCAR on these calls: {t_as:+.3f}")
        analyst_ref = (t_as, f"analysts {t_as:+.3f}")

    pd.concat(results.values()).to_csv(os.path.join(out, "emotion_compare.csv"), index=False)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out, "emotion_compare.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    plot_dots(results["tau_scar"], "Kendall tau of decisions with 5-day SCAR",
              "Kendall tau-b: decision (sell < hold < buy) vs 5-day SCAR", os.path.join(out, "tau_scar.png"), n,
              ref=analyst_ref)
    plot_mix({d: results[d] for d in ["buy", "hold", "sell"]}, os.path.join(out, "decision_mix.png"), n)
    if "tau_analyst" in results:
        plot_dots(results["tau_analyst"], "Kendall tau of decisions with the analyst consensus",
                  "Kendall tau-b: decision vs analyst consensus score", os.path.join(out, "tau_analyst.png"),
                  int(cov.sum()))
    plot_pairs(mats, prompts, os.path.join(out, "pairwise.png"))
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results_sum/emotions")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(a.paths, a.out, a.boot)
