"""Robustness checks for a run on the call summaries (main.py --summaries), next to the
movement metrics of movement_metrics.py:

    python summary_run_checks.py "anon_sum_openai/gpt-oss-20b_*.pkl" --out results_sum/checks

  baselines   tau-b vs scar5 of the EPS and revenue beat / miss (Seeking Alpha header, which
              the model never sees) and of LIWC tone, on the same calls
  incremental OLS scar5 ~ model decision + EPS beat + revenue beat (HC1 errors): does the
              model know more than whether the quarter beat?
  within      the model's tau among beats, in-lines and misses separately
  cut-off     replies that hit the token limit have no decision; are those calls different?
              tau with them counted as hold
  ensemble    tau of the mean decision over all prompts
  agreement   share of calls where each prompt's decision equals the neutral one
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

import visualize as v
from visualize import plt, SURFACE, INK_2, MUTED, AXIS, SERIES
from movement_metrics import tau_b, tau_test, POSITION, MOVE

_EPS = re.compile(r"EPS of .*?(beats|misses|in-line)", re.I)
_REV = re.compile(r"Revenue of .*?(beats|misses|in-line)", re.I)
_SIGN = {"beats": 1, "misses": -1, "in-line": 0}


def surprises(call_ids):
    """EPS and revenue beat (+1) / in line (0) / miss (-1) from each transcript's header."""
    from llms import get_text
    rows = []
    for cid in call_ids:
        head = "\n".join(get_text(cid).split("\n")[:12])
        e, r = _EPS.search(head), _REV.search(head)
        rows.append((cid, _SIGN[e.group(1).lower()] if e else np.nan, _SIGN[r.group(1).lower()] if r else np.nan))
    return pd.DataFrame(rows, columns=["call_id", "eps_beat", "rev_beat"])


def ols_hc1(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ (X.T * e ** 2) @ X @ bread * n / (n - k)
    se = np.sqrt(np.diag(cov))
    return beta[1:], se[1:], 2 * stats.t.sf(np.abs(beta[1:] / se[1:]), n - k)


def main(paths, out):
    os.makedirs(out, exist_ok=True)
    from llms import load_catalogue
    res = v.load_results(paths, with_returns=False)
    cat = load_catalogue()
    info = cat[["call_id", "ticker", "call_date", "usable", MOVE, "Posemo", "Negemo"]].drop_duplicates(["ticker", "call_date"])
    df = res.drop(columns=[c for c in ("call_id", "car5", "scar5") if c in res.columns]).merge(
        info, on=["ticker", "call_date"], how="left")
    df = df[df["usable"] == True].copy()  # noqa: E712
    df["d"] = df["decision"].map(POSITION)
    calls = df.drop_duplicates("call_id")[["call_id", MOVE, "Posemo", "Negemo"]]
    calls = calls.merge(surprises(calls["call_id"]), on="call_id")
    calls["tone"] = calls["Posemo"] - calls["Negemo"]
    df = df.merge(calls[["call_id", "eps_beat", "rev_beat", "tone"]], on="call_id")
    prompts = v._ordered(df["emotion"])
    lines = []

    # baselines on the same calls
    base = []
    for name, col in [("EPS beat / miss", "eps_beat"), ("Revenue beat / miss", "rev_beat"), ("LIWC tone", "tone")]:
        c = calls.dropna(subset=[col])
        t, p, lo, hi = tau_test(c[col], c[MOVE], n_perm=2000)
        base.append(dict(signal=name, calls=len(c), tau=t, lo=lo, hi=hi, p=p))
    base = pd.DataFrame(base)
    lines.append("Baselines (tau-b vs scar5, same calls):\n" + base.round(3).to_string(index=False))
    lines.append(f"EPS header found for {calls['eps_beat'].notna().mean():.0%} of calls; "
                 f"beat {(calls['eps_beat'] == 1).mean():.0%}, in line {(calls['eps_beat'] == 0).mean():.0%}, "
                 f"miss {(calls['eps_beat'] == -1).mean():.0%}")

    rows = []
    mean_d = df.groupby("call_id")["d"].mean()
    for e in prompts:
        s = df[df["emotion"] == e]
        cut = s["d"].isna()
        r = dict(prompt=e, n=len(s), no_decision=cut.mean(),
                 scar_no_decision=s.loc[cut, MOVE].mean(), scar_decided=s.loc[~cut, MOVE].mean(),
                 tau=tau_b(s.loc[~cut, "d"], s.loc[~cut, MOVE]),
                 tau_cut_as_hold=tau_b(s["d"].fillna(0), s[MOVE]))
        k = s.dropna(subset=["d", "eps_beat", "rev_beat"])
        r["tau_with_eps"] = tau_b(k["d"], k["eps_beat"])
        b, se, p = ols_hc1(k[MOVE].to_numpy(), k[["d", "eps_beat", "rev_beat"]].to_numpy(float))
        r.update(b_model=b[0], se_model=se[0], p_model=p[0], b_eps=b[1], b_rev=b[2])
        for name, val in [("beat", 1), ("inline", 0), ("miss", -1)]:
            g = k[k["eps_beat"] == val]
            r[f"tau_{name}"] = tau_b(g["d"], g[MOVE]) if len(g) > 30 else np.nan
        if e != "neutral":
            both = s[["call_id", "d"]].merge(df[df["emotion"] == "neutral"][["call_id", "d"]], on="call_id")
            r["agree_neutral"] = (both["d_x"] == both["d_y"]).mean()
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "checks_by_prompt.csv"), index=False)
    lines.append("\nPer prompt:\n" + table.round(3).to_string(index=False))

    ens = calls.set_index("call_id").join(mean_d.rename("mean_d")).dropna(subset=["mean_d"])
    t, p, lo, hi = tau_test(ens["mean_d"], ens[MOVE], n_perm=2000)
    lines.append(f"\nEnsemble (mean decision over {len(prompts)} prompts): tau {t:.3f} [{lo:.3f}, {hi:.3f}], "
                 f"p {p:.4f}, {len(ens)} calls")
    k = ens.dropna(subset=["eps_beat", "rev_beat"])
    b, se, p = ols_hc1(k[MOVE].to_numpy(), k[["mean_d", "eps_beat", "rev_beat"]].to_numpy(float))
    lines.append(f"  with EPS + revenue beat: model {b[0]:+.3f} (se {se[0]:.3f}, p {p[0]:.4f}), "
                 f"EPS {b[1]:+.3f} (se {se[1]:.3f}), revenue {b[2]:+.3f} (se {se[2]:.3f})")
    b2, se2, p2 = ols_hc1(k[MOVE].to_numpy(), k[["eps_beat", "rev_beat"]].to_numpy(float))
    r2 = lambda X, bb: 1 - np.var(k[MOVE] - X @ bb - (k[MOVE] - X @ bb).mean()) / np.var(k[MOVE])  # noqa: E731
    lines.append(f"  R^2: beats only {r2(k[['eps_beat', 'rev_beat']].to_numpy(float), b2):.4f}, "
                 f"beats + model {r2(k[['mean_d', 'eps_beat', 'rev_beat']].to_numpy(float), b):.4f}")

    sub = ens.dropna(subset=["eps_beat"])
    neu = df[df["emotion"] == "neutral"].dropna(subset=["d", "eps_beat"])
    lines.append(f"\nOn the {len(sub)} calls with an EPS header: ensemble tau {tau_b(sub['mean_d'], sub[MOVE]):.3f}, "
                 f"neutral tau {tau_b(neu['d'], neu[MOVE]):.3f} (EPS beat {base.loc[0, 'tau']:.3f})")
    b, se, p = ols_hc1(ens[MOVE].to_numpy(), ens[["mean_d", "tone"]].to_numpy(float))
    lines.append(f"All calls, scar5 ~ ensemble + LIWC tone: model {b[0]:+.3f} (se {se[0]:.3f}, p {p[0]:.2g}), "
                 f"tone {b[1]:+.4f} (se {se[1]:.4f}, p {p[1]:.2g})")
    mix = df.dropna(subset=["d"]).groupby("emotion")["decision"].value_counts(normalize=True).unstack()
    lines.append("\nDecision mix (share of decided calls):\n" + mix.reindex(prompts).round(3).to_string())

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out, "checks.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    plot(table, base, t, lo, hi, out)
    return table, base


def plot(table, base, ens_tau, ens_lo, ens_hi, out):
    # model tau per prompt vs baselines
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    names = list(table["prompt"]) + ["ensemble"] + list(base["signal"])
    vals = list(table["tau"]) + [ens_tau] + list(base["tau"])
    colors = [SERIES[0]] * len(table) + [SERIES[1]] + [MUTED] * len(base)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.65)
    ax.errorbar([ens_tau], [y[len(table)]], xerr=[[ens_tau - ens_lo], [ens_hi - ens_tau]], color=INK_2, capsize=3, lw=1)
    for yy, b in zip(y[len(table) + 1:], base.itertuples()):
        ax.errorbar([b.tau], [yy], xerr=[[b.tau - b.lo], [b.hi - b.tau]], color=INK_2, capsize=3, lw=1)
    ax.set_yticks(y, names)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Kendall tau-b with 5-day SCAR")
    ax.set_title("Model decisions vs simple signals, same calls", loc="left")
    fig.savefig(os.path.join(out, "tau_vs_baselines.png"), dpi=160, bbox_inches="tight", facecolor=SURFACE)

    # within beat / in line / miss
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(table))
    for i, (col, lab) in enumerate([("tau_beat", "EPS beat"), ("tau_miss", "EPS miss")]):
        ax.bar(x + (i - 0.5) * 0.38, table[col], width=0.38, color=SERIES[i * 2], label=lab)
    ax.set_xticks(x, table["prompt"], rotation=40, ha="right")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("Kendall tau-b with 5-day SCAR")
    ax.set_title("Model's tau within each EPS outcome", loc="left")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out, "tau_within_eps.png"), dpi=160, bbox_inches="tight", facecolor=SURFACE)

    # cut-off share and its SCAR
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x, table["no_decision"] * 100, color=SERIES[3])
    ax.set_xticks(x, table["prompt"], rotation=40, ha="right")
    ax.set_ylabel("% of calls with no decision (reply cut off)")
    ax.set_title("Replies cut off at the 2,048-token limit", loc="left")
    fig.savefig(os.path.join(out, "cut_off.png"), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    # this run vs the earlier full-transcript run (results3/movement), if it is there
    prev = "results3/movement/movement_metrics.csv"
    if os.path.exists(prev):
        old = pd.read_csv(prev).set_index("strategy")
        new = pd.read_csv(os.path.join(os.path.dirname(out), "movement", "movement_metrics.csv")).set_index("strategy")
        names = [p for p in new.index if p in old.index]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        y = np.arange(len(names))[::-1]
        for k, (t, lab, c) in enumerate([(old, f"earlier: gpt-oss-120b, full text, {int(old['calls'].median())} calls", MUTED),
                                         (new, f"now: gpt-oss-20b, summaries, ~{int(new['calls'].median())} calls", SERIES[0])]):
            t = t.loc[names]
            yy = y + (0.18 if k == 0 else -0.18)
            ax.errorbar(t["tau"], yy, xerr=[t["tau"] - t["lo"], t["hi"] - t["tau"]], fmt="o", color=c,
                        capsize=2, lw=1.2, label=lab)
        ax.set_yticks(y, names)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlabel("Kendall tau-b with 5-day SCAR (95% interval)")
        ax.set_title("Same metric, two runs", loc="left")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.savefig(os.path.join(out, "compare_runs.png"), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close("all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results_sum/checks")
    a = ap.parse_args()
    main(a.paths, a.out)
