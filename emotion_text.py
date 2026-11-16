"""Emotion content of the earnings calls and of the model's replies, scored with RoBERTa emotion classifiers.

    python emotion_text.py                                   # gpt-oss-120b neutral run -> results_120b/text_emotions
    python emotion_text.py --run anon_sum_openai/gpt-oss-20b_neutral.pkl --out results_run2/text_emotions
    python emotion_text.py --limit 50                        # quick test on 50 calls

For every call the neutral prompt decided, three texts are scored:
  transcript   the earnings call as the dataset has it (Seeking Alpha header stripped)
  reasoning    the model's reasoning ("analysis" channel of its reply)
  answer       the model's final answer (after "assistantfinal")
Each text is split into sentences, packed into chunks of up to CHUNK_TOKENS tokens, classified, and averaged
over chunks weighted by their token counts, so every call gets one probability per emotion per text.

Models (--models, default both):
  go      SamLowe/roberta-base-go_emotions: 28 labels (27 emotions + neutral), multi-label (sigmoid). Covers most
          of the prompt emotions directly: amusement, anger, confusion, disgust, excitement, optimism, surprise,
          plus nervousness (~anxiety), fear (~panic), sadness / grief (~depression).
  ekman   j-hartmann/emotion-english-distilroberta-base: Ekman's 6 + neutral, softmax (the one in
          notebooks/analysis.ipynb).

Writes, in --out:
  scores_<model>.parquet   one row per call x text: call_id, ticker, decision, scar5, text, n_chunks, <label>...
  summary_<model>.txt      mean of each emotion per text; Spearman correlation transcript vs reasoning / answer
                           per emotion; mean emotion by decision; Kendall tau of each transcript emotion with SCAR
  means_<model>.png        mean probability of each emotion in the transcript, reasoning and answer
  link_<model>.png         per emotion: correlation of the transcript's level with the reply's
  by_decision_<model>.png  transcript emotion by the decision the model made
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from visualize import plt, SURFACE, INK_2, MUTED, AXIS, SERIES, DECISION_COLORS, DECISIONS

MODELS = {
    "go": "SamLowe/roberta-base-go_emotions",
    "ekman": "j-hartmann/emotion-english-distilroberta-base",
}
CHUNK_TOKENS = 128   # go_emotions was trained on short Reddit comments; short chunks keep inputs in-distribution
BATCH = 256
TEXTS = ["transcript", "reasoning", "answer"]
TEXT_COLORS = dict(zip(TEXTS, [INK_2, SERIES[0], SERIES[1]]))
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


# ------------------------------------------------------------------ texts

def split_reply(reply):
    """gpt-oss replies are 'analysis<reasoning>assistantfinal<answer>'; Qwen-style use <think>...</think>."""
    reply = reply or ""
    if "assistantfinal" in reply:
        reasoning, answer = reply.rsplit("assistantfinal", 1)
    elif "</think>" in reply:
        reasoning, answer = reply.rsplit("</think>", 1)
    else:
        reasoning, answer = "", reply
    reasoning = re.sub(r"^\s*(analysis|<think>)", "", reasoning)
    return reasoning.strip(), answer.strip()


def load_texts(run, limit=None):
    from llms import call_text
    res = pd.read_pickle(run)
    res = res.drop_duplicates("call_id")
    if limit:
        res = res.head(limit)
    calls = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Earnings_call_transcripts",
                                         "sp1500_earnings_calls.parquet"))
    calls = calls[calls["call_id"].isin(res["call_id"])].set_index("call_id", drop=False)
    rows = []
    for r in res.itertuples():
        if r.call_id not in calls.index:
            continue
        reasoning, answer = split_reply(r.summary)
        transcript = call_text(calls.loc[r.call_id])
        for name, text in zip(TEXTS, [transcript, reasoning, answer]):
            rows.append(dict(call_id=r.call_id, ticker=r.ticker, decision=r.decision, scar5=r.scar5,
                             text_type=name, text=text))
    df = pd.DataFrame(rows)
    print(f"{df['call_id'].nunique()} calls, {len(df)} texts")
    return df


def chunks(text, tok, max_tokens=CHUNK_TOKENS):
    """Sentences packed greedily into chunks of at most max_tokens tokens (a longer sentence is cut)."""
    sents = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    out, cur, n = [], [], 0
    for s in sents:
        k = len(tok.tokenize(s))
        if cur and n + k > max_tokens:
            out.append((" ".join(cur), n))
            cur, n = [], 0
        cur.append(s)
        n += k
    if cur:
        out.append((" ".join(cur), n))
    return [(c, min(n, max_tokens)) for c, n in out]


# ------------------------------------------------------------------ scoring

def score(df, model_name, batch=BATCH):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    multi = model.config.problem_type == "multi_label_classification" or "go_emotions" in model_name

    pieces = []  # (row index, chunk text, weight)
    for i, t in enumerate(df["text"]):
        for c, w in chunks(t, tok):
            pieces.append((i, c, w))
    print(f"{model_name}: {len(pieces)} chunks on {dev}", flush=True)
    probs = np.zeros((len(pieces), len(labels)), dtype=np.float32)
    order = np.argsort([len(p[1]) for p in pieces])  # similar lengths per batch: less padding
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([pieces[j][1] for j in idx], padding=True, truncation=True, max_length=CHUNK_TOKENS + 16,
                      return_tensors="pt").to(dev)
            logits = model(**enc).logits.float()
            p = torch.sigmoid(logits) if multi else torch.softmax(logits, -1)
            probs[idx] = p.cpu().numpy()
            if (s // batch) % 200 == 0:
                print(f"  {s}/{len(order)}", flush=True)

    rows = np.array([p[0] for p in pieces])
    w = np.array([p[2] for p in pieces], dtype=float)
    out = np.zeros((len(df), len(labels)))
    tot = np.zeros(len(df))
    np.add.at(out, rows, probs * w[:, None])
    np.add.at(tot, rows, w)
    out = out / np.where(tot > 0, tot, np.nan)[:, None]
    res = df.drop(columns="text").reset_index(drop=True)
    res["n_chunks"] = np.bincount(rows, minlength=len(df))
    return pd.concat([res, pd.DataFrame(out, columns=labels)], axis=1), labels


# ------------------------------------------------------------------ analysis

def summarize(scores, labels, title):
    lines = [f"##### {title}", f"{scores['call_id'].nunique()} calls; probabilities averaged over chunks "
             f"(weighted by tokens), then over calls"]
    wide = scores.pivot(index="call_id", columns="text_type", values=labels)
    means = scores.groupby("text_type")[labels].mean().T[TEXTS]
    means["reply minus call"] = means["answer"] - means["transcript"]
    lines.append("\n=== mean probability of each emotion, by text ===")
    lines.append(means.sort_values("transcript", ascending=False).round(4).to_string())

    lines.append("\n=== does the reply carry the call's emotion? Spearman rho across calls ===")
    link = []
    for lab in labels:
        t = wide[(lab, "transcript")]
        row = {"emotion": lab}
        for other in ("reasoning", "answer"):
            rho, p = stats.spearmanr(t, wide[(lab, other)], nan_policy="omit")
            row[f"rho call~{other}"], row[f"p {other}"] = rho, p
        link.append(row)
    link = pd.DataFrame(link).set_index("emotion")
    lines.append(link.sort_values("rho call~answer", ascending=False).round(4).to_string())

    calls = scores[scores["text_type"] == "transcript"].set_index("call_id")
    lines.append("\n=== transcript emotion by the decision the model made (mean probability) ===")
    by = calls.groupby("decision")[labels].mean().T.reindex(columns=[d for d in DECISIONS if d in
                                                                      calls["decision"].unique()])
    kw = {lab: stats.kruskal(*[g[lab].dropna() for _, g in calls.groupby("decision")]).pvalue for lab in labels}
    by["Kruskal p"] = pd.Series(kw)
    lines.append(by.round(4).to_string())

    lines.append("\n=== Kendall tau-b of each emotion with SCAR (transcript level) ===")
    tau = {lab: stats.kendalltau(calls[lab], calls["scar5"], nan_policy="omit") for lab in labels}
    tau = pd.DataFrame({"tau": {k: v.statistic for k, v in tau.items()},
                        "p": {k: v.pvalue for k, v in tau.items()}})
    lines.append(tau.sort_values("tau", ascending=False).round(4).to_string())
    return "\n".join(lines), means, link, by


def plot_means(means, labels, title, path):
    order = means.sort_values("transcript").index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(order) + 1.6))
    for i, t in enumerate(TEXTS):
        ax.barh(y + (i - 1) * 0.27, means.loc[order, t], height=0.26, color=TEXT_COLORS[t], label=t, alpha=0.9)
    ax.set_yticks(y, order, fontsize=9)
    ax.set_xlabel("mean probability")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_link(link, title, path):
    d = link.sort_values("rho call~answer")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(d) + 1.6))
    ax.scatter(d["rho call~reasoning"], y, color=TEXT_COLORS["reasoning"], s=36, label="reasoning", zorder=3)
    ax.scatter(d["rho call~answer"], y, color=TEXT_COLORS["answer"], s=36, label="answer", zorder=3)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, d.index, fontsize=9)
    ax.set_xlabel("Spearman rho: the call's level of the emotion vs the reply's, across calls")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_by_decision(by, title, path):
    d = by.drop(columns="Kruskal p")
    rel = d.div(d.mean(axis=1), axis=0) - 1      # each decision's level relative to the average over decisions
    order = rel.index[np.argsort((rel.get("buy", 0) - rel.get("sell", 0)).to_numpy())]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(order) + 1.6))
    for i, dec in enumerate(d.columns):
        ax.barh(y + (i - 1) * 0.27, rel.loc[order, dec] * 100, height=0.26, color=DECISION_COLORS[dec], label=dec)
    for yy, lab in zip(y, order):
        if by.loc[lab, "Kruskal p"] < 0.001:
            ax.text(ax.get_xlim()[1], yy, " ***", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, order, fontsize=9)
    ax.set_xlabel("emotion in the call, % above or below its average over the three decisions")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.01, -0.01, "*** decisions differ, Kruskal-Wallis p < .001", fontsize=8.5, color=MUTED, va="top")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="anon_sum_openai/gpt-oss-120b_neutral.pkl",
                    help="the neutral run's pickle (one row per call with the model's reply in 'summary')")
    ap.add_argument("--out", default="results_120b/text_emotions")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--limit", type=int, help="score only the first N calls (testing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    texts = load_texts(args.run, args.limit)
    run_name = os.path.splitext(os.path.basename(args.run))[0]
    for key in args.models:
        path = os.path.join(args.out, f"scores_{key}.parquet")
        scores, labels = score(texts, MODELS[key])
        scores.to_parquet(path, index=False)
        title = f"{MODELS[key]} on {run_name}"
        text, means, link, by = summarize(scores, labels, title)
        with open(os.path.join(args.out, f"summary_{key}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        plot_means(means, labels, f"Emotion in the call vs the reply ({key})",
                   os.path.join(args.out, f"means_{key}.png"))
        plot_link(link, f"Does the reply carry the call's emotion? ({key})",
                  os.path.join(args.out, f"link_{key}.png"))
        plot_by_decision(by, f"Emotion in the call, by the model's decision ({key})",
                         os.path.join(args.out, f"by_decision_{key}.png"))
    print(f"-> {args.out}")
