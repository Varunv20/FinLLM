"""Emotion content of the earnings calls and of the model's replies, judged sentence by sentence by an LLM
(default Qwen3-30B-A3B-Instruct-2507) for the 11 prompt emotions.

    python emotion_llm.py --validate-only                 # StockEmotions test set only (~1 min after loading)
    python emotion_llm.py                                 # validation, then the gpt-oss-120b neutral run
    python emotion_llm.py --limit 20                      # quick test on 20 calls

How: every text (transcript, reasoning, answer; see emotion_text.py) is split into sentences. For each sentence
the LLM gets the emotion definitions and instructions (SYSTEM), the previous sentence as context, and the target
sentence, and answers yes / no for each of the 11 emotions on its own line. At each answer the probability of
"yes" vs "no" is read from the model's token probabilities, so every sentence gets 11 probabilities in 0-1
rather than hard 0/1 answers. A text's score for an emotion is the mean over its sentences (roughly: the share of
its sentences that express that emotion).

Validation: the same prompt on the 1,000 StockEmotions test tweets (one gold label each, 12 classes = the 11 +
ambiguous). A tweet is predicted as its most likely emotion, or "ambiguous" if no emotion reaches 0.5; macro F1
over the 12 is directly comparable to the fine-tuned RoBERTa (0.257) and the paper (0.29 RoBERTa, 0.40 DistilBERT,
mean of its per-class F1s). Also per-emotion ROC AUC: how well the yes-probability ranks tweets with that label
above those without.

Writes, in --out (default results_120b/text_emotions):
  llm_validation.txt                 the validation above
  llm_parts/<text>_<block>.parquet   per-sentence probabilities, written as it goes (resubmit to continue)
  scores_llm.parquet, summary_llm.txt, means_llm.png, link_llm.png, by_decision_llm.png
                                     same layout as emotion_text.py's go / ekman outputs
"""
import argparse
import glob
import math
import os
import re
import time

import numpy as np
import pandas as pd

EMOTIONS = ["amusement", "anger", "anxiety", "belief", "confusion", "depression", "disgust", "excitement",
            "optimism", "panic", "surprise"]

SYSTEM = """You label the emotions expressed in single sentences from company earnings calls, from analysts' notes about them, and from investors' posts.

The emotions:
- amusement: humor, joking, playfulness, finding something funny.
- anger: irritation, frustration, outrage, hostility, blaming someone.
- anxiety: worry, nervousness, unease, concern about risks or an uncertain future.
- belief: conviction and confidence; firmly asserting that something is true or will happen; trust in a view, plan or company.
- confusion: doubt, puzzlement, skepticism; not understanding something or questioning whether it makes sense.
- depression: sadness, discouragement, disappointment, hopelessness, resignation about bad outcomes.
- disgust: strong disapproval, contempt or revulsion at something seen as wrong, foul or distasteful.
- excitement: enthusiasm, thrill, eagerness, high energy.
- optimism: hope and positive expectations about the future.
- panic: alarm, sudden fear, a sense of crisis, urgency to act immediately.
- surprise: astonishment at something unexpected.

Rules:
- Judge only the TARGET sentence. The previous sentence is context and must not be labeled.
- Judge the emotion the speaker or writer expresses through their words, not how a reader might feel about the facts. Good or bad numbers stated plainly express no emotion.
- Most sentences in earnings calls are factual or procedural and express no emotion. Then answer "no" for every emotion.
- A sentence can express more than one emotion.

Answer with exactly these 11 lines, in this order, and nothing else:
""" + "\n".join(f"{e}: yes or no" for e in EMOTIONS)

STOCKEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collection", "StockEmotions", "tweet")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_WORDS = 3        # shorter "sentences" ("Thank you.", "Operator?") are skipped
MAX_CHARS = 1000     # transcripts have some run-on "sentences" (lists, tables); longer ones are cut
BLOCK = 200          # calls per saved part


def sentences(text):
    out = []
    for s in _SENT.split(text or ""):
        s = " ".join(s.split())
        if len(s.split()) >= MIN_WORDS:
            out.append(s[:MAX_CHARS])
    return out


class Judge:
    def __init__(self, model, max_num_seqs=512):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # see agent.py: no nvcc on the GPU nodes
        from vllm import LLM, SamplingParams
        self.model = model
        self.llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.92, enable_prefix_caching=True,
                       max_num_seqs=max_num_seqs)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=60, logprobs=5)
        self.yes = self._ids(["yes", "Yes", "YES"])
        self.no = self._ids(["no", "No", "NO"])
        self._dec = {}

    def _ids(self, words):
        ids = set()
        for w in words:
            for v in (w, " " + w):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
        return ids

    def _prompt(self, target, prev=None):
        user = f"Previous sentence (context only): {prev or '(none)'}\nTARGET sentence: {target}"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    def _decode(self, tid):
        if tid not in self._dec:
            self._dec[tid] = self.tok.decode([tid])
        return self._dec[tid]

    def _parse(self, out):
        """P(yes) for each emotion, read at the token right after '<emotion>:' in the reply."""
        res = np.full(len(EMOTIONS), np.nan, dtype=np.float32)
        text, k = "", 0
        for tid, lp in zip(out.token_ids, out.logprobs or []):
            if k < len(EMOTIONS) and re.search(rf"{EMOTIONS[k]}\s*:\s*$", text.lower()):
                py = sum(math.exp(v.logprob) for t, v in lp.items() if t in self.yes)
                pn = sum(math.exp(v.logprob) for t, v in lp.items() if t in self.no)
                if py + pn > 0.5:          # otherwise this token is e.g. a space; read the next one
                    res[k] = py / (py + pn)
                    k += 1
            text += self._decode(tid)
        return res

    def score(self, targets, prevs=None):
        prevs = prevs if prevs is not None else [None] * len(targets)
        prompts = [{"prompt_token_ids": self._prompt(t, p)} for t, p in zip(targets, prevs)]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        return np.vstack([self._parse(o.outputs[0]) for o in outs])


# ------------------------------------------------------------------ validation

def validate(judge, path):
    from sklearn.metrics import classification_report, roc_auc_score
    d = pd.read_csv(os.path.join(STOCKEMO, "test_stockemo.csv"))
    # the original post: the model can read emojis, and the 'processed' column's ticker -> name swap isn't needed
    p = judge.score(d["original"].astype(str).str.strip().tolist())
    labels = ["ambiguous"] + EMOTIONS
    filled = np.nan_to_num(p, nan=0.0)
    pred = np.where(filled.max(1) >= 0.5, np.array(EMOTIONS)[filled.argmax(1)], "ambiguous")
    rep = classification_report(d["emo_label"], pred, labels=labels, digits=3, zero_division=0, output_dict=True)
    auc = {e: roc_auc_score(d["emo_label"] == e, filled[:, i]) for i, e in enumerate(EMOTIONS)}
    tab = pd.DataFrame({"precision": [rep[l]["precision"] for l in labels],
                        "recall": [rep[l]["recall"] for l in labels],
                        "F1": [rep[l]["f1-score"] for l in labels],
                        "ROC AUC": [auc.get(l, np.nan) for l in labels],
                        "mean P(yes)": [np.nan] + list(np.nanmean(p, 0)),
                        "test n": [int(rep[l]["support"]) for l in labels]}, index=labels)
    text = (f"StockEmotions test set, {len(d)} posts, {judge.model}\n"
            f"prediction: most likely emotion, or 'ambiguous' if none has P(yes) >= 0.5\n"
            f"accuracy {(pred == d['emo_label']).mean():.3f}, macro F1 (12 classes) {rep['macro avg']['f1-score']:.3f}"
            f"   [fine-tuned RoBERTa 0.257; paper RoBERTa 0.29, DistilBERT 0.40]\n"
            f"answers that couldn't be read: {np.isnan(p).mean():.2%}\n\n" + tab.round(3).to_string())
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, flush=True)


# ------------------------------------------------------------------ calls

def score_texts(judge, texts, parts_dir):
    """Per-sentence scores for every text, saved a block of calls at a time; skips blocks already saved."""
    os.makedirs(parts_dir, exist_ok=True)
    call_ids = texts["call_id"].drop_duplicates().tolist()
    # replies first: they are short, so their results are there early
    for text_type in ["answer", "reasoning", "transcript"]:
        sub = texts[texts["text_type"] == text_type]
        for b in range(0, len(call_ids), BLOCK):
            path = os.path.join(parts_dir, f"{text_type}_{b // BLOCK:03d}.parquet")
            if os.path.exists(path):
                continue
            rows = sub[sub["call_id"].isin(call_ids[b:b + BLOCK])]
            ids, targets, prevs, pos = [], [], [], []
            for r in rows.itertuples():
                s = sentences(r.text)
                ids += [r.call_id] * len(s)
                targets += s
                prevs += [None] + s[:-1]
                pos += list(range(len(s)))
            t0 = time.time()
            p = judge.score(targets, prevs) if targets else np.zeros((0, len(EMOTIONS)))
            part = pd.DataFrame(p, columns=EMOTIONS)
            part.insert(0, "sentence", targets)
            part.insert(0, "position", pos)
            part.insert(0, "text_type", text_type)
            part.insert(0, "call_id", ids)
            part.to_parquet(path, index=False)
            print(f"{text_type} block {b // BLOCK}: {len(targets)} sentences in {time.time() - t0:.0f}s, "
                  f"unread {np.isnan(p).mean():.2%}", flush=True)


def aggregate(texts, parts_dir):
    sent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))])
    g = sent.groupby(["call_id", "text_type"])
    means = g[EMOTIONS].mean()
    means["n_chunks"] = g.size()        # number of sentences (column name shared with emotion_text.py)
    meta = texts.drop(columns="text").set_index(["call_id", "text_type"])
    return meta.join(means, how="inner").reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--run", default="anon_sum_openai/gpt-oss-120b_neutral.pkl")
    ap.add_argument("--out", default="results_120b/text_emotions")
    ap.add_argument("--limit", type=int, help="score only the first N calls (testing)")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    judge = Judge(args.model)

    val_path = os.path.join(args.out, "llm_validation.txt")
    if args.validate_only or not os.path.exists(val_path):
        validate(judge, val_path)
    if args.validate_only:
        raise SystemExit

    from emotion_text import load_texts, summarize, plot_means, plot_link, plot_by_decision
    texts = load_texts(args.run, args.limit)
    parts = os.path.join(args.out, "llm_parts" + (f"_limit{args.limit}" if args.limit else ""))
    score_texts(judge, texts, parts)
    scores = aggregate(texts, parts)
    scores.to_parquet(os.path.join(args.out, "scores_llm.parquet"), index=False)
    title = f"{args.model} (sentence yes/no) on {os.path.splitext(os.path.basename(args.run))[0]}"
    text, means, link, by = summarize(scores, EMOTIONS, title)
    with open(os.path.join(args.out, "summary_llm.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    plot_means(means, EMOTIONS, "Emotion in the call vs the reply (LLM judge)", os.path.join(args.out, "means_llm.png"))
    plot_link(link, "Does the reply carry the call's emotion? (LLM judge)", os.path.join(args.out, "link_llm.png"))
    plot_by_decision(by, "Emotion in the call, by the model's decision (LLM judge)",
                     os.path.join(args.out, "by_decision_llm.png"))
    print(f"-> {args.out}")
