"""Fine-tune RoBERTa on StockEmotions (Lee et al. 2023, arXiv:2301.09279) to classify text into the
12 investor emotions, which are the 11 prompt emotions of this project plus "ambiguous".

    git clone --depth 1 https://github.com/adlnlp/StockEmotions.git data_collection/StockEmotions
    python data_collection/train_stockemotions.py                  # -> models/stockemotions-roberta

Data: 10,000 StockTwits posts from 2020 with one emotion label each, split 8,000 / 1,000 / 1,000 by the
authors (tweet/{train,val,test}_stockemo.csv). The authors released no trained weights, so this trains one.

Text: the "processed" column (tickers spelled out as company names) with the emoji tags it adds
("[thinking face]") removed, since earnings calls and the LLM's replies have no emojis: the classifier
has to learn from the words. --keep-emoji keeps them (closer to the paper's setup).

Training: roberta-base, max 96 tokens, AdamW with linear warmup / decay, mixed precision. One run per
learning rate in --lrs; each keeps the epoch with the best validation macro F1, and the run with the best
validation macro F1 is saved. Test results (per-emotion precision / recall / F1, macro F1) are reported
next to the paper's best baselines (macro F1 0.39 RoBERTa, 0.42 DistilBERT, its Table 2).
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "StockEmotions", "tweet")
OUT = os.path.join(os.path.dirname(HERE), "models", "stockemotions-roberta")
LABELS = ["ambiguous", "amusement", "anger", "anxiety", "belief", "confusion", "depression", "disgust",
          "excitement", "optimism", "panic", "surprise"]
PAPER_F1 = {"RoBERTa (paper)": [0.09, 0.25, 0.13, 0.44, 0.29, 0.50, 0.21, 0.43, 0.44, 0.39, 0.11, 0.21],
            "DistilBERT (paper)": [0.12, 0.37, 0.56, 0.42, 0.42, 0.51, 0.29, 0.43, 0.51, 0.42, 0.48, 0.21]}
_EMOJI = re.compile(r"\[[a-z0-9 _:'&-]+\]", re.I)


def load(split, keep_emoji=False):
    d = pd.read_csv(os.path.join(DATA, f"{split}_stockemo.csv"))
    text = d["processed"].fillna(d["original"]).astype(str)
    if not keep_emoji:
        text = text.str.replace(_EMOJI, " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    y = d["emo_label"].map({l: i for i, l in enumerate(LABELS)})
    assert y.notna().all(), set(d["emo_label"]) - set(LABELS)
    return text.tolist(), y.to_numpy()


def batches(tok, texts, y, bs, shuffle, rng, max_len):
    idx = rng.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        enc = tok([texts[i] for i in j], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield enc, torch.tensor(y[j])


@torch.no_grad()
def predict(model, tok, texts, dev, max_len, bs=128):
    model.eval()
    out = []
    for enc, _ in batches(tok, texts, np.zeros(len(texts), dtype=int), bs, False, None, max_len):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(**enc.to(dev)).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def train_one(base, lr, data, args, dev):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    ).to(dev)
    (xtr, ytr), (xva, yva) = data["train"], data["val"]
    steps = args.epochs * int(np.ceil(len(xtr) / args.bs))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best = (-1, None, None)
    for ep in range(1, args.epochs + 1):
        model.train()
        for enc, yb in batches(tok, xtr, ytr, args.bs, True, rng, args.max_len):
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(**enc.to(dev), labels=yb.to(dev)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        f1 = f1_score(yva, predict(model, tok, xva, dev, args.max_len).argmax(1), average="macro")
        print(f"  lr {lr:g} epoch {ep}: val macro F1 {f1:.3f}", flush=True)
        if f1 > best[0]:
            best = (f1, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return best[0], best[1], model, tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-5, 3e-5])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-emoji", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: load(s, args.keep_emoji) for s in ("train", "val", "test")}
    print({s: len(v[0]) for s, v in data.items()}, "on", dev)

    runs = []
    for lr in args.lrs:
        f1, ep, model, tok = train_one(args.base, lr, data, args, dev)
        runs.append(dict(lr=lr, epoch=ep, val_macro_f1=f1))
        if f1 >= max(r["val_macro_f1"] for r in runs):
            best_model, best_tok, best_run = model, tok, runs[-1]
        else:
            del model
        torch.cuda.empty_cache()
    print("best:", best_run)

    xte, yte = data["test"]
    pred = predict(best_model, best_tok, xte, dev, args.max_len).argmax(1)
    rep = classification_report(yte, pred, labels=range(len(LABELS)), target_names=LABELS, digits=3,
                                output_dict=True, zero_division=0)
    ours = [rep[l]["f1-score"] for l in LABELS]
    tab = pd.DataFrame({"precision": [rep[l]["precision"] for l in LABELS],
                        "recall": [rep[l]["recall"] for l in LABELS], "F1 (ours)": ours,
                        **PAPER_F1, "test n": [int(rep[l]["support"]) for l in LABELS]}, index=LABELS)
    tab.loc["macro avg"] = [rep["macro avg"]["precision"], rep["macro avg"]["recall"],
                            rep["macro avg"]["f1-score"], *[np.mean(v) for v in PAPER_F1.values()], len(yte)]
    acc = float((pred == yte).mean())
    text = (f"StockEmotions test set ({len(yte)} posts), {args.base}, lr {best_run['lr']:g}, epoch "
            f"{best_run['epoch']}, emoji tags {'kept' if args.keep_emoji else 'removed'}\n"
            f"accuracy {acc:.3f}, macro F1 {rep['macro avg']['f1-score']:.3f}\n\n" + tab.round(3).to_string()
            + "\n\nvalidation runs: " + json.dumps(runs))
    print(text)

    os.makedirs(args.out, exist_ok=True)
    best_model.save_pretrained(args.out)
    best_tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("->", args.out)
