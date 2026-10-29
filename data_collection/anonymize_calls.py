"""Remove company names, people, tickers and dates from every transcript.

Writes a copy of the transcripts that llms.get_text(call_id, anonymized=True) reads:
    Earnings_call_transcripts/sp1500_earnings_calls_anon.parquet
    columns: call_id, text, n_replaced (substitutions made), leaks (mentions of the
    company, ticker or a participant's surname still left; 0 is the goal)

What is removed, per call:
  header        the Seeking Alpha header before the participant list (title, publish
                date, ticker, and the EPS / revenue beat-or-miss summary)
  people        everyone in the Executives / Analysts lists -> "Executive 1", "Analyst 2";
                full names, "Mr. Surname", surnames, first names and common nicknames.
                Executives keep their title in the list (roles are not identifying).
  firms         the analysts' brokerages -> [FIRM]
  company       the company's name and its variants (without Inc, Corp, Holdings ...),
                and its first word when that word is distinctive -> [COMPANY]
  ticker        "NYSE:LW", "(LW)" and the bare ticker when it can't be an ordinary word
                -> [TICKER]
  web           websites and e-mail addresses -> [WEBSITE]
  dates         full dates and month-day dates -> [DATE]; numeric dates -> [DATE]
  years         every year, relative to the call's year, so "fiscal 2018 outlook" on a
                2017 call becomes "fiscal [YEAR+1] outlook"; also FY17, F4Q08, '17
  --ner         also replaces any remaining PERSON / ORG / PRODUCT names found by spaCy
                (en_core_web_sm). Stricter, but takes hours on all calls.

What is NOT removed: numbers (revenue, EPS, store counts), product and brand names,
places, customers and competitors (unless --ner). A model that knows the company well
may still recognise it from these; check `leaks` and spot-check with --preview.

Run from the FinLLM project root, after data_collection/build_calls.py:
    python data_collection/anonymize_calls.py                  # all calls
    python data_collection/anonymize_calls.py --limit 200      # quick test
    python data_collection/anonymize_calls.py --preview 12345  # show one call before/after
"""
import argparse
import os
import re
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(os.environ.get("FINLLM_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "Earnings_call_transcripts"))
CALLS = HERE / "sp1500_earnings_calls.parquet"
OUT = HERE / "sp1500_earnings_calls_anon.parquet"
BATCH_ROWS = 2000

EXEC_HEADS = {"executives", "executive", "company participants", "corporate participants",
              "company representatives"}
ANALYST_HEADS = {"analysts", "analyst", "conference call participants", "other participants"}
LEGAL_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
                  "plc", "llc", "lp", "l.p", "n.v", "nv", "s.a", "sa", "ag", "se", "holdings",
                  "holding", "group", "the", "&"}
# words that are company first words / surnames / first names but also ordinary English;
# these are only replaced as part of a longer name
COMMON = set("""
a an and the of for in on at to by with from new first general american national united
international global energy financial capital health bank community home life real data select
south north west east southern northern western eastern pacific atlantic central great royal
standard universal advanced applied trust public service services systems industries
technologies solutions resources partners brands foods products group regional federal
mid midwest us u.s alpha omega core prime summit pioneer liberty eagle star sun gold silver
park hill price long young strong good best white black brown green gray grey king rich
will may mark grant bill frank chase dean wade hope joy faith rose june april summer art guy
close bond cash dollar gain gross net margin sales sale cost case lane wall street street
""".split())
# capitalised first names that are also common sentence-initial words
FIRST_COMMON = set("will may grant hope joy faith june april summer art guy chase long price sky rose "
                   "ray gene pat sue don".split())
NICKNAMES = {
    "thomas": ["tom", "tommy"], "robert": ["rob", "bob", "bobby", "robbie"],
    "william": ["bill", "will", "billy"], "michael": ["mike", "mick"], "richard": ["rick", "rich", "dick"],
    "james": ["jim", "jimmy", "jamie"], "joseph": ["joe", "joey"], "charles": ["charlie", "chuck"],
    "christopher": ["chris"], "daniel": ["dan", "danny"], "david": ["dave"], "edward": ["ed", "eddie", "ted"],
    "jonathan": ["jon"], "john": ["jack", "johnny"], "kenneth": ["ken", "kenny"], "matthew": ["matt"],
    "nicholas": ["nick"], "patrick": ["pat"], "peter": ["pete"], "stephen": ["steve"],
    "steven": ["steve"], "timothy": ["tim"], "anthony": ["tony"], "andrew": ["andy", "drew"],
    "benjamin": ["ben"], "donald": ["don"], "douglas": ["doug"], "gregory": ["greg"],
    "jeffrey": ["jeff"], "jeffery": ["jeff"], "lawrence": ["larry"], "leonard": ["len"],
    "ronald": ["ron"], "samuel": ["sam"], "frederick": ["fred"], "gerald": ["jerry"],
    "alexander": ["alex"], "elizabeth": ["liz", "beth"], "katherine": ["kate", "kathy"],
    "catherine": ["cathy", "kate"], "jennifer": ["jen", "jenny"], "margaret": ["meg", "peggy"],
    "patricia": ["pat", "patty"], "susan": ["sue"], "deborah": ["deb"], "rebecca": ["becky"],
    "christine": ["chris"], "kimberly": ["kim"], "victoria": ["vicky"], "zachary": ["zach"],
    "philip": ["phil"], "phillip": ["phil"], "raymond": ["ray"], "ronald ": ["ron"],
    "eugene": ["gene"], "harold": ["hal"], "henry": ["hank"], "terrence": ["terry"],
    "theodore": ["ted"], "vincent": ["vince"], "walter": ["walt"], "jacob": ["jake"],
    "joshua": ["josh"], "nathaniel": ["nate"], "nathan": ["nate"], "frank": ["frankie"],
}
NAME_JUNK = {"jr", "sr", "ii", "iii", "iv", "cfa", "cpa", "phd", "ph.d", "md", "dr", "mr", "ms", "mrs", "esq",
             # placeholders in the participant lists, and our own labels
             "analyst", "analysts", "executive", "executives", "operator", "unidentified", "unknown",
             "participant", "company", "representative", "speaker", "person"}
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
RE_FULL_DATE = re.compile(rf"\b{MONTH}\.?\s+{DAY},?\s+(?:19|20)\d{{2}}\b")
RE_MONTH_DAY = re.compile(rf"\b{MONTH}\.?\s+{DAY}\b(?!\s*(?:%|percent|million|billion))")
RE_DAY_OF_MONTH = re.compile(rf"\b{DAY}\s+of\s+{MONTH}\b")
RE_NUM_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b")
RE_YEAR = re.compile(r"(?<![\d.$])(?<!\d,)(?:19[5-9]\d|20[0-4]\d)(?!\d|,\d|\.\d)")
RE_FY2 = re.compile(r"\b(FY|F)\s?'?(\d{2})\b")
RE_FQ = re.compile(r"\bF([1-4])Q\s?'?(\d{2})\b")
RE_APOS_YEAR = re.compile(r"(?<=\s)'(\d{2})\b")
RE_COMPANY_SUFFIX = re.compile(r"\[COMPANY\],?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|"
                               r"Limited|plc|PLC|LLC|L\.P|LP|Holdings|Group)\b\.?")
RE_WEB = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\."
                    r"(?:com|net|org|io|co|us|biz|info)(?:/[^\s]*)?\b", re.I)
RE_TICKER_EXCH = r"\b(?:NYSE|NASDAQ|Nasdaq|NasdaqGS|NasdaqGM|NYSEMKT|NYSE ?American|AMEX)\s*:\s*{}\b"


def _norm_text(text):
    # the dataset encodes curly quotes and dashes as U+FFFD
    return text.replace(" \ufffd ", " - ").replace("\ufffd", "'").replace("\xa0", " ")


def _rel_year(year, ref):
    k = year - ref
    return "[YEAR]" if k == 0 else f"[YEAR{k:+d}]"


def _two_digit(yy):
    yy = int(yy)
    return 2000 + yy if yy < 50 else 1900 + yy


def _split_participant(line):
    """'Name - Title' (or 'Name, Title') -> (name, title)."""
    for sep in (" - ", " -- ", " – ", " — ", ", "):
        if sep in line:
            name, title = line.split(sep, 1)
            return name.strip(), title.strip()
    return line.strip(), ""


def _name_tokens(name):
    toks = [t.strip(".,") for t in name.split()]
    return [t for t in toks if t and t[0].isupper() and t.lower().strip(".") not in NAME_JUNK
            and len(t.strip(".")) > 1]


def strip_header(lines):
    """Drop the Seeking Alpha header; return (body lines, index of participant list or None)."""
    low = [l.strip().lower() for l in lines]
    for i, l in enumerate(low[:80]):
        if l in EXEC_HEADS:
            return lines[i:], 0
    for i, l in enumerate(low[:80]):
        if l == "operator":
            return lines[i:], None
    for i, l in enumerate(low[:15]):
        if "earnings call" in l or "conference call" in l:
            return lines[i + 1:], None
    return lines, None


def parse_participants(lines):
    """Executives and analysts from the participant list at the top of the body."""
    execs, analysts, section, end = [], [], None, 0
    for i, line in enumerate(lines[:120]):
        l = line.strip()
        if l.lower() in EXEC_HEADS:
            section = execs
        elif l.lower() in ANALYST_HEADS:
            section = analysts
        elif l.lower() == "operator" or len(l) > 120 or section is None:
            end = i
            break
        elif l:
            section.append(_split_participant(l))
        end = i + 1
    return execs, analysts, end


def _alternation(words):
    words = sorted({w for w in words if w}, key=len, reverse=True)
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(w) for w in words) + r")(?![\w])") if words else None


def _company_variants(company):
    if not isinstance(company, str) or not company.strip():
        return set(), None
    full = company.strip().rstrip(".")
    toks = full.replace(",", " ").split()
    while len(toks) > 1 and toks[-1].lower().strip(".") in LEGAL_SUFFIXES:
        toks.pop()
    while len(toks) > 1 and toks[0].lower() == "the":
        toks.pop(0)
    core = " ".join(toks)
    variants = {full, company.strip(), core, core.upper()}
    if len(toks) >= 3:
        variants.add(" ".join(toks[:2]))
    first = toks[0] if toks else ""
    if len(first) >= 4 and first.lower().strip(".") not in COMMON and not first.isdigit():
        variants.update({first, first.upper()})
    return {v for v in variants if len(v) >= 3}, core


def anonymize(text, ticker, company, call_date, nlp=None):
    """Anonymized text and the number of substitutions."""
    n = 0
    lines = _norm_text(text).split("\n")
    lines, _ = strip_header(lines)
    execs, analysts, end = parse_participants(lines)

    people = {}   # surface form -> label
    ambiguous_first = set()
    first_seen = {}
    labels = []
    for kind, group in (("Executive", execs), ("Analyst", analysts)):
        for j, (name, _) in enumerate(group, 1):
            label = f"{kind} {j}"
            labels.append((label, name))
            toks = _name_tokens(name)
            if not toks:
                continue
            people[name] = label
            first, last = toks[0], toks[-1]
            firsts = [first] + [nk.capitalize() for nk in NICKNAMES.get(first.lower(), [])]
            for f in firsts:
                people[f"{f} {last}"] = label
            for title in ("Mr.", "Ms.", "Mrs.", "Dr.", "Mr", "Ms", "Mrs", "Dr"):
                people[f"{title} {last}"] = label
                people[f"{title} {name}"] = label
                people[f"{title} {first} {last}"] = label
            if len(last) >= 3 and last.lower() not in COMMON:
                people[last] = label
            for f in firsts:
                if f in first_seen and first_seen[f] != label:
                    ambiguous_first.add(f)
                first_seen[f] = label
    for f, label in first_seen.items():
        if len(f) >= 3 and f.lower() not in FIRST_COMMON:
            people[f] = "[PERSON]" if f in ambiguous_first else label

    firms = set()
    for _, title in analysts:
        if title:
            firms.add(title)
            toks = title.replace(",", " ").split()
            while len(toks) > 1 and toks[-1].lower().strip(".") in LEGAL_SUFFIXES | {"securities", "llc"}:
                toks.pop()
            firms.add(" ".join(toks))

    # rewrite the participant list: executives keep their title, analysts lose their firm
    head = []
    for line in lines[:end]:
        l = line.strip()
        if l.lower() in EXEC_HEADS | ANALYST_HEADS or not l:
            head.append(line)
            continue
        name, title = _split_participant(l)
        label = people.get(name, "[PERSON]")
        head.append(f"{label} - {title}" if title and label.startswith("Executive") else label)
    body = "\n".join(head + lines[end:])

    def sub(pattern, repl, s):
        nonlocal n
        if pattern is None:
            return s
        s, k = pattern.subn(repl, s)
        n += k
        return s

    names_re = _alternation(people)
    body = sub(names_re, lambda m: people[m.group(0)], body)
    variants, core = _company_variants(company)
    body = sub(_alternation(variants), "[COMPANY]", body)
    # distinctive names also inside run-together or derived words (Alcoans, attendingAlcoa)
    glued = sorted((v for v in variants if len(v) >= 5 and v[0].isupper() and v.lower() not in COMMON),
                   key=len, reverse=True)
    if glued:
        body = sub(re.compile("(?:" + "|".join(re.escape(v) for v in glued) + ")[a-z]*"), "[COMPANY]", body)
    body = sub(RE_COMPANY_SUFFIX, "[COMPANY]", body)
    body = sub(_alternation(firms - {""}), "[FIRM]", body)
    if isinstance(ticker, str) and ticker:
        t = re.escape(ticker)
        body = sub(re.compile(RE_TICKER_EXCH.format(t)), "[TICKER]", body)
        body = sub(re.compile(rf"\(\s*{t}\s*\)"), "([TICKER])", body)
        if len(ticker) >= 3 and ticker.lower() not in COMMON:
            body = sub(re.compile(rf"(?<![\w$]){t}(?![\w])"), "[TICKER]", body)
    body = sub(RE_WEB, "[WEBSITE]", body)

    ref = pd.Timestamp(call_date).year if pd.notna(call_date) else None
    body = sub(RE_FULL_DATE, "[DATE]", body)
    body = sub(RE_DAY_OF_MONTH, "[DATE]", body)
    body = sub(RE_MONTH_DAY, "[DATE]", body)
    body = sub(RE_NUM_DATE, "[DATE]", body)
    if ref is not None:
        body = sub(RE_FQ, lambda m: f"Q{m.group(1)} {_rel_year(_two_digit(m.group(2)), ref)}", body)
        body = sub(RE_FY2, lambda m: f"{m.group(1)}{_rel_year(_two_digit(m.group(2)), ref)}", body)
        body = sub(RE_APOS_YEAR, lambda m: _rel_year(_two_digit(m.group(1)), ref), body)
        body = sub(RE_YEAR, lambda m: _rel_year(int(m.group(0)), ref), body)
    else:
        body = sub(RE_YEAR, "[YEAR]", body)

    if nlp is not None:
        tags = {"PERSON": "[PERSON]", "ORG": "[ORG]", "PRODUCT": "[PRODUCT]"}
        out, last_end = [], 0
        for ent in nlp(body).ents:
            if ent.label_ in tags and not ent.text.startswith("["):
                out.append(body[last_end:ent.start_char] + tags[ent.label_])
                last_end = ent.end_char
                n += 1
        body = "".join(out) + body[last_end:]

    text = body
    return text, n, count_leaks(text, ticker, core, [name for _, name in labels])


def count_leaks(text, ticker, core, names):
    """Mentions of the company's core name, its ticker or a participant's surname still in text."""
    pats = []
    if core and len(core) >= 3:
        pats.append(re.escape(core))
    if isinstance(ticker, str) and len(ticker) >= 3 and ticker.lower() not in COMMON:
        pats.append(rf"(?<![\w$]){re.escape(ticker)}(?![\w])")
    for name in names:
        toks = _name_tokens(name)
        if toks and len(toks[-1]) >= 3 and toks[-1].lower() not in COMMON:
            pats.append(rf"\b{re.escape(toks[-1])}\b")
    return len(re.findall("|".join(pats), text)) if pats else 0


_NLP = None


def _work(rec, ner=False):
    global _NLP
    if ner and _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        _NLP.max_length = 2_000_000
    call_id, ticker, company, call_date, text = rec
    if not isinstance(text, str):
        return call_id, None, 0, 0
    out, n, leaks = anonymize(text, ticker, company, call_date, _NLP if ner else None)
    return call_id, out, n, leaks


def run(out_path, limit=None, workers=None, ner=False):
    f = pq.ParquetFile(CALLS)
    cols = ["call_id", "ticker", "company", "call_date", "text"]
    schema = pa.schema([("call_id", pa.int64()), ("text", pa.string()),
                        ("n_replaced", pa.int32()), ("leaks", pa.int32())])
    tmp = Path(str(out_path) + ".tmp")
    writer = pq.ParquetWriter(tmp, schema)
    done, leaky, t0 = 0, 0, time.time()
    total = min(limit, f.metadata.num_rows) if limit else f.metadata.num_rows
    with Pool(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for batch in f.iter_batches(batch_size=BATCH_ROWS, columns=cols):
            d = batch.to_pandas()
            if limit:
                d = d.iloc[:max(0, limit - done)]
            if d.empty:
                break
            recs = list(zip(d["call_id"], d["ticker"], d["company"], d["call_date"], d["text"]))
            res = pool.map(partial(_work, ner=ner), recs, chunksize=20)
            ids, texts, ns, leaks = zip(*res)
            writer.write_table(pa.table({"call_id": pa.array(ids, pa.int64()), "text": pa.array(texts, pa.string()),
                                         "n_replaced": pa.array(ns, pa.int32()),
                                         "leaks": pa.array(leaks, pa.int32())}, schema=schema))
            done += len(d)
            leaky += sum(1 for k in leaks if k)
            rate = done / (time.time() - t0)
            print(f"\r  {done}/{total} calls  ({rate:.0f}/s, ~{(total - done) / rate / 60:.0f} min left)",
                  end="", flush=True)
            if limit and done >= limit:
                break
    writer.close()
    os.replace(tmp, out_path)
    print(f"\n{done} calls -> {out_path}")
    print(f"{leaky} calls ({leaky / max(done, 1):.1%}) still mention the company, ticker or a participant's "
          f"surname; inspect one with --preview <call_id>")


def preview(call_id, ner=False, chars=3000):
    import pyarrow.compute as pc
    t = pq.read_table(CALLS, columns=["call_id", "ticker", "company", "call_date", "text"],
                      filters=[("call_id", "=", int(call_id))]).to_pandas()
    if t.empty:
        raise SystemExit(f"no call_id {call_id}")
    r = t.iloc[0]
    _, out, n, leaks = _work((r.call_id, r.ticker, r.company, r.call_date, r.text), ner=ner)
    print(f"{r.ticker} | {r.company} | {pd.Timestamp(r.call_date).date()} | {n} replacements, {leaks} leaks")
    print("=" * 30, "before", "=" * 30)
    print(_norm_text(r.text)[:chars])
    print("=" * 30, "after", "=" * 31)
    print(out[:chars])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT), help=f"output parquet (default {OUT})")
    ap.add_argument("--limit", type=int, help="only the first N calls (for testing)")
    ap.add_argument("--workers", type=int, help="processes (default: all cores but one)")
    ap.add_argument("--ner", action="store_true", help="also remove spaCy PERSON/ORG/PRODUCT names (slow)")
    ap.add_argument("--preview", type=int, metavar="CALL_ID", help="print one call before and after")
    a = ap.parse_args()
    if a.preview is not None:
        preview(a.preview, ner=a.ner)
    else:
        run(a.out, limit=a.limit, workers=a.workers, ner=a.ner)
