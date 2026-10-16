"""Build the full DataFrame of every S&P 1500 earnings call in the dataset.

One row per transcript: directory.csv metadata + index membership (sp1500.ipynb)
+ daily prices around the call (Prices&Indices/results_CSV_*.csv) + full text.
Each row gets a stable `call_id`, which the catalogue (build_catalogue.py) shares.

Run from the FinLLM project root:
    python data_collection/build_calls.py
then build the catalogue:
    python data_collection/build_catalogue.py
"""
import os
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# raw dataset and outputs both live in Earnings_call_transcripts/ (gitignored, so it has to be
# copied to each machine); set FINLLM_DATA_DIR to use a folder somewhere else
HERE = Path(os.environ.get("FINLLM_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "Earnings_call_transcripts"))
OUT = HERE / "sp1500_earnings_calls.parquet"
CONSTITUENTS_CACHE = HERE / "sp1500_constituents.csv"

HEADERS = {"User-Agent": "Mozilla/5.0"}  # Wikipedia returns 403 without a UA
INDEXES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def get_constituents(refresh: bool = False) -> pd.DataFrame:
    """Current S&P 500/400/600 members from Wikipedia (same source as sp1500.ipynb)."""
    if CONSTITUENTS_CACHE.exists() and not refresh:
        return pd.read_csv(CONSTITUENTS_CACHE)
    frames = []
    for name, url in INDEXES.items():
        html = requests.get(url, headers=HEADERS, timeout=30).text
        tables = pd.read_html(StringIO(html))
        tbl = next(t for t in tables if any(c in t.columns for c in ("Symbol", "Ticker symbol")))
        tbl = tbl.rename(columns={"Ticker symbol": "Symbol"})
        df = tbl[["Symbol", "Security", "GICS Sector"]].copy()
        df["Index"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["Symbol"] = out["Symbol"].str.strip()
    out = out.drop_duplicates("Symbol")
    out.to_csv(CONSTITUENTS_CACHE, index=False)
    return out


def load_prices() -> pd.DataFrame:
    """Close prices for calendar days -10..+10 around each call (NaN on non-trading days),
    plus the raw call timestamp."""
    r = pd.concat(
        (pd.read_csv(p) for p in sorted((HERE / "Prices&Indices").glob("results_CSV_*.csv"))),
        ignore_index=True,
    )
    day_cols = [c for c in r.columns if c.startswith("day_")]
    r = r.rename(columns={"date": "call_datetime_raw", **{c: "price_" + c for c in day_cols}})
    r["_key"] = r["file_name"].map(_norm)
    keep = ["_key", "chunk", "call_datetime_raw"] + ["price_" + c for c in day_cols]
    return r[keep].drop_duplicates(["_key", "chunk"])


def _norm(name: str) -> str:
    # directory.csv file names were produced with rstrip(".txt"), which also ate
    # trailing t's ("..._Transcrip"); normalise both sides the same way to join.
    return name.removesuffix(".txt").rstrip(".tx")


class TranscriptReader:
    """Reads transcript text by (chunk, file_name). Layout: data_{chunk//100}.zip/data_i/chunk_{chunk}/<file>.txt"""

    def __init__(self):
        self.zips, self.members = {}, {}
        for zi in range(6):
            z = self.zips[zi] = zipfile.ZipFile(HERE / f"data_{zi}.zip")
            for m in z.namelist():
                parts = m.split("/")
                if len(parts) == 3 and parts[2]:
                    self.members.setdefault((int(parts[1].split("_")[1]), _norm(parts[2])), (zi, m))

    def read(self, chunk, key):
        hit = self.members.get((chunk, key))
        if hit is None:
            return None
        zi, m = hit
        return self.zips[zi].read(m).decode("utf-8", errors="replace").replace("\r\n", "\n")

    def close(self):
        for z in self.zips.values():
            z.close()


def check_inputs():
    """Fail early, before any downloads, if the raw dataset isn't in HERE."""
    needed = (["directory.csv"] + [f"data_{i}.zip" for i in range(6)]
              + [f"Prices&Indices/results_CSV_{i}.csv" for i in range(6)])
    missing = [n for n in needed if not (HERE / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Raw earnings call dataset not found in {HERE} (missing: {', '.join(missing)}). "
            "Copy the Earnings_call_transcripts folder there, or set FINLLM_DATA_DIR to where it is."
        )


def build_sp1500_calls(save: bool = True) -> pd.DataFrame:
    """Writes OUT (with text) and returns the metadata, i.e. everything except the text."""
    check_inputs()
    sp = get_constituents()
    cat = pd.read_csv(HERE / "directory.csv")

    # Wikipedia writes share classes as BRK.B, Yahoo as BRK-B; accept either spelling.
    lookup = {}
    for _, row in sp.iterrows():
        for sym in {row["Symbol"], row["Symbol"].replace(".", "-"), row["Symbol"].replace("-", ".")}:
            lookup.setdefault(sym, row)
    df = cat[cat["ticker"].isin(lookup)].copy()
    members = pd.DataFrame([lookup[t] for t in df["ticker"]], index=df.index)
    df["sp_index"] = members["Index"]
    df["sp_security"] = members["Security"]
    df["gics_sector"] = members["GICS Sector"]

    df["_key"] = df["file_name"].map(_norm)
    df = df.merge(load_prices(), on=["_key", "chunk"], how="left")

    # directory dates are ISO but a few are garbage (year 0000 / 3009) -> NaT
    df["call_date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df.loc[~df["call_date"].between("2000-01-01", "2020-12-31"), "call_date"] = pd.NaT
    # quarter labels come as "Q3 2019", "F4Q08", "Q1 FY08", "F2Q 2014", ...
    qs = df["quarter"].str.strip()
    qnum = qs.str.extract(r"^Q([1-4])")[0].fillna(qs.str.extract(r"^F([1-4])Q")[0])
    year = pd.to_numeric(qs.str.extract(r"(?:FY|Q|\s)\s*(\d{4}|\d{2})$")[0]).astype("Int64")
    df["fiscal_quarter"] = pd.to_numeric(qnum).astype("Int64")
    df["fiscal_year"] = year.where(year >= 100, year + 2000)

    df["text_chars"] = pd.Series(pd.NA, index=df.index, dtype="Int64")  # filled while writing

    front = ["ticker", "company", "sp_security", "sp_index", "exchange", "sector", "gics_sector",
             "quarter", "fiscal_year", "fiscal_quarter", "call_date", "call_datetime_raw"]
    rest = [c for c in df.columns if c not in front + ["_key", "date"]]
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    df.insert(0, "call_id", df.index)
    keys = df.pop("_key")
    df = df[["call_id"] + front + rest]

    if save:
        write_with_text(df, keys)
    return df


def write_with_text(meta: pd.DataFrame, keys: pd.Series, batch_rows: int = 2000):
    """Write meta + transcript text to OUT a batch at a time, so the full ~2 GB of text is
    never in memory at once (peak stays well under 1 GB). Fills meta["text_chars"] in place."""
    schema = pa.Schema.from_pandas(meta, preserve_index=False).append(pa.field("text", pa.string()))
    reader = TranscriptReader()
    try:
        with pq.ParquetWriter(OUT, schema) as writer:
            for start in range(0, len(meta), batch_rows):
                batch = meta.iloc[start:start + batch_rows].copy()
                texts = [reader.read(c, k) for c, k in zip(batch["chunk"], keys.iloc[start:start + batch_rows])]
                batch["text_chars"] = pd.array([None if t is None else len(t) for t in texts], dtype="Int64")
                batch["text"] = texts
                meta.loc[batch.index, "text_chars"] = batch["text_chars"]
                writer.write_table(pa.Table.from_pandas(batch, schema=schema, preserve_index=False))
    finally:
        reader.close()


if __name__ == "__main__":
    out = build_sp1500_calls()
    print(out.shape[0], "calls,", out.shape[1] + 1, "columns ->", OUT)
    print("missing text:", out["text_chars"].isna().sum())
    print("missing price window:", out["price_day_0"].isna().sum())
    print("missing call_date:", out["call_date"].isna().sum())
    print(out["sp_index"].value_counts().to_string())
    print("tickers:", out["ticker"].nunique())
    print(out.head().T.to_string())
