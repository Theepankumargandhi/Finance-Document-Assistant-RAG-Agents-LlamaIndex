#!/usr/bin/env python3
"""
Converts data/raw/finance/all-data.csv → data/processed/finance/allDocuments.json

- Robust to weird encodings (UTF-8/UTF-8-SIG/CP1252/Latin-1; final fallback replaces bad bytes)
- Auto-detects delimiter (comma/semicolon/tab)
- Expects headers: sentiment, text  (rename your CSV headers if different)

Output schema (LlamaIndex-ready):
[
  {
    "doc_id": "finance-news-000001",
    "title": "Financial snippet #1",
    "text":  "... main body ...",
    "metadata": { "source": "...", "sentiment": "...", "ingest_date": "YYYY-MM-DD" }
  },
  ...
]
"""
import csv, json
from datetime import date
from pathlib import Path

# -------------------------
# Resolve paths robustly
# -------------------------
HERE = Path(__file__).resolve()
# parents: [utils, src, streamlit, llm-app, <repo>]
REPO_ROOT = HERE.parents[4]        # <- repo root
LLM_APP_ROOT = HERE.parents[3]     # <- llm-app/

# Preferred CSV under repo root
csv_repo = REPO_ROOT / "data" / "raw" / "finance" / "all-data.csv"
# Fallback (your earlier layout) under llm-app/
csv_llmapp = LLM_APP_ROOT / "data" / "raw" / "finance" / "all-data.csv"

if csv_repo.exists():
    INPUT_CSV = csv_repo
elif csv_llmapp.exists():
    INPUT_CSV = csv_llmapp
else:
    # last resort: tell user where we looked
    raise FileNotFoundError(
        f"Could not find CSV.\nTried:\n  {csv_repo}\n  {csv_llmapp}\n"
        "Place all-data.csv in one of those locations."
    )

OUT_DIR  = REPO_ROOT / "data" / "processed" / "finance"
OUT_JSON = OUT_DIR / "allDocuments.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]


def open_safely(path: Path):
    # try clean decodes first
    for enc in ENCODINGS:
        try:
            f = open(path, "r", encoding=enc, newline="")
            _ = f.read(4096)  # validate
            f.seek(0)
            print(f"✓ Using encoding: {enc}")
            return f
        except UnicodeDecodeError:
            continue
    # last resort: replace bad bytes so we never crash
    print("⚠️  All strict decodes failed — using utf-8 with replacement.")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def sniff_dialect(sample_text: str):
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=[",",";","\t","|"])
    except Exception:
        class _D(csv.excel):
            delimiter = ","
        return _D


def pick_header(header_map, candidates):
    for canon in candidates:
        for raw, norm in header_map.items():
            if norm == canon:
                return raw
    return None


def main():
    print(f"Input CSV: {INPUT_CSV}")
    docs = []

    try:
        f = open_safely(INPUT_CSV)
        sample = f.read(8192)
        f.seek(0)
        dialect = sniff_dialect(sample)
        reader = csv.DictReader(f, dialect=dialect)

        print("CSV headers detected:", reader.fieldnames)

        # Normalize header keys (lowercase, strip spaces)
        header_map = {h: (h or "").strip().lower() for h in (reader.fieldnames or [])}

        sentiment_key = pick_header(header_map, ["sentiment","label"])
        text_key      = pick_header(header_map, ["text","sentence","content","body"])
        title_key     = pick_header(header_map, ["title","headline"])

        if not sentiment_key or not text_key:
            raise RuntimeError(
                f"Couldn't find required columns in {reader.fieldnames}. "
                "Expect something like sentiment/text."
            )

        for i, row in enumerate(reader, start=1):
            text = (row.get(text_key) or "").strip()
            if len(text) < 60:  # skip very short lines to reduce noise
                continue

            title = (row.get(title_key) or "").strip() if title_key else ""
            sent  = (row.get(sentiment_key) or "").strip()

            docs.append({
                "doc_id": f"finance-news-{i:06d}",
                "title": title or f"Financial snippet #{i}",
                "text": text,   # standardized key used by our LlamaIndex backend
                "metadata": {
                    "source": "Financial PhraseBank",
                    "sentiment": sent,
                    "ingest_date": str(date.today()),
                }
            })

        f.close()

        with open(OUT_JSON, "w", encoding="utf-8") as out:
            json.dump(docs, out, ensure_ascii=False, indent=2)

        print(f"\n✅ Wrote {len(docs)} documents to:\n{OUT_JSON}\n")

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
