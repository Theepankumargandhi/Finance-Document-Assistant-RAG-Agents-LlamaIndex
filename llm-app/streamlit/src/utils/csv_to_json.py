#!/usr/bin/env python3
"""
Converts data/raw/finance/all-data.csv → data/processed/finance/allDocuments.json
- Robust to weird encodings (UTF-8/UTF-8-SIG/CP1252/Latin-1; final fallback replaces bad bytes)
- Auto-detects delimiter (comma/semicolon/tab)
- Expects headers: sentiment, text  (rename your CSV headers if different)
"""

import csv, json, os
from datetime import date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "finance", "all-data.csv")
OUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed", "finance")
OUT_JSON  = os.path.join(OUT_DIR, "allDocuments.json")
os.makedirs(OUT_DIR, exist_ok=True)

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

def open_safely(path):
    # try clean decodes first
    for enc in ENCODINGS:
        try:
            f = open(path, "r", encoding=enc, newline="")
            # try reading a tiny sample to validate
            _ = f.read(4096)
            f.seek(0)
            print(f"✓ Using encoding: {enc}")
            return f
        except UnicodeDecodeError:
            continue
    # last resort: replace bad bytes so we never crash
    print("⚠️  All strict decodes failed — using utf-8 with replacement.")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")

def sniff_dialect(sample_text):
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",",";","\t","|"])
        return dialect
    except Exception:
        # default to comma if sniffing fails
        class _D(csv.excel): delimiter = ","
        return _D

docs = []
try:
    f = open_safely(INPUT_CSV)
    sample = f.read(8192)
    f.seek(0)
    dialect = sniff_dialect(sample)
    reader = csv.DictReader(f, dialect=dialect)

    print("CSV headers detected:", reader.fieldnames)

    # Normalize header keys (lowercase, strip spaces)
    norm = {h: h.strip().lower() for h in (reader.fieldnames or [])}
    # Try common variants for the two required fields
    def pick(key_opts):
        for opt in key_opts:
            for h in norm.values():
                if h == opt:
                    return [k for k,v in norm.items() if v==h][0]
        return None

    sentiment_key = pick(["sentiment","label"])
    text_key      = pick(["text","sentence","content","body"])

    if not sentiment_key or not text_key:
        raise RuntimeError(
            "Couldn't find required columns. "
            f"Have headers={reader.fieldnames}. Need something like sentiment/text."
        )

    for i, row in enumerate(reader, start=1):
        text = (row.get(text_key) or "").strip()
        sent = (row.get(sentiment_key) or "").strip()
        if len(text) < 60:  # skip very short lines
            continue
        docs.append({
            "id": f"finance-news-{i:06d}",
            "title": f"Financial snippet #{i}",
            "content": text,
            "metadata": {
                "source": "Financial PhraseBank",
                "sentiment": sent,
                "ingest_date": str(date.today())
            }
        })

    f.close()

    with open(OUT_JSON, "w", encoding="utf-8") as out:
        json.dump(docs, out, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote {len(docs)} documents to:\n{OUT_JSON}\n")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find file at: {INPUT_CSV}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
