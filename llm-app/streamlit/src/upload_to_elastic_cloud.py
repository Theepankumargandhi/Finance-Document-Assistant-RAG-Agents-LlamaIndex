#!/usr/bin/env python3
"""
Uploads data/processed/finance/allDocuments.json → your Elastic Cloud index
"""

import os
import json
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

# -----------------------------
# 0) Load environment (.env at repo root)
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # llm-app/
load_dotenv(ROOT / ".env")

# -----------------------------
# 1) Configuration (from env)
# -----------------------------
ES_HOST    = os.getenv("ES_HOST", "").strip()
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
INDEX_NAME = os.getenv("ES_INDEX", "finance_docs").strip()

if not ES_HOST or not ES_API_KEY:
    raise SystemExit(
        "Missing ES_HOST or ES_API_KEY.\n"
        "Add them to your .env (see .env.example):\n"
        "  ES_HOST=https://<your-elastic-host>:443\n"
        "  ES_API_KEY=<your-elastic-api-key>\n"
        "  ES_INDEX=finance_docs"
    )

print("ES host:", urlsplit(ES_HOST).hostname)
print("Index  :", INDEX_NAME)

# -----------------------------
# 2) Connect to Elastic Cloud
# -----------------------------
es = Elasticsearch(ES_HOST, api_key=ES_API_KEY)
info = es.info().body
print("✅ Connected to Elastic Cloud:", info.get("cluster_name", "<unknown>"))

# -----------------------------
# 3) Create index if not exists
# -----------------------------
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "title":    {"type": "text"},
                    "content":  {"type": "text"},
                    "metadata": {"type": "object"},
                }
            }
        },
    )
    print(f"✅ Created index: {INDEX_NAME}")

# -----------------------------
# 4) Load your JSON documents
# -----------------------------
json_path = (Path(__file__).resolve()
             .parents[2]  # llm-app/
             / "data" / "processed" / "finance" / "allDocuments.json")

with json_path.open("r", encoding="utf-8") as f:
    docs = json.load(f)

# -----------------------------
# 5) Bulk upload (id-safe, batched)
# -----------------------------
def iter_actions(items):
    for doc in items:
        _id = doc.get("id") or doc.get("_id")
        yield {"_index": INDEX_NAME, "_id": _id, "_source": doc}

success, errors = helpers.bulk(es, iter_actions(docs), stats_only=False, raise_on_error=False)
print(f"✅ Uploaded {success} documents to '{INDEX_NAME}'")
if errors:
    print(f"⚠️  {len(errors)} errors reported (showing first 3):")
    for e in errors[:3]:
        print(e)
