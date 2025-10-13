#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Uploads data/processed/finance/allDocuments.json -> your Elastic Cloud index
"""

import os, json
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

# --- Load .env from repo root (adjust parents if layout differs) ---
ROOT = Path(__file__).resolve().parents[2]   # points to llm-app/
load_dotenv(ROOT / ".env")

# --- Configuration from environment (NO SECRETS IN CODE) ---
ES_HOST    = os.getenv("ES_HOST", "").strip()
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
INDEX_NAME = os.getenv("ES_INDEX", "finance_docs").strip()

if not ES_HOST or not ES_API_KEY:
    raise SystemExit(
        "Missing ES_HOST or ES_API_KEY.\n"
        "Create a .env (see .env.example) with:\n"
        "  ES_HOST=https://<your-elastic-host>:443\n"
        "  ES_API_KEY=<your-elastic-api-key>\n"
        "  ES_INDEX=finance_docs"
    )

print("ES host:", urlsplit(ES_HOST).hostname)
print("Index  :", INDEX_NAME)

# --- Connect to Elastic Cloud ---
es = Elasticsearch(ES_HOST, api_key=ES_API_KEY)
print("✅ Connected to Elastic Cloud:",
      es.info().body.get("cluster_name", "<unknown>"))

# --- Create index if not exists (rest of your file continues unchanged) ---
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body={
        "mappings": {
            "properties": {
                "title":    {"type": "text"},
                "content":  {"type": "text"},
                "metadata": {"type": "object"}
            }
        }
    })
    print(f"✅ Created index: {INDEX_NAME}")

es = Elasticsearch(ES_HOST, api_key=ES_API_KEY)

def search(q, k=5):
    resp = es.search(
        index=INDEX,
        query={"match": {"content": q}},
        size=k,
        source_includes=["id","title","content","metadata"]
    )
    print(f"\nQ: {q}\nTop {k} hits:")
    for i, hit in enumerate(resp["hits"]["hits"], 1):
        print(f"{i}. {_short(hit['_source']['content'])}  (score={hit['_score']:.2f})")

def _short(text, n=140):
    t = " ".join(text.split())
    return t[:n] + ("..." if len(t) > n else "")

if __name__ == "__main__":
    # Try a few finance-y queries
    search("operating profit increased")
    search("net sales growth")
    search("share buyback program")
