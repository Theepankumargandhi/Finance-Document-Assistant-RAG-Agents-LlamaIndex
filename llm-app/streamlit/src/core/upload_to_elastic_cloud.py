#!/usr/bin/env python3
"""
Uploads data/processed/finance/allDocuments.json → Elasticsearch BM25 index,
with MLflow tracking. Vector indexing is handled separately by
add_vectors_and_hybrid_search.py (via LlamaIndex).
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
import mlflow

from .elasticSearch import (
    get_es_client,
    ensure_bm25_index,
    bulk_index_bm25,
)

# -----------------------------
# 0) Load environment
# -----------------------------
# File location: <repo>/llm-app/streamlit/src/core/upload_to_elastic_cloud.py
# parents: [core, src, streamlit, llm-app, <repo>]
ROOT = Path(__file__).resolve().parents[4]  # <- repo root
load_dotenv(ROOT / ".env")

ES_HOST       = os.getenv("ES_HOST", "").strip()
ES_CLOUD_ID   = os.getenv("ES_CLOUD_ID", "").strip()
ES_API_KEY    = os.getenv("ES_API_KEY", "").strip()
ES_BM25_INDEX = os.getenv("ES_BM25_INDEX", os.getenv("ES_INDEX", "finance_docs_bm25")).strip()
ES_TEXT_FIELD = os.getenv("ES_TEXT_FIELD", "text").strip()
ES_ID_FIELD   = os.getenv("ES_ID_FIELD", "doc_id").strip()

if not (ES_HOST or ES_CLOUD_ID):
    raise SystemExit("Missing ES config: set ES_HOST or ES_CLOUD_ID (and ES_API_KEY) in .env")

host_label = (urlsplit(ES_HOST).hostname if ES_HOST else "elastic-cloud")
print("ES host/cluster  :", host_label)
print("BM25 index name  :", ES_BM25_INDEX)
print("Text field key   :", ES_TEXT_FIELD)
print("ID field key     :", ES_ID_FIELD)

# -----------------------------
# 1) Resolve input JSON (at repo root)
# -----------------------------
json_path = ROOT / "data" / "processed" / "finance" / "allDocuments.json"
if not json_path.exists():
    raise SystemExit(f"Input file not found: {json_path}")

with json_path.open("r", encoding="utf-8") as f:
    docs = json.load(f)
if not isinstance(docs, list) or not docs:
    raise SystemExit("Input JSON must be a non-empty array of documents.")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

file_hash = _sha256(json_path)
print(f"Found {len(docs)} docs in {json_path.name} (sha256={file_hash[:12]}…)")

# -----------------------------
# 2) Normalize docs for BM25
# -----------------------------
def _normalize(d):
    d = dict(d)
    # map 'content' or 'text' into ES_TEXT_FIELD if needed
    if ES_TEXT_FIELD not in d:
        if "content" in d:
            d[ES_TEXT_FIELD] = d.get("content", "")
        elif "text" in d:
            d[ES_TEXT_FIELD] = d.get("text", "")
        else:
            d[ES_TEXT_FIELD] = ""
    # ensure ID field present if possible
    if ES_ID_FIELD not in d:
        if "doc_id" in d:
            d[ES_ID_FIELD] = d.get("doc_id")
        elif "id" in d:
            d[ES_ID_FIELD] = d.get("id")
    return d

norm_docs = [_normalize(d) for d in docs]

# -----------------------------
# 3) Connect & ensure index
# -----------------------------
es = get_es_client()
ensure_bm25_index(es, index_name=ES_BM25_INDEX)

# -----------------------------
# 4) MLflow wrapper around bulk indexing
# -----------------------------
mlflow.set_experiment("finance-rag")
run_name = f"bm25-ingest-{ES_BM25_INDEX}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_param("es_host", host_label)
    mlflow.log_param("bm25_index", ES_BM25_INDEX)
    mlflow.log_param("text_field", ES_TEXT_FIELD)
    mlflow.log_param("id_field", ES_ID_FIELD)
    mlflow.log_param("input_json", str(json_path))
    mlflow.log_param("input_sha256", file_hash)
    mlflow.log_param("input_doc_count", len(norm_docs))
    if os.getenv("GITHUB_SHA"):
        mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA"))

    success = bulk_index_bm25(es, norm_docs, index_name=ES_BM25_INDEX)
    mlflow.log_metric("docs_uploaded", int(success))

    mlflow.log_artifact(str(json_path), artifact_path="datasets")
    print(f"✅ BM25 upserts: {success} into '{ES_BM25_INDEX}'")

print("Done. Reminder: run add_vectors_and_hybrid_search.py to build the vector index via LlamaIndex.")
