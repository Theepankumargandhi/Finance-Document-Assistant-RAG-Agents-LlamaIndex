#!/usr/bin/env python3
"""
Uploads data/processed/finance/allDocuments.json → your Elastic Cloud index
with MLflow tracking (params, metrics, artifacts).
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from typing import Iterable, Dict, Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
import mlflow


# -----------------------------
# 0) Load environment (.env at repo root)
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # points to llm-app/
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

host_label = urlsplit(ES_HOST).hostname or ES_HOST
print("ES host:", host_label)
print("Index  :", INDEX_NAME)

# -----------------------------
# 2) Resolve input JSON
# -----------------------------
json_path = (
    Path(__file__).resolve()
    .parents[2]  # llm-app/
    / "data" / "processed" / "finance" / "allDocuments.json"
)

if not json_path.exists():
    raise SystemExit(f"Input file not found: {json_path}")

with json_path.open("r", encoding="utf-8") as f:
    docs = json.load(f)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

file_hash = _sha256(json_path)
num_input_docs = len(docs)
print(f"Found {num_input_docs} documents in {json_path.name} (sha256={file_hash[:12]}…)")

# -----------------------------
# 3) Connect to Elastic Cloud
# -----------------------------
es = Elasticsearch(ES_HOST, api_key=ES_API_KEY)
info = es.info().body
print("✅ Connected to Elastic Cloud:", info.get("cluster_name", "<unknown>"))

# -----------------------------
# 4) Create index if not exists (simple mapping)
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
# 5) Bulk upload (id-safe, batched)
# -----------------------------
def iter_actions(items: Iterable[Dict[str, Any]]):
    for doc in items:
        _id = doc.get("id") or doc.get("_id")
        yield {"_index": INDEX_NAME, "_id": _id, "_source": doc}

# -----------------------------
# 6) MLflow: log params/metrics/artifact around the bulk ingest
# -----------------------------
mlflow.set_experiment("finance-rag")
run_name = f"ingest-elastic-{INDEX_NAME}"

with mlflow.start_run(run_name=run_name):
    # Params that describe the run
    mlflow.log_param("es_host", host_label)
    mlflow.log_param("es_index", INDEX_NAME)
    mlflow.log_param("input_json", str(json_path))
    mlflow.log_param("input_sha256", file_hash)
    mlflow.log_param("input_doc_count", num_input_docs)
    # Helpful for CI builds
    if os.getenv("GITHUB_SHA"):
        mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA"))

    # Perform bulk ingest
    success, errors = helpers.bulk(
        es,
        iter_actions(docs),
        stats_only=False,
        raise_on_error=False
    )

    # Metrics
    mlflow.log_metric("docs_uploaded", int(success))
    mlflow.log_metric("errors_count", int(len(errors) if errors else 0))

    # Attach the exact dataset used
    mlflow.log_artifact(str(json_path), artifact_path="datasets")

    print(f"✅ Uploaded {success} documents to '{INDEX_NAME}'")
    if errors:
        print(f"⚠️  {len(errors)} errors reported (showing first 3):")
        for e in errors[:3]:
            print(e)
