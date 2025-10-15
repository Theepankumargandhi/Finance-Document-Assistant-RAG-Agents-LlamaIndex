# llm-app/streamlit/src/core/elasticSearch.py
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

load_dotenv()

# Env config (works with Docker/K8s or local .env)
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID", "")
ES_USERNAME = os.getenv("ES_USERNAME", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_API_KEY = os.getenv("ES_API_KEY", "")

ES_BM25_INDEX = os.getenv("ES_BM25_INDEX", os.getenv("ES_INDEX", "finance_docs_bm25"))
ES_TEXT_FIELD = os.getenv("ES_TEXT_FIELD", "text")
ES_TITLE_FIELD = os.getenv("ES_TITLE_FIELD", "title")
ES_ID_FIELD = os.getenv("ES_ID_FIELD", "doc_id")
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def get_es_client() -> Elasticsearch:
    """
    Prefer host+api_key; only use cloud_id if it looks valid.
    """
    if ES_HOST and ES_API_KEY:
        return Elasticsearch(ES_HOST, api_key=ES_API_KEY)

    # Cloud ID must contain a ':' (name:base64(...))
    if ES_CLOUD_ID and ES_API_KEY and (":" in ES_CLOUD_ID):
        return Elasticsearch(cloud_id=ES_CLOUD_ID, api_key=ES_API_KEY)

    if ES_HOST and ES_USERNAME and ES_PASSWORD:
        return Elasticsearch(ES_HOST, basic_auth=(ES_USERNAME, ES_PASSWORD))
    if ES_HOST:
        return Elasticsearch(ES_HOST)
    raise RuntimeError("Elasticsearch connection not configured properly.")


def ensure_bm25_index(es: Elasticsearch, index_name: Optional[str] = None) -> None:
    """
    Create a simple BM25 index if it doesn't exist.
    Keeps mappings minimal so both legacy code and LlamaIndex BM25 retriever can use it.
    """
    idx = index_name or ES_BM25_INDEX
    
    try:
        if es.indices.exists(index=idx):
            print(f"✓ BM25 index '{idx}' already exists")
            return
    except Exception as e:
        print(f"⚠ Error checking index existence: {e}")

    # Basic mapping: text fields analyzed; metadata kept dynamic
    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "similarity": {
                    "default": {"type": "BM25"}
                }
            }
        },
        "mappings": {
            "properties": {
                ES_TEXT_FIELD: {"type": "text"},
                ES_TITLE_FIELD: {"type": "text"},
                ES_ID_FIELD: {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True}
            }
        },
    }
    
    try:
        es.indices.create(index=idx, body=body)
        print(f"✓ Created BM25 index: {idx}")
    except Exception as e:
        print(f"❌ Failed to create BM25 index: {e}")
        raise


def bulk_index_bm25(
    es: Elasticsearch,
    docs: Iterable[Dict],
    index_name: Optional[str] = None,
    id_field: str = ES_ID_FIELD,
    text_field: str = ES_TEXT_FIELD,
) -> int:
    """
    Bulk index plain docs for BM25 searching.
    Each doc should at least have `text_field` in it. If `id_field` is present, it's used as _id.
    Returns number of successful actions.
    """
    idx = index_name or ES_BM25_INDEX
    ensure_bm25_index(es, idx)

    def _actions():
        for d in docs:
            _id = d.get(id_field)
            # Ensure text field exists and is not empty
            if not d.get(text_field):
                continue
                
            yield {
                "_op_type": "index",
                "_index": idx,
                "_id": _id if _id else None,
                "_source": d,
            }

    try:
        success, errors = bulk(es, _actions(), raise_on_error=False)
        if errors:
            print(f"⚠ Some documents failed during bulk indexing: {len(errors)} errors")
        print(f"✓ Indexed {success} documents to '{idx}'")
        return success
    except Exception as e:
        print(f"❌ Bulk indexing failed: {e}")
        return 0


def bm25_search(
    es: Elasticsearch,
    query: str,
    index_name: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    text_field: str = ES_TEXT_FIELD,
    title_field: str = ES_TITLE_FIELD,
) -> List[Dict]:
    """
    BM25 search over `text_field` (+ boosts for title if present).
    Returns a list of `_source` dicts augmented with `_score` and `_es_id`.
    """
    idx = index_name or ES_BM25_INDEX

    fields = [text_field]
    # If title exists in mapping, keep a small boost
    if title_field:
        fields.append(f"{title_field}^2")

    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "type": "best_fields",
                    }
                }
            }
        },
    }

    try:
        resp = es.search(index=idx, body=search_query)
        hits = resp.get("hits", {}).get("hits", [])
        out = []
        for h in hits:
            src = h.get("_source", {}).copy()
            src["_score"] = h.get("_score")
            src["_es_id"] = h.get("_id")
            out.append(src)
        return out
    except Exception as e:
        print(f"❌ BM25 search error: {e}")
        return []


# ---- Backward compatible names (so other files won't break immediately) ----
def getEsClient() -> Elasticsearch:
    return get_es_client()


def elasticSearch(esClient: Elasticsearch, query: str, indexName: str) -> List[Dict]:
    # Keep legacy signature but route to new function
    return bm25_search(esClient, query, index_name=indexName, top_k=DEFAULT_TOP_K)