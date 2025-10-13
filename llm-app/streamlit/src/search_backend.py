# search_backend.py
import os
from typing import List, Dict

from dotenv import load_dotenv, find_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# ---- Load .env robustly ----
# This will search upwards from the CWD and also consider the package dir.
# Works whether Streamlit runs from streamlit/ or project root.
dotenv_path = find_dotenv(filename=".env", usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    # Fallback: explicitly try ../.env relative to this file
    here = os.path.dirname(__file__)
    alt = os.path.abspath(os.path.join(here, "../.env"))
    if os.path.exists(alt):
        load_dotenv(alt)

ES_HOST = os.getenv("ES_HOST", "").strip()
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
ES_INDEX = os.getenv("ES_INDEX", "finance_docs").strip()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

# Debug prints so you can see what’s happening in Streamlit logs
print("🔧 search_backend.py loaded")
print("  ES_HOST set? ", bool(ES_HOST))
print("  ES_INDEX    : ", ES_INDEX)

if not ES_HOST or not ES_API_KEY:
    raise ValueError("Missing ES_HOST or ES_API_KEY. Check your .env file path and contents.")

# Single client + single model, module-level
_es = Elasticsearch(hosts=[ES_HOST], api_key=ES_API_KEY)
_model = SentenceTransformer(EMBED_MODEL_NAME)


def bm25(query: str, k: int = 10) -> List[Dict]:
    r = _es.search(
        index=ES_INDEX,
        query={"match": {"content": query}},
        size=k,
        source_includes=["id", "title", "content", "metadata"],
    )
    return r["hits"]["hits"]


def knn(query: str, k: int = 10, num_candidates: int = 100) -> List[Dict]:
    qvec = _model.encode([query], normalize_embeddings=True)[0].tolist()
    r = _es.search(
        index=ES_INDEX,
        knn={"field": "embedding", "query_vector": qvec, "k": k, "num_candidates": num_candidates},
        size=k,
        source_includes=["id", "title", "content", "metadata"],
    )
    return r["hits"]["hits"]


def hybrid(query: str, k: int = 10, c: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion of BM25 + kNN."""
    b = bm25(query, k=k)
    v = knn(query, k=k)

    scores = {}
    def add(lst):
        for rank, h in enumerate(lst, 1):
            scores[h["_id"]] = scores.get(h["_id"], 0.0) + 1.0 / (c + rank)

    add(b); add(v)
    by_id = {h["_id"]: h for h in b + v}
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"_id": _id, "_rrf": s, **by_id[_id]} for _id, s in fused]

from sentence_transformers import CrossEncoder

# one-time load (lightweight model)
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_rerank(query: str, k: int = 10, candidates: int = 20, c: int = 60):
    """RRF fuse BM25+kNN, then rerank top-N with a cross-encoder."""
    fused = hybrid(query, k=candidates, c=c)  # get more than you plan to show
    pairs = [(query, (h.get("_source", {}) or {}).get("content", "")) for h in fused]
    scores = _reranker.predict(pairs).tolist()
    reranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)[:k]
    return [h for h, _ in reranked]
