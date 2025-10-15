# search_backend.py
# llm-app/streamlit/src/core/search_backend.py
"""
Bridged search backend that preserves the old function names
(bm25 / knn / hybrid / hybrid_rerank) but delegates to LlamaIndex.

- bm25       -> LlamaIndex BM25 retriever (Elasticsearch)
- knn        -> LlamaIndex vector retriever (Elasticsearch vector store)
- hybrid     -> LlamaIndex QueryFusionRetriever (RRF over bm25 + vector)
- reranking  -> optional CrossEncoder pass over fused results

Return shape stays compatible with the old code: a list of ES-like hits,
each item shaped like:
  {
    "_id": "<doc_id>",
    "_score": <float>,            # similarity or ES score
    "_rrf": <float>,              # only for hybrid[_rerank]
    "_source": {
        "id": "<doc_id>",         # if available from metadata
        "title": "...",           # if available
        "content": "...",         # node text
        "metadata": {...}         # remaining metadata
    }
  }
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv, find_dotenv
from llama_index.core.schema import NodeWithScore

# LlamaIndex backend we created
from .llamaindex_backend import LlamaIndexBackend

# Optional cross-encoder reranker (kept from your original impl)
from sentence_transformers import CrossEncoder

# ---- Load .env robustly (same behavior you had) ----
dotenv_path = find_dotenv(filename=".env", usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    here = os.path.dirname(__file__)
    alt = os.path.abspath(os.path.join(here, "../.env"))
    if os.path.exists(alt):
        load_dotenv(alt)

# Lazy instantiation of backend
_backend = None

def _get_backend():
    global _backend
    if _backend is None:
        _backend = LlamaIndexBackend()
    return _backend

# Light model; swap if you want a bigger one later
_RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_reranker: Optional[CrossEncoder]
try:
    _reranker = CrossEncoder(_RERANK_MODEL_NAME)
except Exception:
    _reranker = None


# ---------- helpers ----------
def _node_to_hit(nws: NodeWithScore) -> Dict:
    """Map LlamaIndex NodeWithScore => ES-like hit dict."""
    node = nws.node
    meta = (node.metadata or {}).copy()

    # Try to extract some common fields
    doc_id = getattr(node, "node_id", None) or meta.get("doc_id") or meta.get("id") or getattr(node, "id_", None)
    title = meta.get("title")
    content = getattr(node, "text", "") or meta.get("text", "")

    src = {
        "id": doc_id,
        "title": title,
        "content": content,
        "metadata": meta,
    }
    hit = {
        "_id": str(doc_id) if doc_id is not None else None,
        "_score": float(nws.score) if nws.score is not None else None,
        "_source": src,
    }
    return hit


def _fuse_rrf(hits_a: List[Dict], hits_b: List[Dict], k: int = 10, c: int = 60) -> List[Dict]:
    """RRF fuse two ES-like hit lists by _id (fallback to object id)."""
    scores = {}
    index = {}

    def add(lst):
        for rank, h in enumerate(lst, 1):
            _id = h.get("_id") or id(h)
            index[_id] = h
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (c + rank)

    add(hits_a)
    add(hits_b)

    fused_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    fused = []
    for _id, rrf in fused_ids:
        h = index[_id]
        out = dict(h)
        out["_rrf"] = rrf
        fused.append(out)
    return fused


# ---------- API (same names as before) ----------
def bm25(query: str, k: int = 10) -> List[Dict]:
    """BM25 only, via LlamaIndex BM25 retriever."""
    nodes = _get_backend().query(query, top_k=k, mode="bm25")
    return [_node_to_hit(n) for n in nodes]


def knn(query: str, k: int = 10, num_candidates: int = 100) -> List[Dict]:
    """
    Vector-only search via LlamaIndex (Elasticsearch vector store).
    num_candidates is ignored (handled internally by store); kept for signature compatibility.
    """
    nodes = _get_backend().query(query, top_k=k, mode="vector")
    return [_node_to_hit(n) for n in nodes]


def hybrid(query: str, k: int = 10, c: int = 60) -> List[Dict]:
    """
    Hybrid (RRF) using LlamaIndex's QueryFusionRetriever directly.
    We still attach an '_rrf' field to each result for continuity.
    """
    # Ask backend for hybrid nodes
    nodes = _get_backend().query(query, top_k=k, mode="hybrid")
    # Convert to ES-like list and synthesize an RRF score locally (rank-based)
    hits = [_node_to_hit(n) for n in nodes]
    # Create a rank-based RRF to populate _rrf similar to your legacy function
    return _fuse_rrf(hits, [], k=k, c=c)


def hybrid_rerank(query: str, k: int = 10, candidates: int = 20, c: int = 60) -> List[Dict]:
    """
    Get hybrid (RRF) candidates and rerank top N with a cross-encoder.
    If the reranker model isn't available, gracefully fall back to plain hybrid.
    """
    # Get more candidates for reranking
    base = hybrid(query, k=min(candidates, max(k, candidates)), c=c)

    if not _reranker:
        # No reranker available; return the hybrid results as-is, truncated to k
        return base[:k]

    # Pair (query, content) for cross-encoder
    pairs = [(query, (h.get("_source") or {}).get("content", "")) for h in base]
    scores = _reranker.predict(pairs).tolist()
    reranked = sorted(zip(base, scores), key=lambda x: x[1], reverse=True)[:k]
    return [h for h, _ in reranked]