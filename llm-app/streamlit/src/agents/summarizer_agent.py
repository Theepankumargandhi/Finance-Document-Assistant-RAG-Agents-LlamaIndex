from typing import Dict, Any, List
from transformers import pipeline
from src.core.search_backend import hybrid

# Local summarization model (fast and no API keys needed)
_summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize_query(query: str, k: int = 6, max_chars: int = 6000) -> Dict[str, Any]:
    """
    Retrieve top-k docs via hybrid search (BM25 + vector RRF) and summarize them.

    Args:
        query: user question
        k: number of documents to retrieve
        max_chars: maximum total characters passed to summarizer

    Returns:
        dict with {"summary": str, "sources": list[dict]}
    """
    hits = hybrid(query, k=k)
    texts: List[str] = []

    for h in hits:
        src = h.get("_source", {}) or {}
        # Support both "text" (new) and "content" (old)
        t = (src.get("text") or src.get("content") or "").strip()
        if t:
            texts.append(t)

    if not texts:
        return {"summary": "No relevant passages found to summarize.", "sources": []}

    joined = "\n\n".join(texts)[:max_chars]
    out = _summ(joined, max_length=220, min_length=90, do_sample=False)[0]["summary_text"].strip()
    return {"summary": out, "sources": hits}
