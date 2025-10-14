#!/usr/bin/env python3
# Adds a dense_vector field to Elastic Cloud, indexes vectors, and runs a hybrid search.
import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

# Load .env from repo root (adjust if your layout differs)
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- CONFIG (from environment) ---
ES_HOST = os.getenv("ES_HOST", "").strip()
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
INDEX = os.getenv("ES_INDEX", "finance_docs").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

# Path to allDocuments.json relative to this file
JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/processed/finance/allDocuments.json")
)

# --- Sanity checks ---
if not ES_HOST or not ES_API_KEY:
    raise SystemExit(
        "Missing ES_HOST or ES_API_KEY.\n"
        "Create a .env (see .env.example) and set:\n"
        "  ES_HOST=https://<your-elastic-host>:443\n"
        "  ES_API_KEY=<your-elastic-api-key>\n"
        f"Current JSON path: {JSON_PATH}"
    )

# Safe-ish logging (don’t print secrets)
host_display = urlsplit(ES_HOST).hostname
print("ES host   :", host_display)
print("Index name:", INDEX)
print("JSON path :", JSON_PATH)


# ----------------------------
# Connect
# ----------------------------
es = Elasticsearch(ES_HOST, api_key=ES_API_KEY)
info = es.info().body
print("Connected to cluster:", info.get("cluster_name", "<unknown>"))

# ----------------------------
# Ensure index + mappings
# ----------------------------
if not es.indices.exists(index=INDEX):
    es.indices.create(index=INDEX, body={
        "mappings": {
            "properties": {
                "title":   {"type": "text"},
                "content": {"type": "text"},
                "metadata":{"type": "object"}
            }
        }
    })
    print(f"✅ Created index: {INDEX}")

# Add / ensure vector field
es.indices.put_mapping(index=INDEX, body={
    "properties": {
        "embedding": {
            "type": "dense_vector",
            "dims": 384,
            "index": True,
            "similarity": "cosine"
        }
    }
})
print("✅ dense_vector field ensured:", INDEX)

# ----------------------------
# Load docs
# ----------------------------
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"Could not find JSON at: {JSON_PATH}")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

if not isinstance(docs, list) or not docs:
    raise SystemExit("No documents found in JSON. Expecting a JSON array of objects.")

print(f"Docs to embed/upsert: {len(docs)}")

# ----------------------------
# Embed & upsert vectors
# ----------------------------
model = SentenceTransformer(EMBED_MODEL)

def batched(it, n=256):
    for i in range(0, len(it), n):
        yield it[i:i+n]

total = 0
for batch in batched(docs, 256):
    texts = [d.get("content", "") for d in batch]
    vecs = model.encode(texts, normalize_embeddings=True)  # cosine-friendly

    actions = []
    for d, v in zip(batch, vecs):
        actions.append({
            "_op_type": "update",
            "_index": INDEX,
            "_id": d["id"],
            "doc": {
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                "metadata": d.get("metadata", {}),
                "embedding": (v.tolist() if hasattr(v, "tolist") else list(map(float, v)))
            },
            "doc_as_upsert": True
        })

    helpers.bulk(es, actions)
    total += len(actions)
    print(f"Upserted vectors: {total}/{len(docs)}")

print("✅ All vectors upserted.")

# ----------------------------
# Hybrid search (BM25 + kNN + simple RRF)
# ----------------------------
def bm25(query, k=20):
    r = es.search(
        index=INDEX,
        query={"match": {"content": query}},
        size=k,
        source_includes=["id", "title", "content", "metadata"]
    )
    return [{"_id": h["_id"], "_source": h["_source"]} for h in r["hits"]["hits"]]

def knn(query, k=20):
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    r = es.search(
        index=INDEX,
        knn={"field": "embedding", "query_vector": qvec, "k": k, "num_candidates": 100},
        size=k,
        source_includes=["id", "title", "content", "metadata"]
    )
    return [{"_id": h["_id"], "_source": h["_source"]} for h in r["hits"]["hits"]]

def rrf_fuse(bm25_hits, knn_hits, k=10, c=60):
    scores = {}
    def add(lst):
        for rank, h in enumerate(lst, 1):
            scores[h["_id"]] = scores.get(h["_id"], 0.0) + 1.0 / (c + rank)
    add(bm25_hits); add(knn_hits)
    by_id = {h["_id"]: h for h in bm25_hits + knn_hits}
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"_id": _id, "_rrf": s, "_source": by_id[_id]["_source"]} for _id, s in fused]

def preview(rows, n=5):
    for i, h in enumerate(rows[:n], 1):
        t = " ".join(h["_source"]["content"].split())[:140]
        print(f"{i}. {t}  (rrf={h.get('_rrf', 0):.4f})")

for q in ["operating profit increased", "net sales growth guidance", "share buyback program"]:
    print(f"\nQ: {q}")
    preview(rrf_fuse(bm25(q), knn(q)))

print("\n✅ Hybrid search ready.")
