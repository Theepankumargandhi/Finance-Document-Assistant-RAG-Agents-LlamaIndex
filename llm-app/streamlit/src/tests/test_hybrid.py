# test_hybrid.py
import os
from search_backend import hybrid
from dotenv import load_dotenv
load_dotenv()
# make sure the index name is set (env or fallback)
os.environ.setdefault("ES_INDEX", "finance_docs")

def short(s, n=120):
    return " ".join(s.split())[:n]

for q in ["operating profit increased", "net sales growth"]:
    print("\nQ:", q)
    hits = hybrid(q, k=5)
    for i, h in enumerate(hits, 1):
        txt = short(h["_source"]["content"])
        print(f"{i}. {txt}  (rrf={h.get('_rrf',0):.4f})")
