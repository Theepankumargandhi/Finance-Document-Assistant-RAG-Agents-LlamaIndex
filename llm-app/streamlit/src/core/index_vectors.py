#!/usr/bin/env python3
"""
Phase 2: Index documents with embeddings into Elasticsearch vector store.
Run this from repo root: python -m llm-app.streamlit.src.core.index_vectors
Or: cd llm-app/streamlit && python -m src.core.index_vectors
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment
ROOT = Path(__file__).resolve().parents[4]  # repo root
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"✓ Loaded .env from: {ENV_PATH}")
else:
    print(f"⚠ .env not found at: {ENV_PATH}")
    load_dotenv()

from llamaindex_backend import LlamaIndexBackend

# Configuration
DATA_PATH = ROOT / "data" / "processed" / "finance" / "allDocuments.json"
BATCH_SIZE = 32  # Process in batches to avoid memory issues

def main():
    print("=" * 60)
    print("Phase 2: Vector Indexing with LlamaIndex")
    print("=" * 60)
    
    # Check data file exists
    if not DATA_PATH.exists():
        print(f"❌ Data file not found: {DATA_PATH}")
        print("   Make sure allDocuments.json exists in data/processed/finance/")
        sys.exit(1)
    
    # Load documents
    print(f"\n📁 Loading documents from: {DATA_PATH.name}")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    
    if not isinstance(docs, list) or not docs:
        print("❌ Invalid data format. Expected non-empty JSON array.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(docs)} documents")
    
    # Show sample
    if docs:
        sample = docs[0]
        print(f"\n📄 Sample document structure:")
        print(f"   - doc_id: {sample.get('doc_id', 'N/A')}")
        print(f"   - title: {sample.get('title', 'N/A')}")
        print(f"   - text length: {len(sample.get('text', ''))} chars")
        print(f"   - metadata keys: {list(sample.get('metadata', {}).keys())}")
    
    # Initialize backend
    print(f"\n🔧 Initializing LlamaIndex backend...")
    try:
        backend = LlamaIndexBackend()
        print(f"✓ Backend initialized")
        print(f"   - Vector index: {os.getenv('ES_VECTOR_INDEX', 'finance_docs_vector')}")
        print(f"   - Embedding model: {os.getenv('EMBED_MODEL', 'text-embedding-3-small')}")
        print(f"   - Chunk size: {os.getenv('CHUNK_SIZE', '800')}")
    except Exception as e:
        print(f"❌ Failed to initialize backend: {e}")
        sys.exit(1)
    
    # Index documents
    print(f"\n🚀 Starting vector indexing (batch size: {BATCH_SIZE})...")
    print("   This may take several minutes depending on document count...")
    
    try:
        backend.index_documents(docs, batch_size=BATCH_SIZE)
        print(f"\n✅ Successfully indexed {len(docs)} documents into vector store!")
        
    except Exception as e:
        print(f"\n❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify indexing
    print(f"\n🔍 Verifying vector search...")
    try:
        test_query = "operating profit"
        results = backend.query(test_query, top_k=3, mode="vector")
        
        if results:
            print(f"✓ Vector search working! Found {len(results)} results for '{test_query}'")
            print(f"   Top result: {results[0].node.text[:100]}...")
        else:
            print(f"⚠ Vector search returned no results. Index may be empty.")
            
    except Exception as e:
        print(f"⚠ Verification failed: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)
    print("\n✅ Next steps:")
    print("   1. Test vector search in Streamlit app")
    print("   2. Try hybrid mode for best results")
    print("   3. Run: streamlit run app.py")
    print()

if __name__ == "__main__":
    main()