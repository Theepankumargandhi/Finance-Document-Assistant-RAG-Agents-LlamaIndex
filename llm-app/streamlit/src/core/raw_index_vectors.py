#!/usr/bin/env python3
"""
Raw Vector Indexing - Using pure REST API (no elasticsearch-py)
Run: python src/core/raw_index_vectors.py
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import requests

# Load environment
ROOT = Path(__file__).resolve().parents[4]
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

from openai import OpenAI

# Configuration
ES_HOST = os.getenv("ES_HOST", "").rstrip("/")
ES_API_KEY = os.getenv("ES_API_KEY", "")
ES_VECTOR_INDEX = os.getenv("ES_VECTOR_INDEX", "finance_docs_vector")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

DATA_PATH = ROOT / "data" / "processed" / "finance" / "allDocuments.json"
BULK_SIZE = 100

if not ES_HOST or not ES_API_KEY:
    print("❌ ES_HOST and ES_API_KEY required in .env")
    sys.exit(1)

def chunk_text(text, chunk_size, overlap):
    """Simple text chunking"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        
    return chunks

def main():
    print("=" * 70)
    print("Raw Vector Indexing - Pure REST API")
    print("=" * 70)
    
    # Use ES_HOST directly
    es_url = ES_HOST
    print(f"\n✓ Elasticsearch URL: {es_url}")
    
    headers = {
        "Authorization": f"ApiKey {ES_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Check connection
    print(f"\n🔧 Testing connection...")
    try:
        resp = requests.get(f"{es_url}/", headers=headers, timeout=10)
        if resp.status_code == 200:
            info = resp.json()
            print(f"✓ Connected: {info.get('cluster_name', 'Unknown')}")
        else:
            print(f"❌ Connection failed: {resp.status_code} - {resp.text}")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Check/Delete index
    print(f"\n📦 Checking index: {ES_VECTOR_INDEX}")
    try:
        resp = requests.head(f"{es_url}/{ES_VECTOR_INDEX}", headers=headers, timeout=10)
        
        if resp.status_code == 200:
            print(f"⚠ Index exists")
            response = input("Delete and recreate? (yes/no): ")
            if response.lower() == 'yes':
                resp = requests.delete(f"{es_url}/{ES_VECTOR_INDEX}", headers=headers, timeout=10)
                print("✓ Deleted")
            else:
                print("❌ Aborted")
                return
    except Exception as e:
        print(f"⚠ Error checking index: {e}")
    
    # Create index
    print(f"✓ Creating index...")
    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                },
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "metadata": {"type": "object"}
            }
        }
    }
    
    try:
        resp = requests.put(f"{es_url}/{ES_VECTOR_INDEX}", headers=headers, json=mapping, timeout=30)
        if resp.status_code in [200, 201]:
            print(f"✓ Index created successfully")
        else:
            print(f"❌ Failed to create index: {resp.status_code}")
            print(f"   Response: {resp.text}")
            return
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return
    
    # Load data
    print(f"\n📁 Loading documents...")
    if not DATA_PATH.exists():
        print(f"❌ Data file not found: {DATA_PATH}")
        return
        
    with DATA_PATH.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"✓ Loaded {len(docs)} documents")
    
    # Chunk documents
    print(f"\n📄 Chunking documents...")
    all_chunks = []
    for doc in tqdm(docs, desc="Chunking"):
        text = doc.get("text", "")
        if not text:
            continue
        
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "metadata": doc.get("metadata", {})
            })
    
    print(f"✓ Created {len(all_chunks)} chunks")
    
    # Initialize OpenAI
    print(f"\n🤖 Initializing OpenAI ({EMBED_MODEL})...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Process in batches
    print(f"\n🚀 Indexing {len(all_chunks)} chunks (batch size: {BULK_SIZE})...")
    print(f"   Estimated time: ~{(len(all_chunks) // BULK_SIZE) * 2} minutes\n")
    
    total_indexed = 0
    total_batches = (len(all_chunks) + BULK_SIZE - 1) // BULK_SIZE
    
    for i in tqdm(range(0, len(all_chunks), BULK_SIZE), desc="Progress", total=total_batches):
        batch = all_chunks[i:i + BULK_SIZE]
        texts = [c["text"] for c in batch]
        
        # Get embeddings
        try:
            response = client.embeddings.create(input=texts, model=EMBED_MODEL)
            embeddings = [item.embedding for item in response.data]
        except Exception as e:
            print(f"\n⚠ Embedding error: {e}")
            continue
        
        # Build bulk request (ndjson format)
        bulk_lines = []
        for chunk, embedding in zip(batch, embeddings):
            bulk_lines.append(json.dumps({"index": {"_index": ES_VECTOR_INDEX}}))
            bulk_lines.append(json.dumps({
                "text": chunk["text"],
                "embedding": embedding,
                "doc_id": chunk.get("doc_id", ""),
                "title": chunk.get("title", ""),
                "metadata": chunk.get("metadata", {})
            }))
        
        bulk_data = "\n".join(bulk_lines) + "\n"
        
        # Send bulk request
        try:
            resp = requests.post(
                f"{es_url}/_bulk",
                headers={"Authorization": f"ApiKey {ES_API_KEY}", "Content-Type": "application/x-ndjson"},
                data=bulk_data.encode('utf-8'),
                timeout=60
            )
            
            if resp.status_code == 200:
                result = resp.json()
                if not result.get("errors"):
                    total_indexed += len(batch)
                else:
                    print(f"\n⚠ Some documents failed in batch")
            else:
                print(f"\n⚠ Bulk request failed: {resp.status_code}")
        except Exception as e:
            print(f"\n⚠ Request error: {e}")
    
    # Verify
    print(f"\n✅ Successfully indexed {total_indexed} chunks!")
    
    # Refresh and count
    try:
        requests.post(f"{es_url}/{ES_VECTOR_INDEX}/_refresh", headers=headers, timeout=10)
        resp = requests.get(f"{es_url}/{ES_VECTOR_INDEX}/_count", headers=headers, timeout=10)
        count = resp.json()["count"]
        print(f"✓ Verification: {count} documents in index")
    except Exception as e:
        print(f"⚠ Verification error: {e}")
    
    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print("=" * 70)
    print("\n✅ Vector index ready!")
    print("   Next: streamlit run app.py")

if __name__ == "__main__":
    main()