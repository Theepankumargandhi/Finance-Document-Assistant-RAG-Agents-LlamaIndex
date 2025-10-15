# test_llamaindex.py (place in llm-app/streamlit/)
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

print("=" * 50)
print("Testing LlamaIndex Backend")
print("=" * 50)

# Test 1: Import
print("\n1. Testing imports...")
try:
    from core.llamaindex_backend import LlamaIndexBackend
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Initialize backend
print("\n2. Initializing backend...")
try:
    backend = LlamaIndexBackend()
    print("✓ Backend initialized")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Check ES connection
print("\n3. Checking Elasticsearch connection...")
try:
    info = backend.es.info()
    print(f"✓ Connected to ES: {info.get('version', {}).get('number', 'unknown')}")
except Exception as e:
    print(f"✗ ES connection failed: {e}")
    exit(1)

# Test 4: Check indices
print("\n4. Checking indices...")
try:
    vector_exists = backend.es.indices.exists(index=os.getenv("ES_VECTOR_INDEX", "finance_docs_vector"))
    bm25_exists = backend.es.indices.exists(index=os.getenv("ES_BM25_INDEX", "finance_docs_bm25"))
    print(f"  Vector index exists: {vector_exists}")
    print(f"  BM25 index exists: {bm25_exists}")
except Exception as e:
    print(f"✗ Index check failed: {e}")

# Test 5: Test BM25 query
print("\n5. Testing BM25 query...")
try:
    results = backend.query("test", top_k=3, mode="bm25")
    print(f"✓ BM25 returned {len(results)} results")
except Exception as e:
    print(f"✗ BM25 query failed: {e}")

# Test 6: Test Vector query
print("\n6. Testing Vector query...")
try:
    results = backend.query("test", top_k=3, mode="vector")
    print(f"✓ Vector returned {len(results)} results")
except Exception as e:
    print(f"✗ Vector query failed: {e}")

# Test 7: Test Hybrid query
print("\n7. Testing Hybrid query...")
try:
    results = backend.query("test", top_k=3, mode="hybrid")
    print(f"✓ Hybrid returned {len(results)} results")
    if results:
        print(f"  Sample result score: {results[0].score}")
except Exception as e:
    print(f"✗ Hybrid query failed: {e}")

print("\n" + "=" * 50)
print("Testing complete!")
print("=" * 50)