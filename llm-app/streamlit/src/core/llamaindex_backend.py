# llm-app/streamlit/src/core/llamaindex_backend.py
from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from repo root
ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"✓ Loaded .env from: {ENV_PATH}")
else:
    print(f"⚠ .env not found at: {ENV_PATH}")
    load_dotenv()  # fallback to default search

import json
from typing import Any, Dict, Iterable, List, Optional
from elasticsearch import Elasticsearch

# LlamaIndex core
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# Vector store adapter (Elasticsearch) — handle old/new class names
try:
    from llama_index.vector_stores.elasticsearch import (
        ElasticsearchVectorStore as ElasticsearchStore,
    )
except Exception:
    from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# Embeddings
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except Exception:
    OpenAIEmbedding = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception:
    HuggingFaceEmbedding = None

# LLM imports
try:
    from llama_index.llms.openai import OpenAI
except Exception:
    OpenAI = None

try:
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
except Exception:
    HuggingFaceInferenceAPI = None

# Environment / configuration
ES_HOST: str = os.getenv("ES_HOST", "")
ES_CLOUD_ID: str = os.getenv("ES_CLOUD_ID", "")
ES_USERNAME: str = os.getenv("ES_USERNAME", "")
ES_PASSWORD: str = os.getenv("ES_PASSWORD", "")
ES_API_KEY: str = os.getenv("ES_API_KEY", "")

ES_VECTOR_INDEX: str = os.getenv("ES_VECTOR_INDEX", "finance_docs_vector")
ES_BM25_INDEX: str = os.getenv("ES_BM25_INDEX", "finance_docs_bm25")
ES_TEXT_FIELD: str = os.getenv("ES_TEXT_FIELD", "text")
ES_ID_FIELD: str = os.getenv("ES_ID_FIELD", "doc_id")

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
HF_LLM_MODEL: str = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

DEFAULT_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Utilities
def _get_es_client() -> Elasticsearch:
    if ES_CLOUD_ID and ES_API_KEY:
        return Elasticsearch(
            cloud_id=ES_CLOUD_ID,
            api_key=ES_API_KEY,
        )
    if ES_HOST and ES_API_KEY:
        return Elasticsearch(
            ES_HOST,
            api_key=ES_API_KEY,
        )
    if ES_HOST and ES_USERNAME and ES_PASSWORD:
        return Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USERNAME, ES_PASSWORD),
        )
    if ES_HOST:
        return Elasticsearch(ES_HOST)
    raise RuntimeError("Elasticsearch configuration missing. Set ES_HOST or ES_CLOUD_ID + ES_API_KEY.")


def _get_embeddings():
    if LLM_PROVIDER.lower() == "openai" and OPENAI_API_KEY and OpenAIEmbedding is not None:
        return OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    
    if HuggingFaceEmbedding is None:
        raise RuntimeError(
            "No embedding backend available. Install/enable either OpenAI (OPENAI_API_KEY) "
            "or sentence-transformers (HuggingFaceEmbedding)."
        )
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)


def _get_llm():
    """Get LLM instance based on provider configuration."""
    if LLM_PROVIDER.lower() == "openai" and OPENAI_API_KEY and OpenAI is not None:
        print(f"✓ Using OpenAI LLM: {OPENAI_MODEL}")
        return OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.1)
    
    if HuggingFaceInferenceAPI is not None and HUGGINGFACE_API_KEY:
        print(f"✓ Using HuggingFace LLM: {HF_LLM_MODEL}")
        return HuggingFaceInferenceAPI(
            model_name=HF_LLM_MODEL,
            token=HUGGINGFACE_API_KEY
        )
    
    # Fallback warning
    print("⚠ No LLM configured. QueryEngine will fail without an LLM.")
    print("  Set OPENAI_API_KEY or HUGGINGFACE_API_KEY in .env")
    return None


def _configure_llamaindex_globals():
    Settings.embed_model = _get_embeddings()
    Settings.llm = _get_llm()
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# BM25 Retriever (Elasticsearch) — wrapper
class ElasticsearchBM25Retriever(BaseRetriever):
    def __init__(
        self,
        es: Elasticsearch,
        index: str,
        text_field: str = ES_TEXT_FIELD,
        id_field: str = ES_ID_FIELD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        super().__init__()
        self._es = es
        self._index = index
        self._text_field = text_field
        self._id_field = id_field
        self._top_k = top_k

    def _hit_to_node(self, hit: Dict[str, Any]) -> NodeWithScore:
        src = hit.get("_source", {})
        text = src.get(self._text_field, "") or src.get("text", "") or ""
    
        if not text or not text.strip():
            return None
        
        metadata = {k: v for k, v in src.items() if k not in [self._text_field, "text", "embedding"]}
        node_id = src.get(self._id_field) or hit.get("_id") or ""
    
        node = TextNode(text=text, id_=str(node_id), metadata=metadata)
        return NodeWithScore(node=node, score=float(hit.get("_score", 0.0)))

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        body = {
            "query": {"match": {self._text_field: {"query": query}}},
        }
    
        try:
            resp = self._es.search(index=self._index, size=self._top_k, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            nodes = []
            for h in hits:
                node = self._hit_to_node(h)
                if node is not None:
                    nodes.append(node)
            return nodes
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)


# Custom Hybrid Retriever without LLM dependency
class SimpleHybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = 60,
    ) -> None:
        super().__init__()
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._top_k = top_k
        self._rrf_k = rrf_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get results from both retrievers
        try:
            vector_results = self._vector_retriever.retrieve(query_bundle.query_str)
        except Exception as e:
            print(f"Vector retrieval error: {e}")
            vector_results = []
        
        try:
            bm25_results = self._bm25_retriever.retrieve(query_bundle.query_str)
        except Exception as e:
            print(f"BM25 retrieval error: {e}")
            bm25_results = []
        
        # If both failed, return empty
        if not vector_results and not bm25_results:
            return []
        
        # RRF fusion
        scores = {}
        all_nodes = {}
        
        for rank, node_with_score in enumerate(vector_results, 1):
            node_id = node_with_score.node.node_id
            all_nodes[node_id] = node_with_score.node
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (self._rrf_k + rank)
        
        for rank, node_with_score in enumerate(bm25_results, 1):
            node_id = node_with_score.node.node_id
            all_nodes[node_id] = node_with_score.node
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (self._rrf_k + rank)
        
        # Sort by RRF score and take top_k
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self._top_k]
        
        return [NodeWithScore(node=all_nodes[node_id], score=score) for node_id, score in sorted_nodes]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)


# Vector Index (Elasticsearch vector store)
def _get_vector_index(es: Elasticsearch) -> VectorStoreIndex:
    # Use cloud_id if available, otherwise es_url
    if ES_CLOUD_ID:
        vector_store = ElasticsearchStore(
            index_name=ES_VECTOR_INDEX,
            es_cloud_id=ES_CLOUD_ID,
            es_api_key=ES_API_KEY,
        )
    else:
        vector_store = ElasticsearchStore(
            index_name=ES_VECTOR_INDEX,
            es_url=ES_HOST,
            es_api_key=ES_API_KEY,
        )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents([], storage_context=storage_context)


# Public API
class LlamaIndexBackend:
    def __init__(self) -> None:
        _configure_llamaindex_globals()
        self.es = _get_es_client()

        # Check if BM25 index exists (use elasticSearch.py helper)
        try:
            from .elasticSearch import ensure_bm25_index
            ensure_bm25_index(self.es, ES_BM25_INDEX)
        except Exception as e:
            print(f"BM25 index setup: {e}")

        # Initialize vector index and BM25 retriever
        self.vector_index = _get_vector_index(self.es)
        self.bm25_retriever = ElasticsearchBM25Retriever(
            es=self.es,
            index=ES_BM25_INDEX,
            text_field=ES_TEXT_FIELD,
            id_field=ES_ID_FIELD,
            top_k=DEFAULT_TOP_K,
        )


    def index_documents(
        self,
        docs: Iterable[Dict[str, Any]] | Iterable[Document] | str,
        batch_size: int = 64,
    ) -> None:
        if isinstance(docs, str):
            with open(docs, "r", encoding="utf-8") as f:
                raw = json.load(f)
            doc_dicts = list(raw)
        else:
            docs_list = list(docs)
            if docs_list and isinstance(docs_list[0], Document):
                self._index_documents_as_documents(docs_list, batch_size=batch_size)
                return
            doc_dicts = docs_list

        li_docs: List[Document] = []
        for d in doc_dicts:
            text = d.get(ES_TEXT_FIELD, "")
            meta = {k: v for k, v in d.items() if k != ES_TEXT_FIELD}
            li_docs.append(Document(text=text, metadata=meta))

        self._index_documents_as_documents(li_docs, batch_size=batch_size)

    def _index_documents_as_documents(
        self,
        documents: Iterable[Document],
        batch_size: int = 64,
    ) -> None:
        documents_list = list(documents)
        total = len(documents_list)
        print(f"Indexing {total} documents in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch = documents_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
            
            for doc in batch:
                self.vector_index.insert(doc)
            
        print(f"✓ Indexed {total} documents successfully")

    def _get_vector_retriever(self, top_k: int) -> BaseRetriever:
        return self.vector_index.as_retriever(similarity_top_k=top_k)

    def _get_hybrid_retriever(self, top_k: int) -> BaseRetriever:
        vec = self._get_vector_retriever(top_k=top_k)
        bm25 = self.bm25_retriever
        return SimpleHybridRetriever(
            vector_retriever=vec,
            bm25_retriever=bm25,
            top_k=top_k,
            rrf_k=60,
        )

    def query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        mode: str = "hybrid",
    ) -> List[NodeWithScore]:
        mode = mode.lower()
        if mode == "vector":
            retriever = self._get_vector_retriever(top_k)
        elif mode == "bm25":
            retriever = self.bm25_retriever
        else:
            retriever = self._get_hybrid_retriever(top_k)
        return retriever.retrieve(query_text)

    def build_query_engine(
        self,
        top_k: int = DEFAULT_TOP_K,
        mode: str = "hybrid",
        response_mode: str = "compact",
    ) -> RetrieverQueryEngine:
        mode = mode.lower()
        if mode == "vector":
            retriever = self._get_vector_retriever(top_k)
        elif mode == "bm25":
            retriever = self.bm25_retriever
        else:
            retriever = self._get_hybrid_retriever(top_k)

        synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            use_async=False,
            streaming=False
        )
        return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)