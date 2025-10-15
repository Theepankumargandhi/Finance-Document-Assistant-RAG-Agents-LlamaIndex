# llm-app/streamlit/src/core/llamaindex_backend.py
from __future__ import annotations

import os
from typing import List, Dict

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# LlamaIndex core
from llama_index.core import Document, Settings, StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

# HuggingFace embeddings (we're not using OpenAI here)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Vector store adapter (Elasticsearch) — handle old/new class names
try:
    # Older releases exposed this name
    from llama_index.vector_stores.elasticsearch import (
        ElasticsearchVectorStore as ElasticsearchStore,
    )
except Exception:
    # Newer releases (e.g., 0.5.x)
    from llama_index.vector_stores.elasticsearch import ElasticsearchStore


class LlamaIndexBackend:
    def __init__(self) -> None:
        # Read env
        load_dotenv()
        self.es_host = os.getenv("ES_HOST", "").strip()
        self.es_api_key = os.getenv("ES_API_KEY", "").strip()
        self.es_vector_index = os.getenv("ES_VECTOR_INDEX", "finance_docs_vector").strip()
        self.text_field = os.getenv("ES_TEXT_FIELD", "text").strip()
        self.embed_model_name = os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))

        if not self.es_host or not self.es_api_key:
            raise RuntimeError("ES_HOST and ES_API_KEY must be set in .env")

        # ES client
        self.es = Elasticsearch(self.es_host, api_key=self.es_api_key)

        # Embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

        # Vector store (note: we alias to ElasticsearchStore above)
        self.vector_store = ElasticsearchStore(
            es_client=self.es,
            index_name=self.es_vector_index,
            text_key=self.text_field,  # where the main text is stored
        )

        # Storage context bound to ES vector store
        self.storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)

        # Simple splitter (you can tune later)
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _to_nodes(self, docs: List[Dict]) -> List[TextNode]:
        nodes: List[TextNode] = []
        for d in docs:
            text = d.get(self.text_field, "") or d.get("content", "")
            if not text:
                continue
            # keep everything else as metadata
            meta = {k: v for k, v in d.items() if k not in {self.text_field, "content"}}
            # split into chunks
            for chunk in self.splitter.split_text(text):
                nodes.append(TextNode(text=chunk, metadata=meta))
        return nodes

    def index_documents(self, docs: List[Dict], batch_size: int = 128) -> None:
        """Embed and upsert into Elasticsearch vector index via LlamaIndex."""
        nodes = self._to_nodes(docs)
        if not nodes:
            raise RuntimeError("No indexable nodes produced from documents")

        # Build or update the index bound to the ES vector store
        _ = VectorStoreIndex.from_documents(
            documents=[Document(text=n.get_content(), metadata=n.metadata) for n in nodes],
            storage_context=self.storage_ctx,
            show_progress=True,
            embed_model=Settings.embed_model,
            # NOTE: VectorStoreIndex will call into the vector_store to upsert
        )
