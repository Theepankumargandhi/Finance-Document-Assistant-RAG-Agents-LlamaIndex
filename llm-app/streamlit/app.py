# llm-app/streamlit/app.py

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------
# Make ./src importable (core, analytics, agents, etc.)
# ---------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

load_dotenv()

# ---------------------------------------------------
# Internal imports
# ---------------------------------------------------
from core.llamaindex_backend import LlamaIndexBackend

from analytics.llm import (
    generate_document_id,
    captureUserInput,
    captureUserFeedback,
)

# Agents (your files live in src/agents/)
try:
    from agents.agent_layer import init_agent, run_agent
    from agents.summarizer_agent import summarize_query
    HAS_AGENTS = True
except:
    HAS_AGENTS = False

# ---------------------------------------------------
# Helpers: handle 'text' or 'content' transparently
# ---------------------------------------------------
def _src_text(src: dict) -> str:
    return (src.get("text") or src.get("content") or "").strip()

def _src_title(src: dict, fallback: str) -> str:
    return (src.get("title") or fallback).strip()

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(page_title="Finance Document Assistant", page_icon="📈", layout="wide")
st.title("📈 Finance Document Assistant")

# ---------------------------------------------------
# Session state
# ---------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "docId" not in st.session_state:
    st.session_state.docId = None
if "userInput" not in st.session_state:
    st.session_state.userInput = ""
if "feedbackSubmitted" not in st.session_state:
    st.session_state.feedbackSubmitted = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ---------------------------------------------------
# Initialize LlamaIndex Backend
# ---------------------------------------------------
@st.cache_resource(show_spinner="Initializing RAG system...")
def get_backend():
    try:
        return LlamaIndexBackend()
    except Exception as e:
        st.error(f"Failed to initialize backend: {e}")
        st.info("Check your .env file has: ES_HOST, ES_API_KEY, OPENAI_API_KEY")
        return None

backend = get_backend()

if backend is None:
    st.stop()

# ---------------------------------------------------
# Sidebar controls (shared)
# ---------------------------------------------------
st.sidebar.header("Search settings")

mode_options = ["Hybrid (RRF)", "BM25", "kNN (vectors)"]
mode = st.sidebar.selectbox("Retrieval mode", mode_options, index=0)
k = st.sidebar.slider("Results to retrieve", 3, 20, 8)

# Map UI mode to backend mode
mode_map = {
    "Hybrid (RRF)": "hybrid",
    "BM25": "bm25",
    "kNN (vectors)": "vector",
}
backend_mode = mode_map[mode]

# Show index/config hints
st.sidebar.caption(
    "BM25: "
    f"{os.getenv('ES_BM25_INDEX', 'finance_docs_bm25')}  •  "
    "Vector: "
    f"{os.getenv('ES_VECTOR_INDEX', 'finance_docs_vector')}  •  "
    f"LLM: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
)

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab_search, tab_agent = st.tabs(["🔎 Search", "💬 Chat (Agent)"])

# ===========================
# Tab 1: Search
# ===========================
with tab_search:
    query_text = st.text_input(
        "Ask a question about the financial documents:",
        placeholder="e.g., What happened to operating profit in Q3?",
    )

    if st.button("Ask"):
        if not query_text.strip():
            st.warning("Please enter a question before clicking **Ask**.")
        else:
            with st.spinner("Generating answer with LLM..."):
                t0 = time.time()

                try:
                    # Build QueryEngine with LLM
                    query_engine = backend.build_query_engine(
                        top_k=k,
                        mode=backend_mode,
                        response_mode="compact"
                    )
                    
                    # Query and get LLM-generated response
                    response = query_engine.query(query_text)
                    
                    answer = response.response.strip() if response.response else "(no answer generated)"
                    source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
                    
                    latency = time.time() - t0
                    
                    # Calculate hit rate and MRR (simple: 1.0 if we got results)
                    hit_rate = 1.0 if source_nodes else 0.0
                    mrr = 1.0 if source_nodes else 0.0
                    
                    # Telemetry (best-effort)
                    try:
                        doc_id = generate_document_id(query_text, answer)
                        captureUserInput(
                            doc_id,
                            query_text.replace("'", ""),
                            answer,
                            1.0,  # confidence placeholder
                            latency,
                            hit_rate,
                            mrr,
                        )
                        st.session_state.docId = doc_id
                    except Exception as e:
                        st.caption(f"Telemetry skipped: {e}")

                    # Save for feedback section
                    st.session_state.result = answer
                    st.session_state.userInput = query_text.replace("'", "")
                    st.session_state.feedbackSubmitted = False

                    # Show answer
                    st.subheader("Answer")
                    st.markdown(f"**{answer}**")
                    st.caption(f"Mode: {mode} • Latency: {latency:.2f}s • Sources: {len(source_nodes)}")

                    # Show source citations
                    if source_nodes:
                        st.subheader("Sources")
                        for i, node_with_score in enumerate(source_nodes, 1):
                            node = node_with_score.node
                            score = node_with_score.score if hasattr(node_with_score, 'score') else 0.0
                            
                            text = node.text if hasattr(node, 'text') else node.get_content()
                            metadata = node.metadata if hasattr(node, 'metadata') else {}
                            title = metadata.get("title") or metadata.get("doc_id") or f"Document {i}"
                            
                            with st.expander(f"{i}. {title} (Score: {score:.3f})"):
                                st.write(text)
                                if metadata:
                                    meta_str = " | ".join(
                                        f"{k}: {v}" for k, v in metadata.items() 
                                        if v and k not in ["title", "text"]
                                    )
                                    if meta_str:
                                        st.caption(meta_str)
                    else:
                        st.warning("No sources found. Try a different search mode or query.")

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.exception(e)
                    st.info("💡 Troubleshooting:")
                    st.info("1. Check OPENAI_API_KEY is set correctly in .env")
                    st.info("2. Verify vector index has documents (run the indexing script)")
                    st.info("3. Try BM25 mode first to test basic retrieval")

    # Feedback
    if st.session_state.result and not st.session_state.feedbackSubmitted:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Satisfied"):
                try:
                    captureUserFeedback(
                        st.session_state.docId,
                        st.session_state.userInput,
                        st.session_state.result,
                        True,
                    )
                    st.success("Thanks for your feedback!")
                finally:
                    st.session_state.feedbackSubmitted = True
        with c2:
            if st.button("❌ Unsatisfied"):
                try:
                    captureUserFeedback(
                        st.session_state.docId,
                        st.session_state.userInput,
                        st.session_state.result,
                        False,
                    )
                    st.warning("Thanks for your feedback!")
                finally:
                    st.session_state.feedbackSubmitted = True

# ===========================
# Tab 2: Chat (Agent)
# ===========================
with tab_agent:
    if not HAS_AGENTS:
        st.warning("Agent functionality not available. Install required dependencies.")
        st.stop()
    
    st.subheader("Chat with Agent (memory on)")

    # Keep one agent instance per session & sync with current k
    if "agent" not in st.session_state:
        st.session_state.agent = init_agent(k=k)
        st.session_state.agent_k = k
    elif st.session_state.get("agent_k") != k:
        st.session_state.agent = init_agent(k=k)
        st.session_state.agent_k = k

    # Simple chat UI
    col_in, col_btn, col_clear = st.columns([6, 1, 1])
    with col_in:
        user_msg = st.text_input(
            "Message",
            placeholder="Ask anything about the finance corpus...",
            key="agent_input",
        )
    with col_btn:
        send = st.button("Send", key="send_agent")
    with col_clear:
        if st.button("Clear"):
            st.session_state.chat_messages = []

    # Handle chat send
    if send and user_msg.strip():
        with st.spinner("Thinking with tools…"):
            out = run_agent(st.session_state.agent, user_msg, k=k)
            fig = out.get("figure")
            if fig is not None:
                st.pyplot(fig)
            st.session_state.chat_messages.append(("user", user_msg))
            st.session_state.chat_messages.append(("assistant", (out.get("answer") or "").strip()))

            cites = out.get("citations") or []
            if cites:
                st.caption("Sources: " + "; ".join(cites[:5]))

            st.markdown("**Top sources**")
            for i, h in enumerate((out.get("sources") or [])[:3], 1):
                src = h.get("_source", {}) or {}
                title = _src_title(src, f"Document {i}")
                content = _src_text(src)
                with st.expander(f"{i}. {title}"):
                    st.write(content)

    # --------- Quick Summarize control ---------
    try:
        summ_q = st.text_input(
            "Or, quickly summarize results for:",
            placeholder="e.g., operating profit in Q3, EuroChem long-term funds, net sales growth",
            key="summ_q",
        )
        if st.button("Summarize", key="summarize_btn"):
            if not summ_q.strip():
                st.warning("Type something to summarize.")
            else:
                with st.spinner("Summarizing top results…"):
                    res = summarize_query(summ_q, k=k)
                    st.success("Summary")
                    st.write(res.get("summary", ""))

                    st.markdown("**Top sources**")
                    for i, h in enumerate((res.get("sources") or [])[:3], 1):
                        src = h.get("_source", {}) or {}
                        title = _src_title(src, f"Document {i}")
                        content = _src_text(src)
                        with st.expander(f"{i}. {title}"):
                            st.write(content)
    except Exception as e:
        st.info(f"Summarizer not available: {e}")

    # Render transcript
    for role, msg in st.session_state.get("chat_messages", []):
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)