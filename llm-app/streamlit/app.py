# llm-app/streamlit/app.py

import os
import sys
import time
import streamlit as st
from transformers import pipeline
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
# Internal imports (NOTE: no "src." prefix here)
# ---------------------------------------------------
from src.core.search_backend import bm25, knn, hybrid

# Optional reranker (safe import)
hybrid_rerank = None
HAS_RERANK = False
try:
    from src.core.search_backend import hybrid_rerank as _hybrid_rerank
    hybrid_rerank = _hybrid_rerank
    HAS_RERANK = True
except Exception:
    pass

from src.analytics.llm import (
    generate_document_id,
    captureUserInput,
    captureUserFeedback,
)

# Agents (your files live in src/agents/)
from src.agents.agent_layer import init_agent, run_agent
from src.agents.summarizer_agent import summarize_query

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
# QA pipeline (extractive, local)
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_qa():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa = get_qa()

# ---------------------------------------------------
# Sidebar controls (shared)
# ---------------------------------------------------
st.sidebar.header("Search settings")

mode_options = ["Hybrid (RRF)", "BM25", "kNN (vectors)"]
if HAS_RERANK and callable(hybrid_rerank):
    mode_options.insert(1, "Hybrid + Rerank")

mode = st.sidebar.selectbox("Retrieval mode", mode_options, index=0)
k = st.sidebar.slider("Results to retrieve", 3, 20, 8)
max_ctx_chars = st.sidebar.slider("Max context chars (concat top-k)", 500, 8000, 3000, step=250)

st.sidebar.caption(
    f"Index: {os.getenv('ES_INDEX', 'finance_docs')}  •  Host set: {'✅' if os.getenv('ES_HOST') else '❌'}"
)

# Map retrieval functions
retrievers = {
    "Hybrid (RRF)": hybrid,
    "BM25": bm25,
    "kNN (vectors)": knn,
}
if HAS_RERANK and callable(hybrid_rerank):
    retrievers["Hybrid + Rerank"] = hybrid_rerank

search_fn = retrievers[mode]

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
            with st.spinner("Searching and extracting an answer…"):
                t0 = time.time()

                # 1) Retrieve
                hits = search_fn(query_text, k=k)

                # 2) Build context from top-k (concat)
                snippets = []
                for h in hits:
                    src = h.get("_source", {}) or {}
                    content = (src.get("content") or "").strip()
                    if content:
                        snippets.append(content)
                context = "\n\n".join(snippets)[:max_ctx_chars] if snippets else ""

                # 3) Extract answer
                ans = qa({"question": query_text, "context": context}) if context else {"answer": "", "score": 0.0}
                answer = (ans.get("answer") or "").strip()
                score = float(ans.get("score") or 0.0)
                latency = time.time() - t0

                # 4) Telemetry (best-effort)
                try:
                    doc_id = generate_document_id(query_text, answer)
                    captureUserInput(
                        doc_id,
                        query_text.replace("'", ""),
                        answer,
                        score,
                        latency,
                        1.0 if hits else 0.0,  # UI placeholders
                        1.0 if hits else 0.0,
                    )
                    st.session_state.docId = doc_id
                except Exception as e:
                    st.caption(f"Telemetry skipped: {e}")

                # 5) Save for feedback section
                st.session_state.result = answer or "(no exact span found—try Hybrid and increase k)"
                st.session_state.userInput = query_text.replace("'", "")
                st.session_state.feedbackSubmitted = False

            # 6) Show answer & sources
            st.subheader("Answer")

            best_source = None
            best_text = ""
            if "hits" in locals() and hits:
                s0 = hits[0].get("_source", {}) or {}
                best_source = s0.get("title") or hits[0].get("_id", "Document 1")
                best_text = (s0.get("content") or "").strip()

            st.markdown(f"**{st.session_state.result}**")
            st.caption(f"Confidence: {score:.3f} • Mode: {mode} • Latency: {latency:.2f}s")
            if best_source:
                st.caption(f"Source: {best_source}")

            if best_text and answer and (answer in best_text):
                idx = best_text.find(answer)
                pre = best_text[max(0, idx - 180): idx]
                post = best_text[idx + len(answer): idx + len(answer) + 180]
                st.markdown("**Highlighted passage**")
                st.write(pre + "**" + answer + "**" + post)

            st.subheader("Top results")
            if "hits" in locals():
                for i, h in enumerate(hits, 1):
                    src = h.get("_source", {}) or {}
                    title = src.get("title") or f"Document {i}"
                    content = src.get("content") or ""
                    meta = src.get("metadata") or {}
                    with st.expander(f"{i}. {title}"):
                        st.write(content)
                        if meta:
                            st.caption(" | ".join(f"{mk}: {mv}" for mk, mv in meta.items() if mv))

    # Feedback
    if st.session_state.result and not st.session_state.feedbackSubmitted:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Satisfied"):
                try:
                    captureUserFeedback(
                        st.session_state.docId,
                        st.session_state.userInput,
                        st.session_state.result,
                        True,
                    )
                finally:
                    st.session_state.feedbackSubmitted = True
        with c2:
            if st.button("Unsatisfied"):
                try:
                    captureUserFeedback(
                        st.session_state.docId,
                        st.session_state.userInput,
                        st.session_state.result,
                        False,
                    )
                finally:
                    st.session_state.feedbackSubmitted = True

# ===========================
# Tab 2: Chat (Agent)
# ===========================
with tab_agent:
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
                title = src.get("title") or f"Document {i}"
                content = src.get("content") or ""
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
                        title = src.get("title") or f"Document {i}"
                        content = src.get("content") or ""
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