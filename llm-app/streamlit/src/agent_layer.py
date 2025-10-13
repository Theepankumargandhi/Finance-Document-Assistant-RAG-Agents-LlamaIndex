# llm-app/streamlit/src/agent_layer.py

import os
import sys
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Make sure we can import local modules in ./src
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)

# Robust .env load (same pattern as search_backend)
dotenv_path = find_dotenv(filename=".env", usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    alt = os.path.abspath(os.path.join(HERE, "../.env"))
    if os.path.exists(alt):
        load_dotenv(alt)

# --- Tools we reuse ---
from search_backend import hybrid
from plot_tools import plot_data  # for Phase 5 charts

# --- Lightweight local LLM (no external keys) ---
_MODEL_NAME = os.getenv("AGENT_MODEL", "google/flan-t5-small")

def _build_text2text_llm(model_name: str = _MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=256,
        temperature=0.0
    )
    return HuggingFacePipeline(pipeline=gen)

# Optional summarizer tool (pure local)
def summarize_text(text: str, max_chars: int = 4000) -> str:
    text = (text or "")[:max_chars]
    if not text.strip():
        return ""
    summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    out = summ(text, max_length=180, min_length=50, do_sample=False)
    return out[0]["summary_text"].strip()

# --- Prompt ---
_PROMPT = PromptTemplate(
    input_variables=["history", "question", "context", "citations"],
    template=(
        "You are a precise finance assistant. Answer the QUESTION using only the CONTEXT. "
        "If the answer isn't in the context, say you don't know and suggest a more specific query.\n\n"
        "QUESTION: {question}\n\n"
        "CONTEXT:\n{context}\n\n"
        "CITATIONS (ids or titles): {citations}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Write your response in 2–5 sentences.\n"
        "Then on a new line, write: Sources: <short list of the citations provided>."
    ),
)

# --- Simple intent for plotting ---
PLOT_KEYWORDS = re.compile(r"\b(plot|chart|graph|trend|visualize|time series)\b", re.I)

def _extract_metric_for_plot(q: str) -> Optional[str]:
    ql = q.lower()
    if "operating profit" in ql or "operating_profit" in ql:
        return "operating_profit"
    if "net income" in ql or "net_income" in ql:
        return "net_income"
    if "revenue" in ql:
        return "revenue"
    return None

def _wants_plot(q: str) -> bool:
    return bool(PLOT_KEYWORDS.search(q))


class FinanceAgent:
    def __init__(self, k: int = 8):
        self.k = k
        self.llm = _build_text2text_llm()
        self.memory = ConversationBufferMemory(memory_key="history", input_key="question")
        self.chain = LLMChain(llm=self.llm, prompt=_PROMPT, memory=self.memory, verbose=False)

    # --- Tools ---
    def search_docs(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        return hybrid(query, k=(k or self.k))

    # --- Main run ---
    def run(self, user_message: str, k: Optional[int] = None) -> Dict[str, Any]:
        # 0) If user asked for a chart, try to plot first
        if _wants_plot(user_message):
            metric = _extract_metric_for_plot(user_message)
            if metric:
                try:
                    fig, _ = plot_data(metric, roll=3)  # small moving average
                    return {
                        "answer": f"Here’s the {metric.replace('_',' ')} trend based on the sample metrics data.",
                        "sources": [],
                        "citations": [f"metrics_sample.csv:{metric}"],
                        "figure": fig
                    }
                except Exception:
                    # fall through to normal retrieval if plotting fails
                    pass

        # 1) Retrieve
        hits = self.search_docs(user_message, k)
        if not hits:
            return {
                "answer": "I couldn't find this in the indexed finance documents. Try a more specific query (company, metric, period).",
                "sources": [],
                "citations": []
            }

        # 2) Build context
        snippets, cite_elems = [], []
        top_score = float(hits[0].get("_score") or 0.0)
        for i, h in enumerate(hits, 1):
            src = h.get("_source", {}) or {}
            title = src.get("title") or f"Financial snippet #{i}"
            did = h.get("_id", f"doc-{i}")
            content = (src.get("content") or "").strip()
            if content:
                snippets.append(f"[{i}] {title} (id={did})\n{content}\n")
                cite_elems.append(f"[{i}] {title} (id={did})")

        context = "\n".join(snippets)[:6000]
        citations = "; ".join(cite_elems[:8]) if cite_elems else "—"

        # 3) Guardrail: low relevance → guide user
        RELEVANCE_THRESHOLD = 3.0
        if (not context.strip()) or top_score < RELEVANCE_THRESHOLD:
            return {
                "answer": (
                    "This question doesn’t seem directly answered by the indexed finance documents. "
                    "Try adding specifics like a company, metric, or time period."
                ),
                "sources": hits,
                "citations": cite_elems
            }

        # 4) LLM answer
        answer = self.chain.run(question=user_message, context=context, citations=citations).strip()

        # 5) Fallback: if too short or only citations, summarize the top passage
        if len(answer) < 20 or (answer.lower().startswith("sources") and "sources:" in answer.lower()):
            top_text = (hits[0].get("_source", {}).get("content") or "")[:1500]
            try:
                summary = summarize_text(top_text) if top_text else ""
            except Exception:
                summary = top_text[:400]
            answer = (summary or "I found related snippets but not a direct answer in context.") + f"\n\nSources: {citations}"

        return {"answer": answer, "sources": hits, "citations": cite_elems}


# Convenience entry points
def init_agent(k: int = 8) -> FinanceAgent:
    return FinanceAgent(k=k)

def run_agent(agent: FinanceAgent, user_message: str, k: Optional[int] = None) -> Dict[str, Any]:
    return agent.run(user_message, k=k)
