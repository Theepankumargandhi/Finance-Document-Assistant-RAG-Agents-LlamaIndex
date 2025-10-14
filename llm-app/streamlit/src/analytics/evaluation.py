#evaluation.py
from __future__ import annotations
import time
import json
import tempfile
from typing import Callable, Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timezone
from core.connection import postgre_connection  
import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


def hit_rate(relevance_total: List[List[bool]]) -> float:
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total) if relevance_total else 0.0


def mrr(relevance_total: List[List[bool]]) -> float:
    score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] is True:
                score += 1.0 / (rank + 1)
                break
    return score / len(relevance_total) if relevance_total else 0.0
def _default_gt_path() -> str:
    here = os.path.dirname(__file__)
    cand1 = os.path.join(here, "ground_truth", "ground-truth-data.csv")
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join("streamlit", "src", "ground_truth", "ground-truth-data.csv")
    return cand2


def _extract_query_text(q: Dict[str, Any]) -> str:
    for key in ("query", "question", "text"):
        if key in q and isinstance(q[key], str):
            return q[key]
    return str(q)[:200]


# ---------- Main evaluation with MLflow + Grafana PostgreSQL logging ----------
def evaluate(
    search_function: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    gt_path: Optional[str] = None,
    k: int = 5,
    log_mlflow: bool = True,
    experiment_name: str = "finance-rag",
    run_name: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import mlflow  # keep local import

    gt_csv = gt_path or _default_gt_path()
    if not os.path.exists(gt_csv):
        raise FileNotFoundError(f"Ground truth CSV not found at: {gt_csv}")

    gt_df = pd.read_csv(gt_csv)
    gt_records = gt_df.to_dict(orient="records")

    relevance_total: List[List[bool]] = []
    latencies_ms: List[float] = []
    per_query_rows: List[Dict[str, Any]] = []

    for q in gt_records:
        doc_id = q.get("document") or q.get("doc_id") or q.get("answer_id")
        if doc_id is None:
            raise KeyError("Ground-truth row missing 'document' field.")

        t0 = time.time()
        results = search_function(q) or []
        latencies_ms.append((time.time() - t0) * 1000.0)

        rel = [str(d.get("doc_id")) == str(doc_id) for d in results]
        relevance_total.append(rel)

        top_ids = [str(d.get("doc_id")) for d in results]
        per_query_rows.append({
            "query": _extract_query_text(q),
            "gt_doc_id": str(doc_id),
            "top_doc_ids": json.dumps(top_ids, ensure_ascii=False),
            "hit@k": any(rel),
            "first_hit_rank": (rel.index(True) + 1) if any(rel) else None,
            "latency_ms": latencies_ms[-1],
        })

    metrics = {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "n_queries": len(gt_records),
        "k": k,
    }

    # ---------- Log to MLflow ----------
    if log_mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("k", k)
            if extra_params:
                for pk, pv in extra_params.items():
                    mlflow.log_param(str(pk), str(pv))
            mlflow.log_metric("hit_rate", metrics["hit_rate"])
            mlflow.log_metric("mrr", metrics["mrr"])
            mlflow.log_metric("avg_latency_ms", metrics["avg_latency_ms"])
            mlflow.log_metric("n_queries", metrics["n_queries"])

            with tempfile.TemporaryDirectory() as td:
                per_query_path = os.path.join(td, "per_query_results.csv")
                pd.DataFrame(per_query_rows).to_csv(per_query_path, index=False)
                mlflow.log_artifact(per_query_path, artifact_path="evaluation")

                rel_json_path = os.path.join(td, "relevance_total.json")
                with open(rel_json_path, "w", encoding="utf-8") as f:
                    json.dump(relevance_total, f, ensure_ascii=False)
                mlflow.log_artifact(rel_json_path, artifact_path="evaluation")

    # ---------- Log metrics to PostgreSQL (for Grafana) ----------
    try:
        conn, cur = postgre_connection()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_eval_metrics (
                id SERIAL PRIMARY KEY,
                hit_rate DOUBLE PRECISION,
                mrr DOUBLE PRECISION,
                avg_latency_ms DOUBLE PRECISION,
                n_queries INTEGER,
                top_k INTEGER,
                experiment_name TEXT,
                run_name TEXT,
                created_time TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            INSERT INTO model_eval_metrics
            (hit_rate, mrr, avg_latency_ms, n_queries, top_k, experiment_name, run_name)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                metrics["hit_rate"],
                metrics["mrr"],
                metrics["avg_latency_ms"],
                metrics["n_queries"],
                metrics["k"],
                experiment_name,
                run_name or "default",
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Logged evaluation metrics to PostgreSQL for Grafana.")
    except Exception as e:
        print("⚠️ PostgreSQL logging failed:", e)

    return metrics
