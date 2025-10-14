#llm.py
import requests
import hashlib
import time
from datetime import datetime, timezone
from core.connection import mongodb_connection, postgre_connection
import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
def generate_document_id(userQuery, answer):
    combined = f"{userQuery[:10]}-{answer[:10]}"
    hash_object = hashlib.md5(combined.encode())
    return hash_object.hexdigest()[:8]
def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_KEY')}"}
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=payload)
    latency = (time.time() - start_time) * 1000  # ms
    return response.json(), round(latency, 2)
def captureUserInput(docId, userQuery, result, llmScore, responseTime, hit_rate, mrr):
    conn, cur = postgre_connection()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_data (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(10) NOT NULL,
                user_input TEXT NOT NULL,
                result TEXT NOT NULL,
                llm_score DOUBLE PRECISION NOT NULL,
                response_time DOUBLE PRECISION NOT NULL,
                hit_rate_score DOUBLE PRECISION NOT NULL,
                mrr_score DOUBLE PRECISION NOT NULL,
                latency_ms DOUBLE PRECISION,
                model_name TEXT,
                created_time TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        conn.commit()
    except Exception as e:
        print("[DB] Table create failed:", e)
        conn.rollback()

    try:
        cur.execute(
            """
            INSERT INTO evaluation_data
            (doc_id, user_input, result, llm_score, response_time,
             hit_rate_score, mrr_score, latency_ms, model_name)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                docId,
                userQuery,
                result,
                llmScore,
                responseTime,
                hit_rate,
                mrr,
                float(responseTime) * 1000.0,  # convert to ms
                os.getenv("MODEL_NAME", "bert-large-uncased-squad"),
            ),
        )
        conn.commit()
    except Exception as e:
        print("[DB] Insert failed:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return "✅ Evaluation data logged"


# -------------------------------------------------
# User feedback logging → PostgreSQL  (Grafana reads from this)
# -------------------------------------------------
def captureUserFeedback(docId, userQuery, result, feedback):
    conn, cur = postgre_connection()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_data (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(10) NOT NULL,
                user_input TEXT NOT NULL,
                result TEXT NOT NULL,
                is_satisfied BOOLEAN NOT NULL,
                created_time TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        conn.commit()
    except Exception as e:
        print("[DB] Table create failed:", e)
        conn.rollback()

    try:
        cur.execute(
            """
            INSERT INTO feedback_data
            (doc_id, user_input, result, is_satisfied)
            VALUES (%s,%s,%s,%s)
            """,
            (docId, userQuery, result, feedback),
        )
        conn.commit()
    except Exception as e:
        print("[DB] Insert failed:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return "✅ Feedback data logged"
