"""
Extracts financial/legal documents from Postgres and saves them to JSON
for ingestion into Elasticsearch (BM25 + vector via LlamaIndex).

Output: ground_truth/allDocuments.json
"""
import json
from pathlib import Path
from src.core.connection import postgre_connection

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "ground_truth" / "allDocuments.json"


def get_data():
    """
    Fetch all rows from 'legal_document' table and return as list[dict].
    Each row becomes a dictionary; field names come from the DB cursor.
    """
    conn, cur = postgre_connection()

    query = "SELECT * FROM legal_document;"
    cur.execute(query)
    columns = [desc[0] for desc in cur.description]  # column names
    results = [dict(zip(columns, row)) for row in cur.fetchall()]

    cur.close()
    conn.close()
    return results


def normalize_records(rows):
    """
    Map DB columns to RAG-friendly structure expected by LlamaIndex/Elastic.
    Adjust keys here if your table uses different field names.
    """
    docs = []
    for row in rows:
        text = row.get("content") or row.get("text") or row.get("body") or ""
        doc_id = row.get("id") or row.get("doc_id")
        title = row.get("title") or row.get("heading") or ""

        metadata = {k: v for k, v in row.items() if k not in {"id", "content", "text", "body", "title"}}

        docs.append({
            "doc_id": doc_id,
            "title": title,
            "text": text,           # <-- normalized key used by LlamaIndex backend
            "metadata": metadata,
        })
    return docs


def main():
    print("🔍 Fetching data from Postgres...")
    raw = get_data()
    print(f"Total rows fetched: {len(raw)}")

    print("📄 Normalizing records for LlamaIndex ingestion...")
    final_data = normalize_records(raw)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    print(f"✅ Export complete → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
