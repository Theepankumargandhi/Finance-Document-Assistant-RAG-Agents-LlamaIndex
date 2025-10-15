#connection.py
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

def mongodb_connection(db_name, collection_name):
    """
    Connects to MongoDB (used for logs, telemetry, or caching).
    """
    client = MongoClient('mongodb://admin:admin@mongodb:27017/')
    db = client[db_name]
    collection = db[collection_name]
    return db, collection


def postgre_connection():
    """
    Connects to Postgres (used by Airflow or analytics tables).
    """
    conn = psycopg2.connect(
        dbname="airflow",
        user="airflow",
        password="airflow",
        host="postgres",
        port="5432",
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    return conn, cur


def elastic_connection():
    """
    Shared Elasticsearch client builder.
    Reads config from .env (used by llamaindex_backend or legacy scripts).
    """
    es_host = os.getenv("ES_HOST")
    es_cloud_id = os.getenv("ES_CLOUD_ID", "")
    es_username = os.getenv("ES_USERNAME", "")
    es_password = os.getenv("ES_PASSWORD", "")
    es_api_key = os.getenv("ES_API_KEY", "")

    if es_cloud_id and es_api_key:
        return Elasticsearch(cloud_id=es_cloud_id, api_key=es_api_key)
    if es_host and es_api_key:
        return Elasticsearch(es_host, api_key=es_api_key)
    if es_host and es_username and es_password:
        return Elasticsearch(es_host, basic_auth=(es_username, es_password))
    if es_host:
        return Elasticsearch(es_host)
    raise RuntimeError("Elasticsearch connection not configured properly.")
