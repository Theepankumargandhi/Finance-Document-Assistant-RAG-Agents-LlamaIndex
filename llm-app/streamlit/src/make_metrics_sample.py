# llm-app/streamlit/src/make_metrics_sample.py
import os
import csv
from datetime import date, timedelta
import random

# Output path
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "finance"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "metrics_sample.csv")

# generate ~24 monthly points
rows = []
start = date(2023, 1, 1)
months = 24

# simple synthetic trend with light noise
rev = 100.0
op = 20.0
ni = 12.0

for i in range(months):
    d = (start.replace(day=1) + timedelta(days=31*i)).replace(day=1)  # month steps
    # drift + noise
    rev *= (1.01 + random.uniform(-0.005, 0.008))
    op  *= (1.012 + random.uniform(-0.006, 0.01))
    ni  *= (1.008 + random.uniform(-0.007, 0.012))

    rows.append({
        "date": d.isoformat(),
        "revenue": round(rev, 2),
        "operating_profit": round(op, 2),
        "net_income": round(ni, 2)
    })

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date","revenue","operating_profit","net_income"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT_FILE} ({len(rows)} rows)")
