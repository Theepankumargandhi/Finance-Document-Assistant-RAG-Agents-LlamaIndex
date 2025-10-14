# llm-app/streamlit/src/plot_tools.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# Resolve CSV path relative to this file
HERE = os.path.dirname(__file__)
CSV_PATH = os.path.abspath(os.path.join(HERE, "..", "..", "data", "processed", "finance", "metrics_sample.csv"))

ALLOWED = {"revenue", "operating_profit", "net_income"}

def plot_data(metric: str, roll: int = 1) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot a time series for the given metric with an optional moving average.
    Returns (figure, dataframe).
    """
    metric = metric.strip().lower()
    if metric not in ALLOWED:
        raise ValueError(f"Unsupported metric '{metric}'. Choose one of: {sorted(ALLOWED)}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Create moving average column if requested
    if roll and roll > 1:
        df[f"{metric}_ma{roll}"] = df[metric].rolling(window=roll, min_periods=1).mean()

    # Make the plot
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.gca()
    ax.plot(df["date"], df[metric], label=metric.replace("_", " ").title())

    if roll and roll > 1:
        ax.plot(df["date"], df[f"{metric}_ma{roll}"], label=f"MA({roll})")

    ax.set_title(f"{metric.replace('_',' ').title()} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.legend()
    fig.autofmt_xdate()  # rotates dates for readability

    return fig, df


if __name__ == "__main__":
    # Quick manual test: saves a PNG next to the CSV
    try:
        fig, _ = plot_data("revenue", roll=3)
        out_dir = os.path.abspath(os.path.join(HERE, "..", "..", "images"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "revenue_example.png")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved example chart to {out_path}")
    except Exception as e:
        print("Error:", e)
