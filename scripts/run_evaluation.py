import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "llm-app" / "streamlit" / "src"
sys.path.append(str(SRC_PATH))


from analytics.evaluation import evaluate
from core.search_backend import hybrid

if __name__ == "__main__":
    print("Running evaluation using ground-truth CSV...")
    from pathlib import Path

gt_path = Path(__file__).resolve().parents[1] / "llm-app" / "streamlit" / "src" / "ground_truth" / "ground-truth-data.csv"

metrics = evaluate(
    hybrid,
    gt_path=str(gt_path),
    k=5,
    log_mlflow=True,
    experiment_name="finance-rag",
    run_name="manual-eval",
)
