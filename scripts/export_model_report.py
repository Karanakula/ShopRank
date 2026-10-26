"""Write the Phase 8 model comparison from MLflow to reports/model_comparison.csv for the dashboard.

    python scripts/export_model_report.py
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = {
    "params.model_name": "model",
    "metrics.cv_ap": "cv_ap",
    "metrics.cv_ndcg_at_10": "cv_ndcg@10",
    "metrics.train_ap": "train_ap",
    "metrics.test_roc_auc": "test_roc_auc",
    "metrics.test_pr_auc": "test_pr_auc",
    "metrics.test_log_loss": "test_log_loss",
    "metrics.test_recall_at_10": "test_recall@10",
    "metrics.test_ndcg_at_10": "test_ndcg@10",
    "metrics.test_coverage_at_10": "test_coverage@10",
}


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["shoprank-ranker"])
    table = runs[[c for c in COLUMNS if c in runs]].rename(columns=COLUMNS)
    table["winner"] = runs.get("tags.winner", "").eq("true").to_numpy()
    table = table.drop_duplicates("model", keep="first").sort_values("cv_ndcg@10", ascending=False)
    out = ROOT / "reports" / "model_comparison.csv"
    table.round(5).to_csv(out, index=False)
    print(f"wrote {out} ({len(table)} models)")
    print(table[["model", "cv_ndcg@10", "test_ndcg@10", "winner"]].to_string(index=False))


if __name__ == "__main__":
    main()
