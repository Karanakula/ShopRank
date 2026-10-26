"""The files in `models/` that the API loads exist and fit together.

Runs on the tiny fixture pipeline always, and on the real `models/` when `make train` has been run.
"""

import json

import joblib
import pandas as pd
import pytest

from conftest import REAL_MODELS
from shoprank.ranking.models import MODEL_INPUT, predict_score

EXPORTS = ["ranker.joblib", "item_knn.joblib", "tfidf.joblib", "item_embeddings.npy", "sentiment.joblib",
           "segments.joblib", "segments.parquet", "candidates.parquet", "popular_items.json", "items.json",
           "metadata.json", "rating_gd.joblib", "user_history.parquet"]


@pytest.fixture(params=["tiny", "real"])
def models(request, tiny_models):
    if request.param == "real":
        if not (REAL_MODELS / "metadata.json").exists():
            pytest.skip("run `make train` first")
        return REAL_MODELS
    return tiny_models


def test_every_export_exists(models):
    assert [f for f in EXPORTS if not (models / f).exists()] == []


def test_candidates_score_with_the_ranker(models):
    candidates = pd.read_parquet(models / "candidates.parquet").head(2000)
    assert set(MODEL_INPUT) <= set(candidates.columns)
    scores = predict_score(joblib.load(models / "ranker.joblib"), candidates[MODEL_INPUT])
    assert ((scores >= 0) & (scores <= 1)).all()


def test_candidates_exclude_products_already_bought(models):
    candidates = pd.read_parquet(models / "candidates.parquet", columns=["user_id", "parent_asin"])
    history = pd.read_parquet(models / "user_history.parquet", columns=["user_id", "parent_asin"])
    assert len(candidates.merge(history, on=["user_id", "parent_asin"])) == 0


def test_ids_line_up_with_items(models):
    items = json.loads((models / "items.json").read_text())
    assert joblib.load(models / "tfidf.joblib")["item_ids"] == list(items)
    assert set(json.loads((models / "popular_items.json").read_text())) <= set(items)
    assert set(pd.read_parquet(models / "candidates.parquet", columns=["parent_asin"])["parent_asin"]) <= set(items)


def test_metadata_records_version_split_and_metrics(models):
    meta = json.loads((models / "metadata.json").read_text())
    for key in ["model_version", "trained_at", "split", "serving_cutoff", "test_metrics", "baselines"]:
        assert meta[key], key
