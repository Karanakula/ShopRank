"""Train the final ranker from config and score it on the test copy."""

from collections.abc import Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from shoprank.evaluation import evaluate
from shoprank.evaluation.ranking import ranking_metrics, top_k_lists
from shoprank.ranking.models import MODEL_INPUT, predict_score
from shoprank.ranking.preprocess import make_preprocessor
from shoprank.retrieval.baselines import PopularRecommender


def make_ranker(params: Mapping, random_state: int = 42, top_stores: int = 50) -> Pipeline:
    """Preprocessor + LightGBM classifier with the tuned `params`."""
    model = lgb.LGBMClassifier(**params, random_state=random_state, verbose=-1)
    return Pipeline([("pre", make_preprocessor(top_stores=top_stores)), ("model", model)])


def train_ranker(ranker_train: pd.DataFrame, params: Mapping, random_state: int = 42, top_stores: int = 50) -> Pipeline:
    """Fit on the training copy, sorted by user (row order matters with bagging)."""
    data = ranker_train.sort_values("user_id", kind="stable")
    return make_ranker(params, random_state, top_stores).fit(data[MODEL_INPUT], data["label"])


def score_ranker(model, ranker_test: pd.DataFrame, truth: Mapping, catalog_size: int, k: int = 10) -> dict[str, float]:
    """Classification metrics on the candidate rows plus ranking metrics per user."""
    y = ranker_test["label"].to_numpy()
    s = predict_score(model, ranker_test[MODEL_INPUT])
    out = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
           "log_loss": log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))}
    rank = ranking_metrics(top_k_lists(ranker_test, s, k), truth, k, catalog_size, ranker_test["user_id"].unique())
    return out | rank


def baseline_metrics(history: pd.DataFrame, ranker_test: pd.DataFrame, truth: Mapping, cutoff: pd.Timestamp,
                     catalog_size: int, k: int = 10, trending_days: int = 30) -> dict[str, dict]:
    """The two comparisons the ranker must beat: trending popularity and candidates sorted by a
    single retrieval score."""
    users = ranker_test["user_id"].unique()
    seen = history.groupby("user_id")["parent_asin"].agg(set).to_dict()
    trending = PopularRecommender(history, days=trending_days, cutoff=cutoff)
    out = {"trending": evaluate(trending.recommend, users, truth, k, seen, catalog_size)}
    for col, asc in [("cf_score", False), ("tfidf_score", False), ("emb_score", False), ("pop_rank", True)]:
        out[f"sorted_by_{col}"] = ranking_metrics(top_k_lists(ranker_test, col, k, ascending=asc), truth, k,
                                                  catalog_size, users)
    return out
