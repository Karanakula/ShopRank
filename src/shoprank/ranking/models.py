"""Ranker model zoo, user-grouped cross-validation and scoring helpers."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from shoprank.evaluation.ranking import ranking_metrics, top_k_lists
from shoprank.ranking.preprocess import FEATURES, make_preprocessor

MODEL_INPUT = FEATURES + ["user_id"]  # user_id is dropped by the preprocessor; kept so CV splitters can see it


def make_pipeline(model) -> Pipeline:
    return Pipeline([("pre", make_preprocessor()), ("model", model)])


class UserGroupKFold:
    """GroupKFold that reads the groups from the `user_id` column of X.

    Lets estimators that run their own internal CV (StackingClassifier) keep users in one fold
    without routing `groups` through `fit`.
    """

    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        return GroupKFold(self.n_splits).split(X, y, X["user_id"])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def predict_score(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Rows per user, in order of appearance (data must be sorted by user)."""
    return df.groupby("user_id", sort=False).size().to_numpy()


def cross_validate_ranker(pipe: Pipeline, df: pd.DataFrame, truth: Mapping, n_splits: int = 5, k: int = 10,
                          lambdarank: bool = False) -> tuple[np.ndarray, dict]:
    """GroupKFold by user. Returns out-of-fold scores and fold-averaged AP / ROC-AUC plus
    NDCG@k and Recall@k of the out-of-fold ranking."""
    X, y = df[MODEL_INPUT], df["label"].to_numpy()
    oof = np.zeros(len(df))
    ap, auc = [], []
    for fit_idx, pred_idx in GroupKFold(n_splits).split(X, y, df["user_id"]):
        fit_params = {"model__group": group_sizes(df.iloc[fit_idx])} if lambdarank else {}
        model = clone(pipe).fit(X.iloc[fit_idx], y[fit_idx], **fit_params)
        oof[pred_idx] = predict_score(model, X.iloc[pred_idx])
        ap.append(average_precision_score(y[pred_idx], oof[pred_idx]))
        auc.append(roc_auc_score(y[pred_idx], oof[pred_idx]))
    rank = ranking_metrics(top_k_lists(df, oof, k), truth, k)
    return oof, {"cv_ap": float(np.mean(ap)), "cv_ap_std": float(np.std(ap)), "cv_roc_auc": float(np.mean(auc)),
                 f"cv_ndcg@{k}": rank[f"ndcg@{k}"], f"cv_recall@{k}": rank[f"recall@{k}"]}
