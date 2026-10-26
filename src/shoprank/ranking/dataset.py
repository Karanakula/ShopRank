"""Ranker data: candidates + features built from data before `cutoff`, labels from the window after it.

Train copy: cutoff T1, labels from [T1, T2). Test copy: cutoff T2, labels from [T2, end).
Serving copy: cutoff after the last interaction, every known user, no labels.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from shoprank.features.segments import SegmentModel
from shoprank.features.user_features import user_features
from shoprank.retrieval.baselines import PopularRecommender
from shoprank.retrieval.candidates import run_retrieval
from shoprank.retrieval.generators import ItemKNN, PopularityScorer, ProfileScorer
from shoprank.retrieval.matrix import interaction_matrix, make_index

DEFAULT_RETRIEVAL = {
    "quotas": {"cf": 30, "tfidf": 30, "emb": 30, "pop": 75},
    "item_knn_neighbors": 50,
    "trending_days": 30,
    "tfidf": {"ngram_range": [1, 2], "min_df": 3, "max_features": 50_000, "stop_words": "english"},
}

USER_FEATURES = ["n_purchases", "avg_rating_given", "avg_price_paid", "days_since_last", "segment_id"]
ITEM_FEATURES = ["item_review_count", "item_avg_rating", "item_avg_sentiment", "price", "price_missing",
                 "store", "item_days_since_first"]
PAIR_FEATURES = ["cf_score", "tfidf_score", "emb_score", "pop_rank", "price_minus_user_avg", "bought_store_before"]
ALL_FEATURES = USER_FEATURES + ITEM_FEATURES + PAIR_FEATURES


@dataclass
class Sources:
    """Retrieval sources fit on one history, with the row/column orders they score in."""
    scorers: dict
    X: object
    vectorizer: TfidfVectorizer
    trending: list
    user_index: dict
    item_index: dict


def fit_sources(history: pd.DataFrame, items: pd.DataFrame, embeddings: np.ndarray, cutoff: pd.Timestamp,
                retrieval: dict | None = None) -> Sources:
    """Fit item-KNN, TF-IDF and embedding profiles, and trending popularity, on `history` only."""
    cfg = DEFAULT_RETRIEVAL | (retrieval or {})
    user_index = make_index(history["user_id"])
    item_index = make_index(items["parent_asin"])

    X = interaction_matrix(history, user_index, item_index)
    X_liked = interaction_matrix(history[history["liked"] == 1], user_index, item_index)
    no_liked = X_liked.getnnz(axis=1) == 0
    X_profile = (X_liked + X.multiply(no_liked[:, None])).tocsr()

    seen = items["parent_asin"].isin(history["parent_asin"])
    tf = dict(cfg["tfidf"], ngram_range=tuple(cfg["tfidf"]["ngram_range"]))
    vec = TfidfVectorizer(**tf).fit(items.loc[seen, "item_text"])
    trending = PopularRecommender(history, days=cfg["trending_days"], cutoff=cutoff).ranking

    scorers = {
        "cf": ItemKNN(n_neighbors=cfg["item_knn_neighbors"]).fit(X),
        "tfidf": ProfileScorer(vec.transform(items["item_text"]), X_profile),
        "emb": ProfileScorer(embeddings, X_profile),
        "pop": PopularityScorer(trending, item_index),
    }
    return Sources(scorers, X, vec, trending, user_index, item_index)


def item_features(history: pd.DataFrame, items: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per product, from `history` only: review count, average rating and sentiment, days since
    first review, plus price, price_missing and store from metadata."""
    g = history.groupby("parent_asin")
    out = items.set_index("parent_asin")[["price", "price_missing", "store"]].copy()
    out["item_review_count"] = g.size().reindex(out.index, fill_value=0)
    out["item_avg_rating"] = g["rating"].mean().reindex(out.index).fillna(history["rating"].mean())
    out["item_avg_sentiment"] = g["p_positive"].mean().reindex(out.index).fillna(history["p_positive"].mean())
    out["item_days_since_first"] = (cutoff - g["timestamp"].min()).dt.days.reindex(out.index).fillna(0)
    return out


def build_snapshot(interactions: pd.DataFrame, items: pd.DataFrame, embeddings: np.ndarray,
                   segments: SegmentModel, cutoff: pd.Timestamp, label_end: pd.Timestamp | None = None,
                   retrieval: dict | None = None, all_users: bool = False, return_sources: bool = False):
    """One row per (user, candidate) with `label` and ALL_FEATURES.

    `interactions` needs user_id, parent_asin, rating, liked, timestamp, verified_purchase, p_positive.
    Users scored = seen before `cutoff` and liked something in [cutoff, label_end); with
    `all_users=True`, every user seen before `cutoff` (serving). Returns (table, candidate recall)
    and, with `return_sources=True`, the fitted `Sources` as a third element.
    """
    cfg = DEFAULT_RETRIEVAL | (retrieval or {})
    history = interactions[interactions["timestamp"] < cutoff]
    in_window = interactions["timestamp"] >= cutoff
    if label_end is not None:
        in_window &= interactions["timestamp"] < label_end
    window = interactions[in_window]

    src = fit_sources(history, items, embeddings, cutoff, cfg)
    user_ids, item_ids = np.array(list(src.user_index)), np.array(list(src.item_index))
    truth = interaction_matrix(window[window["liked"] == 1], src.user_index, src.item_index)
    rows = np.arange(len(user_ids)) if all_users else np.flatnonzero(truth.getnnz(axis=1) > 0)
    stats, cands = run_retrieval(src.scorers, rows, src.X, truth, user_ids, item_ids,
                                 unions={"candidates": cfg["quotas"]}, keep="candidates")
    cands["pop_rank"] = src.scorers["pop"].rank[cands["parent_asin"].map(src.item_index).to_numpy()]
    cands = cands.drop(columns="pop_score")

    users = user_features(history, items, as_of=cutoff)
    users["segment_id"] = segments.predict(users)
    cands = cands.join(users[["n_purchases", "avg_rating_given", "avg_price_paid", "days_since_last", "segment_id"]],
                       on="user_id")
    cands = cands.join(item_features(history, items, cutoff), on="parent_asin")

    cands["price_minus_user_avg"] = cands["price"] - cands["avg_price_paid"]
    user_stores = history[["user_id", "parent_asin"]].join(items.set_index("parent_asin")["store"], on="parent_asin")
    user_stores = set(zip(user_stores["user_id"], user_stores["store"]))
    cands["bought_store_before"] = np.fromiter(
        ((u, s) in user_stores and s != "unknown" for u, s in zip(cands["user_id"], cands["store"])),
        dtype=np.int8, count=len(cands))

    cands["cutoff"] = cutoff
    table = cands[["user_id", "parent_asin", "cutoff", "label"] + ALL_FEATURES]
    if return_sources:
        return table, stats.loc["candidates"], src
    return table, stats.loc["candidates"]
