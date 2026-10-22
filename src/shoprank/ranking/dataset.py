"""Ranker data: candidates + features built from data before `cutoff`, labels from the window after it.

Train copy: cutoff T1, labels from [T1, T2). Test copy: cutoff T2, labels from [T2, end).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from shoprank.features.segments import SegmentModel
from shoprank.features.user_features import user_features
from shoprank.retrieval.baselines import PopularRecommender
from shoprank.retrieval.candidates import run_retrieval
from shoprank.retrieval.generators import ItemKNN, PopularityScorer, ProfileScorer
from shoprank.retrieval.matrix import interaction_matrix, make_index

QUOTAS = {"cf": 30, "tfidf": 30, "emb": 30, "pop": 75}

USER_FEATURES = ["n_purchases", "avg_rating_given", "avg_price_paid", "days_since_last", "segment_id"]
ITEM_FEATURES = ["item_review_count", "item_avg_rating", "item_avg_sentiment", "price", "price_missing",
                 "store", "item_days_since_first"]
PAIR_FEATURES = ["cf_score", "tfidf_score", "emb_score", "pop_rank", "price_minus_user_avg", "bought_store_before"]
ALL_FEATURES = USER_FEATURES + ITEM_FEATURES + PAIR_FEATURES


def fit_sources(history: pd.DataFrame, items: pd.DataFrame, embeddings: np.ndarray, cutoff: pd.Timestamp,
                user_index: dict, item_index: dict) -> tuple[dict, object]:
    X = interaction_matrix(history, user_index, item_index)
    X_liked = interaction_matrix(history[history["liked"] == 1], user_index, item_index)
    no_liked = X_liked.getnnz(axis=1) == 0
    X_profile = (X_liked + X.multiply(no_liked[:, None])).tocsr()

    seen = items["parent_asin"].isin(history["parent_asin"])
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=50_000)
    vec.fit(items.loc[seen, "item_text"])

    sources = {
        "cf": ItemKNN(n_neighbors=50).fit(X),
        "tfidf": ProfileScorer(vec.transform(items["item_text"]), X_profile),
        "emb": ProfileScorer(embeddings, X_profile),
        "pop": PopularityScorer(PopularRecommender(history, days=30, cutoff=cutoff).ranking, item_index),
    }
    return sources, X


def item_features(history: pd.DataFrame, items: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    g = history.groupby("parent_asin")
    out = items.set_index("parent_asin")[["price", "price_missing", "store"]].copy()
    out["item_review_count"] = g.size().reindex(out.index, fill_value=0)
    out["item_avg_rating"] = g["rating"].mean().reindex(out.index).fillna(history["rating"].mean())
    out["item_avg_sentiment"] = g["p_positive"].mean().reindex(out.index).fillna(history["p_positive"].mean())
    out["item_days_since_first"] = (cutoff - g["timestamp"].min()).dt.days.reindex(out.index).fillna(0)
    return out


def build_snapshot(interactions: pd.DataFrame, items: pd.DataFrame, embeddings: np.ndarray,
                   segments: SegmentModel, cutoff: pd.Timestamp, label_end: pd.Timestamp | None = None,
                   quotas: dict = QUOTAS) -> tuple[pd.DataFrame, pd.Series]:
    """One row per (user, candidate) with `label` and ALL_FEATURES.

    `interactions` needs user_id, parent_asin, rating, liked, timestamp, verified_purchase, p_positive.
    Users scored = seen before `cutoff` and liked something in [cutoff, label_end). Returns the
    table and candidate recall.
    """
    history = interactions[interactions["timestamp"] < cutoff]
    in_window = interactions["timestamp"] >= cutoff
    if label_end is not None:
        in_window &= interactions["timestamp"] < label_end
    window = interactions[in_window]

    user_index = make_index(history["user_id"])
    item_index = make_index(items["parent_asin"])
    user_ids, item_ids = np.array(list(user_index)), np.array(list(item_index))

    sources, X = fit_sources(history, items, embeddings, cutoff, user_index, item_index)
    truth = interaction_matrix(window[window["liked"] == 1], user_index, item_index)
    rows = np.flatnonzero(truth.getnnz(axis=1) > 0)
    stats, cands = run_retrieval(sources, rows, X, truth, user_ids, item_ids,
                                 unions={"candidates": quotas}, keep="candidates")
    cands["pop_rank"] = sources["pop"].rank[cands["parent_asin"].map(item_index).to_numpy()]
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
    cols = ["user_id", "parent_asin", "cutoff", "label"] + ALL_FEATURES
    return cands[cols], stats.loc["candidates"]
