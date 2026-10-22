"""Ranker feature list and preprocessing shared by training and serving."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "n_purchases", "avg_rating_given", "days_since_last",
    "item_review_count", "item_avg_rating", "item_avg_sentiment", "price", "price_missing", "item_days_since_first",
    "cf_score", "tfidf_score", "emb_score", "pop_rank", "price_minus_user_avg", "bought_store_before",
]
CATEGORICAL = ["segment_id", "store"]
FEATURES = NUMERIC + CATEGORICAL

DROPPED = {
    "avg_price_paid": "exactly price - price_minus_user_avg (perfect collinearity) and the weakest label correlation",
}


def make_preprocessor(numeric: list[str] = NUMERIC, categorical: list[str] = CATEGORICAL,
                      top_stores: int = 50) -> ColumnTransformer:
    """Standardize numeric columns; one-hot `segment_id` and the `top_stores` most common stores
    (the rest become one "infrequent" column, unseen values are ignored)."""
    encoders = []
    if "segment_id" in categorical:
        encoders.append(("segment", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["segment_id"]))
    if "store" in categorical:
        encoders.append(("store", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=top_stores + 1,
                                                sparse_output=False), ["store"]))
    return ColumnTransformer([("num", StandardScaler(), numeric), *encoders])
