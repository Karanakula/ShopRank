"""Clean reviews into interactions and clean product metadata."""

import html
import re

import numpy as np
import pandas as pd

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def clean_reviews(reviews: pd.DataFrame, min_date: pd.Timestamp,
                  liked_min_rating: int = 4) -> tuple[pd.DataFrame, dict]:
    """Drop rows missing ids/rating, exact duplicates and older repeats of a (user, product) pair;
    add `liked`; keep reviews from `min_date`. Returns the frame and row counts per step."""
    counts = {"raw": len(reviews)}
    reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating"])
    counts["after dropna"] = len(reviews)
    reviews = reviews.drop_duplicates()
    counts["after exact dedup"] = len(reviews)
    reviews = reviews.sort_values("timestamp", kind="stable").drop_duplicates(["user_id", "parent_asin"], keep="last")
    counts["after keep-latest per (user, product)"] = len(reviews)
    reviews["liked"] = (reviews["rating"] >= liked_min_rating).astype("int8")
    reviews = reviews[reviews["timestamp"] >= min_date]
    counts[f"from {min_date.date()}"] = len(reviews)
    return reviews, counts


def k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Repeatedly drop users and products with fewer than `k` interactions until none remain."""
    while True:
        user_counts = df["user_id"].map(df["user_id"].value_counts())
        item_counts = df["parent_asin"].map(df["parent_asin"].value_counts())
        keep = (user_counts >= k) & (item_counts >= k)
        if keep.all():
            return df
        df = df[keep]


def to_text(value) -> str:
    """Join list-like metadata fields into one string; missing values become ""."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        return value
    return " ".join(str(x) for x in value)


def clean_text(text: str) -> str:
    """Strip HTML tags and entities, collapse whitespace, lowercase."""
    text = html.unescape(_TAG.sub(" ", text))
    return _SPACE.sub(" ", text).strip().lower()


def clean_items(items: pd.DataFrame, keep_asins) -> tuple[pd.DataFrame, float]:
    """Metadata for `keep_asins`: median-filled `price` + `price_missing`, `item_text`, `store`
    ("unknown" if empty) and `platform`. Returns the frame and the median used."""
    it = items[items["parent_asin"].isin(keep_asins)].copy()
    it["price"] = pd.to_numeric(it["price"], errors="coerce")
    it["price_missing"] = it["price"].isna().astype("int8")
    median_price = float(it["price"].median())
    it["price"] = it["price"].fillna(median_price)
    it["item_text"] = (it["title"].map(to_text) + " " + it["features"].map(to_text) + " "
                       + it["description"].map(to_text)).map(clean_text)
    it["store"] = it["store"].fillna("").str.strip().replace("", "unknown")
    it["platform"] = it["categories"].map(lambda c: c[1] if c is not None and len(c) > 1 else "unknown")
    return it, median_price


def add_train_item_stats(items: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Replace metadata `average_rating`/`rating_number` (collected in 2023, includes the future)
    with `train_rating_count`, `train_avg_rating`, `train_liked_rate` from train reviews only."""
    stats = train.groupby("parent_asin").agg(
        train_rating_count=("rating", "size"),
        train_avg_rating=("rating", "mean"),
        train_liked_rate=("liked", "mean"),
    )
    out = items.drop(columns=["average_rating", "rating_number", "features", "description"])
    out = out.join(stats, on="parent_asin")
    out["train_rating_count"] = out["train_rating_count"].fillna(0).astype("int32")
    out["train_avg_rating"] = out["train_avg_rating"].fillna(train["rating"].mean())
    out["train_liked_rate"] = out["train_liked_rate"].fillna(train["liked"].mean())
    return out
