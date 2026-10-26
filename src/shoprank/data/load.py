"""Read the raw Amazon Reviews 2023 JSONL files."""

from pathlib import Path

import pandas as pd

REVIEW_COLS = ["user_id", "parent_asin", "rating", "title", "text", "timestamp", "verified_purchase", "helpful_vote"]
ITEM_COLS = ["parent_asin", "title", "features", "description", "price", "store", "categories",
             "average_rating", "rating_number"]


def load_reviews(path: str | Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Reviews, read in chunks to bound memory; `timestamp` converted from epoch ms."""
    chunks = pd.read_json(path, lines=True, chunksize=chunksize)
    reviews = pd.concat(chunk[REVIEW_COLS] for chunk in chunks)
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="ms")
    return reviews


def load_items(path: str | Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Product metadata without images/videos/bought_together; `price` coerced to numeric."""
    chunks = pd.read_json(path, lines=True, chunksize=chunksize)
    items = pd.concat(chunk[ITEM_COLS] for chunk in chunks)
    items["price"] = pd.to_numeric(items["price"], errors="coerce")
    return items


def sample_users(reviews: pd.DataFrame, frac: float, random_state: int) -> pd.DataFrame:
    """All reviews of a random `frac` of users (sampling users keeps histories whole)."""
    users = reviews["user_id"].drop_duplicates()
    keep = users.sample(frac=frac, random_state=random_state)
    return reviews[reviews["user_id"].isin(keep)]
