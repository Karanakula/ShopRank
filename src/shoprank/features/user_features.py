"""Per-user behaviour features computed from interactions before a cutoff."""

import pandas as pd


def user_features(interactions: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per user. `items` needs `parent_asin`, `price`, `price_missing`, `store`.

    `avg_price_paid` ignores products whose price was imputed; users with no known price get the
    median of the other users.
    """
    assert interactions["timestamp"].max() < as_of, "features must use only data before the cutoff"
    df = interactions.merge(items[["parent_asin", "price", "price_missing", "store"]], on="parent_asin", how="left")
    df["known_price"] = df["price"].where(df["price_missing"] == 0)

    g = df.groupby("user_id")
    out = pd.DataFrame({
        "n_purchases": g.size(),
        "days_since_last": (as_of - g["timestamp"].max()).dt.days,
        "avg_price_paid": g["known_price"].mean(),
        "avg_rating_given": g["rating"].mean(),
        "verified_share": g["verified_purchase"].mean(),
        "n_stores": g["store"].nunique(),
    })
    out["avg_price_paid"] = out["avg_price_paid"].fillna(out["avg_price_paid"].median())
    return out
