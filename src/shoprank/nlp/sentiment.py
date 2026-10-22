"""Review sentiment: labels, text, and per-product average positive probability."""

import numpy as np
import pandas as pd


def review_text(df: pd.DataFrame) -> pd.Series:
    return df["title"].fillna("") + ". " + df["text"].fillna("")


def sentiment_labels(df: pd.DataFrame) -> pd.Series:
    """1 for 4-5 stars, 0 for 1-2 stars, NaN for 3 stars (mixed, dropped from training)."""
    return pd.Series(np.select([df["rating"] >= 4, df["rating"] <= 2], [1.0, 0.0], np.nan), index=df.index)


def product_sentiment(reviews: pd.DataFrame, p_positive: np.ndarray) -> pd.DataFrame:
    """Per product: `avg_sentiment` (mean positive probability), plus the sum and count used for leave-one-out."""
    df = pd.DataFrame({"parent_asin": reviews["parent_asin"].to_numpy(), "p": p_positive})
    out = df.groupby("parent_asin")["p"].agg(sentiment_sum="sum", sentiment_n="size")
    out["avg_sentiment"] = out["sentiment_sum"] / out["sentiment_n"]
    return out.reset_index()
