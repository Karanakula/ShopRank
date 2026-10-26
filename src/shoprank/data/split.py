"""Time-based split and the users to score in each window."""

import numpy as np
import pandas as pd

SPLIT_COLS = ["user_id", "parent_asin", "rating", "liked", "timestamp", "verified_purchase", "helpful_vote",
              "title", "text"]


def time_split(df: pd.DataFrame, t1: pd.Timestamp, t2: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """train < t1 <= val < t2 <= test, each sorted by time. Never shuffles across time."""
    parts = {
        "train": df[df["timestamp"] < t1],
        "val": df[(df["timestamp"] >= t1) & (df["timestamp"] < t2)],
        "test": df[df["timestamp"] >= t2],
    }
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"]["timestamp"].max() < t1 and parts["val"]["timestamp"].max() < t2
    return {name: p[SPLIT_COLS].sort_values("timestamp", kind="stable") for name, p in parts.items()}


def eval_users(window: pd.DataFrame, history_users: set, split: str) -> pd.DataFrame:
    """Users with ≥1 liked product in `window`: "warm" if in `history_users`, else "cold"."""
    liked_users = pd.Series(window.loc[window["liked"] == 1, "user_id"].unique())
    group = np.where(liked_users.isin(history_users), "warm", "cold")
    return pd.DataFrame({"user_id": liked_users, "split": split, "group": group})
