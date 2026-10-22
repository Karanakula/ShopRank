"""K-Means user segments over `user_features` output; fit once on train, then only `.predict()`."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["n_purchases", "days_since_last", "avg_price_paid", "avg_rating_given", "verified_share", "n_stores"]
LOG_COLS = ["n_purchases", "days_since_last", "avg_price_paid", "n_stores"]


def segment_input(feats: pd.DataFrame) -> pd.DataFrame:
    x = feats[FEATURES].copy()
    x[LOG_COLS] = np.log1p(x[LOG_COLS])
    return x


def name_segments(profile: pd.DataFrame) -> dict:
    """Name clusters from their mean features: the most extreme cluster on each trait takes that
    name, and the last two split on price."""
    left = profile.copy()
    names = {}
    rules = [
        ("Unverified reviewers", "verified_share", "min"),
        ("Critical raters", "avg_rating_given", "min"),
        ("Frequent multi-store buyers", "n_purchases", "max"),
        ("Light buyers", "n_purchases", "min"),
    ]
    for name, col, how in rules:
        if len(left) <= 2:
            break
        seg = left[col].idxmin() if how == "min" else left[col].idxmax()
        names[seg] = name
        left = left.drop(seg)
    by_price = left["avg_price_paid"].sort_values()
    lapsed = ", lapsed" if (left["days_since_last"] > profile["days_since_last"].median()).all() else ""
    names[by_price.index[0]] = "Budget buyers" + lapsed
    names[by_price.index[-1]] = "Premium buyers" + lapsed
    return names


class SegmentModel:
    def __init__(self, k: int, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, feats: pd.DataFrame) -> "SegmentModel":
        x = segment_input(feats)
        self.scaler = StandardScaler().fit(x)
        self.kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_state).fit(self.scaler.transform(x))
        labels = self.kmeans.labels_
        profile = feats[FEATURES].groupby(labels).mean()
        self.names = name_segments(profile)
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(segment_input(feats)))
