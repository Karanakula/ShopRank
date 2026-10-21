"""Non-personalised baselines every later model must beat."""

import numpy as np
import pandas as pd


class PopularRecommender:
    """Ranks products by liked interactions in `train`.

    With `days`, only likes in the last `days` days before `cutoff` count ("trending");
    products tied or absent from that window fall back to all-time popularity.
    """

    def __init__(self, train: pd.DataFrame, days: int | None = None, cutoff: pd.Timestamp | None = None):
        liked = train[train["liked"] == 1]
        all_time = liked["parent_asin"].value_counts()
        if days is None:
            score = all_time.astype(float)
        else:
            cutoff = cutoff if cutoff is not None else train["timestamp"].max()
            recent = liked[liked["timestamp"] >= cutoff - pd.Timedelta(days=days)]
            recent_counts = recent["parent_asin"].value_counts().reindex(all_time.index, fill_value=0)
            score = recent_counts + all_time / (all_time.max() + 1)
        self.ranking = score.sort_values(ascending=False, kind="stable").index.to_list()

    def recommend(self, user, n: int) -> list:
        return self.ranking[:n]


class RandomRecommender:
    """A different random list per user, drawn from `catalog`."""

    def __init__(self, catalog, seed: int = 42):
        self.catalog = np.asarray(list(catalog))
        self.rng = np.random.default_rng(seed)

    def recommend(self, user, n: int) -> list:
        n = min(n, len(self.catalog))
        return self.catalog[self.rng.choice(len(self.catalog), size=n, replace=False)].tolist()
