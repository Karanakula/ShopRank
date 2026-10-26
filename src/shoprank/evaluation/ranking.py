"""Ranking metrics for a scored candidate table (one row per user × candidate)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from shoprank.evaluation.metrics import METRICS, coverage


def top_k_lists(df: pd.DataFrame, score: str | np.ndarray, k: int = 10, ascending: bool = False) -> pd.Series:
    """Per user, the `k` candidates with the best score (ties broken by row order)."""
    s = df[score] if isinstance(score, str) else pd.Series(np.asarray(score), index=df.index)
    order = pd.DataFrame({"user_id": df["user_id"].to_numpy(), "parent_asin": df["parent_asin"].to_numpy(),
                          "s": s.to_numpy()})
    order = order.sort_values(["user_id", "s"], ascending=[True, ascending], kind="stable")
    return order.groupby("user_id", sort=False).head(k).groupby("user_id", sort=False)["parent_asin"].agg(list)


def ranking_metrics(lists: pd.Series, truth: Mapping, k: int = 10, catalog_size: int | None = None,
                    users=None) -> dict[str, float]:
    """Average Recall/Precision/HitRate/NDCG@k over `users` (default: users in `lists`).

    `truth[user]` is every product the user liked in the label window, including ones retrieval
    missed, so the ranker is charged for retrieval misses too. Users without a list score 0.
    """
    users = lists.index if users is None else users
    totals = dict.fromkeys(METRICS, 0.0)
    for u in users:
        recs = lists.get(u, [])
        liked = truth.get(u, set())
        for name, fn in METRICS.items():
            totals[name] += fn(recs, liked, k)
    out = {f"{name}@{k}": t / len(users) for name, t in totals.items()}
    if catalog_size:
        out[f"coverage@{k}"] = coverage(lists.reindex(users).dropna(), catalog_size, k)
    return out
