"""Top-K ranking metrics for one user's recommendation list, plus catalog coverage."""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

Item = Hashable


def _hits(recommended: Sequence[Item], liked: set, k: int) -> list[int]:
    return [1 if item in liked else 0 for item in recommended[:k]]


def recall_at_k(recommended: Sequence[Item], liked: set, k: int) -> float:
    if not liked:
        return 0.0
    return sum(_hits(recommended, liked, k)) / len(liked)


def precision_at_k(recommended: Sequence[Item], liked: set, k: int) -> float:
    return sum(_hits(recommended, liked, k)) / k


def hit_rate_at_k(recommended: Sequence[Item], liked: set, k: int) -> float:
    return float(any(_hits(recommended, liked, k)))


def ndcg_at_k(recommended: Sequence[Item], liked: set, k: int) -> float:
    if not liked:
        return 0.0
    hits = _hits(recommended, liked, k)
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(liked), k)))
    return dcg / idcg


def coverage(recommendation_lists: Iterable[Sequence[Item]], catalog_size: int, k: int) -> float:
    """Share of the catalog that appears in at least one user's top K."""
    shown = set()
    for recs in recommendation_lists:
        shown.update(recs[:k])
    return len(shown) / catalog_size


METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate(
    recommend_fn: Callable[[Hashable, int], Sequence[Item]],
    users: Iterable[Hashable],
    truth: Mapping[Hashable, set],
    k: int = 10,
    seen: Mapping[Hashable, set] | None = None,
    catalog_size: int | None = None,
) -> dict[str, float]:
    """Average top-K metrics over `users`.

    `recommend_fn(user, n)` returns a ranked list of at least `n` items. Items in
    `seen[user]` (already bought before the cutoff) are removed before taking the top K,
    so the function is asked for `k + len(seen[user])` items.
    """
    seen = seen or {}
    totals = dict.fromkeys(METRICS, 0.0)
    lists = []
    n_users = 0
    for user in users:
        already = seen.get(user, set())
        recs = [item for item in recommend_fn(user, k + len(already)) if item not in already][:k]
        liked = truth.get(user, set())
        for name, fn in METRICS.items():
            totals[name] += fn(recs, liked, k)
        lists.append(recs)
        n_users += 1

    result = {f"{name}@{k}": total / max(n_users, 1) for name, total in totals.items()}
    if catalog_size:
        result[f"coverage@{k}"] = coverage(lists, catalog_size, k)
    result["users"] = n_users
    return result
