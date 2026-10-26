import math

import pandas as pd
import pytest

from shoprank.evaluation import (
    coverage,
    evaluate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from shoprank.retrieval.baselines import PopularRecommender

RECS = ["A", "B", "C"]
LIKED = {"B"}


def test_hand_checked_example():
    assert recall_at_k(RECS, LIKED, 3) == 1.0
    assert ndcg_at_k(RECS, LIKED, 3) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(RECS, LIKED, 3) == pytest.approx(0.631, abs=1e-3)
    assert precision_at_k(RECS, LIKED, 3) == pytest.approx(1 / 3)
    assert hit_rate_at_k(RECS, LIKED, 3) == 1.0


def test_perfect_and_empty_lists():
    assert ndcg_at_k(["B", "A"], {"A", "B"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["X", "Y"], {"A"}, 2) == 0.0
    assert recall_at_k(RECS, set(), 3) == 0.0
    assert recall_at_k(RECS, {"B", "Z"}, 1) == 0.0


def test_ndcg_rewards_rank():
    assert ndcg_at_k(["B", "A", "C"], LIKED, 3) > ndcg_at_k(RECS, LIKED, 3)


def test_coverage():
    assert coverage([["A", "B"], ["B", "C"]], catalog_size=10, k=2) == 0.3
    assert coverage([["A", "B"], ["B", "C"]], catalog_size=10, k=1) == 0.2


def test_evaluate_skips_seen_items():
    result = evaluate(lambda user, n: ["A", "B", "C", "D"][:n], users=["u"],
                      truth={"u": {"B"}}, k=1, seen={"u": {"A"}}, catalog_size=4)
    assert result["hit_rate@1"] == 1.0
    assert result["coverage@1"] == 0.25


def test_ranking_metrics_on_scored_candidates():
    from shoprank.evaluation.ranking import ranking_metrics, top_k_lists

    df = pd.DataFrame({"user_id": ["u", "u", "u", "v", "v"], "parent_asin": ["A", "B", "C", "A", "D"],
                       "score": [0.1, 0.9, 0.5, 0.3, 0.2]})
    lists = top_k_lists(df, "score", k=2)
    assert lists["u"] == ["B", "C"] and lists["v"] == ["A", "D"]
    # v liked "Z", which retrieval never surfaced: it still counts as a miss
    out = ranking_metrics(lists, {"u": {"B"}, "v": {"Z"}}, k=2, catalog_size=4)
    assert out["recall@2"] == 0.5 and out["ndcg@2"] == 0.5 and out["coverage@2"] == 1.0


def test_trending_falls_back_to_all_time():
    train = pd.DataFrame({
        "parent_asin": ["old", "old", "old", "new"],
        "liked": [1, 1, 1, 1],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2022-06-20"]),
    })
    cutoff = pd.Timestamp("2022-07-01")
    assert PopularRecommender(train).recommend(None, 2) == ["old", "new"]
    assert PopularRecommender(train, days=30, cutoff=cutoff).recommend(None, 2) == ["new", "old"]
