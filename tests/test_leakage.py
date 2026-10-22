"""Every ranker feature must use only data from before its cutoff.

Builds the T1 snapshot twice: on the real data, and on data where everything at or after T1 is
scrambled (ratings, sentiment, extra fake purchases). Features for the same (user, product) must be
identical; only labels may change. Needs `data/processed/` and `models/` from earlier phases.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from shoprank.nlp.sentiment import review_text
from shoprank.ranking.dataset import ALL_FEATURES, build_snapshot

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
T1, T2 = pd.Timestamp("2022-07-01"), pd.Timestamp("2023-01-01")

pytestmark = pytest.mark.skipif(not (PROC / "train_sentiment.parquet").exists() or not (ROOT / "models" / "segments.joblib").exists(),
                                reason="processed data / models not built")


@pytest.fixture(scope="module")
def inputs():
    train = pd.read_parquet(PROC / "train.parquet").merge(pd.read_parquet(PROC / "train_sentiment.parquet"),
                                                          on=["user_id", "parent_asin"])
    val = pd.read_parquet(PROC / "val.parquet")
    sent = joblib.load(ROOT / "models" / "sentiment_lr.joblib")
    val["p_positive"] = sent["model"].predict_proba(sent["vectorizer"].transform(review_text(val)))[:, 1]
    return {
        "interactions": pd.concat([train, val], ignore_index=True),
        "items": pd.read_parquet(PROC / "items.parquet"),
        "embeddings": np.load(PROC / "item_embeddings.npy"),
        "segments": joblib.load(ROOT / "models" / "segments.joblib"),
    }


def scramble_future(interactions, items, cutoff, seed=0):
    rng = np.random.default_rng(seed)
    df = interactions.copy()
    future = df["timestamp"] >= cutoff
    df.loc[future, "rating"] = rng.integers(1, 6, future.sum())
    df.loc[future, "liked"] = (df.loc[future, "rating"] >= 4).astype("int8")
    df.loc[future, "p_positive"] = rng.random(future.sum())
    df.loc[future, "verified_purchase"] = ~df.loc[future, "verified_purchase"]

    users = df.loc[df["timestamp"] < cutoff, "user_id"].drop_duplicates().sample(2000, random_state=seed)
    fake = pd.DataFrame({
        "user_id": users.to_numpy(),
        "parent_asin": rng.choice(items["parent_asin"].to_numpy(), len(users)),
        "rating": 5, "liked": np.int8(1), "verified_purchase": True, "p_positive": 1.0, "helpful_vote": 0,
        "timestamp": cutoff + pd.to_timedelta(rng.integers(0, 150, len(users)), unit="D"),
    })
    return pd.concat([df, fake], ignore_index=True).drop_duplicates(["user_id", "parent_asin"], keep="first")


def test_features_ignore_everything_after_cutoff(inputs):
    real, _ = build_snapshot(inputs["interactions"], inputs["items"], inputs["embeddings"], inputs["segments"], T1, T2)
    scrambled_data = scramble_future(inputs["interactions"], inputs["items"], T1)
    scrambled, _ = build_snapshot(scrambled_data, inputs["items"], inputs["embeddings"], inputs["segments"], T1, T2)

    key = ["user_id", "parent_asin"]
    both = real.merge(scrambled, on=key, suffixes=("", "_scrambled"))
    assert len(both) > 0.5 * len(real)
    for f in ALL_FEATURES:
        a, b = both[f], both[f"{f}_scrambled"]
        if pd.api.types.is_numeric_dtype(a):
            assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True), f
        else:
            assert (a == b).all(), f
    assert not (both["label"] == both["label_scrambled"]).all(), "labels should react to the scrambled window"


def test_scored_users_and_history_are_before_cutoff(inputs):
    df, _ = build_snapshot(inputs["interactions"], inputs["items"], inputs["embeddings"], inputs["segments"], T1, T2)
    history_users = set(inputs["interactions"].loc[inputs["interactions"]["timestamp"] < T1, "user_id"])
    assert set(df["user_id"]) <= history_users
    assert (df["item_days_since_first"] >= 0).all() and (df["days_since_last"] >= 0).all()
