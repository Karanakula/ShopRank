"""Data validation: cleaning does what it says, and Pandera rejects bad tables."""

import pandas as pd
import pytest

from conftest import FIXTURES, ROOT
from shoprank.data.clean import clean_reviews, k_core
from shoprank.data.load import load_reviews
from shoprank.data.schemas import INTERACTIONS, ITEMS, RANKER_TABLE, validate

PROC = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def raw():
    return load_reviews(FIXTURES / "reviews.jsonl")


@pytest.fixture(scope="module")
def clean(raw):
    reviews, _ = clean_reviews(raw, pd.Timestamp("2018-01-01"))
    return k_core(reviews, 2)


def test_cleaning_drops_duplicates_and_old_reviews(raw, clean):
    assert len(raw) == 26 and len(clean) == 24
    assert not clean.duplicated(["user_id", "parent_asin"]).any()
    assert clean["timestamp"].min() >= pd.Timestamp("2018-01-01")
    assert "USER_7" not in set(clean["user_id"])


def test_k_core_leaves_no_rare_users_or_products(clean):
    assert clean.groupby("user_id").size().min() >= 2
    assert clean.groupby("parent_asin").size().min() >= 2


def test_clean_interactions_pass_schema(clean):
    validate(clean, INTERACTIONS)


@pytest.mark.parametrize("column, bad_value", [
    ("rating", 6), ("rating", 0), ("user_id", ""), ("parent_asin", ""), ("liked", 2), ("helpful_vote", -1),
])
def test_schema_rejects_bad_values(clean, column, bad_value):
    bad = clean.copy()
    bad.loc[bad.index[0], column] = bad_value
    with pytest.raises(ValueError, match="interactions failed validation"):
        validate(bad, INTERACTIONS)


def test_schema_rejects_missing_values_and_duplicates(clean):
    with pytest.raises(ValueError):
        validate(clean.assign(user_id=clean["user_id"].where(clean.index != clean.index[0])), INTERACTIONS)
    with pytest.raises(ValueError):
        validate(pd.concat([clean, clean.head(1)]), INTERACTIONS)
    with pytest.raises(ValueError):
        validate(clean.assign(liked=(1 - clean["liked"]).astype("int8")), INTERACTIONS)


def test_fixture_pipeline_tables_pass_schemas(tiny_project):
    proc = tiny_project.path("processed")
    for name in ["train", "val", "test"]:
        validate(pd.read_parquet(proc / f"{name}.parquet"), INTERACTIONS)
    validate(pd.read_parquet(proc / "items.parquet"), ITEMS)
    for name in ["ranker_train", "ranker_test"]:
        validate(pd.read_parquet(proc / f"{name}.parquet"), RANKER_TABLE)


@pytest.mark.skipif(not (PROC / "train.parquet").exists(), reason="processed data not built")
def test_real_processed_tables_pass_schemas():
    for name in ["train", "val", "test"]:
        validate(pd.read_parquet(PROC / f"{name}.parquet"), INTERACTIONS)
    validate(pd.read_parquet(PROC / "items.parquet"), ITEMS)
    for name in ["ranker_train", "ranker_test"]:
        validate(pd.read_parquet(PROC / f"{name}.parquet"), RANKER_TABLE)
