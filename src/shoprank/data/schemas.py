"""Pandera schemas for the tables the pipeline writes; `validate` raises on the first bad table."""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from shoprank.ranking.dataset import ALL_FEATURES

NON_EMPTY = Check.str_length(min_value=1)
IS_DATETIME = Check(lambda s: pd.api.types.is_datetime64_any_dtype(s), name="is_datetime")
BINARY = Check.isin([0, 1])

INTERACTIONS = DataFrameSchema(
    {
        "user_id": Column(str, NON_EMPTY),
        "parent_asin": Column(str, NON_EMPTY),
        "rating": Column(int, Check.isin([1, 2, 3, 4, 5])),
        "liked": Column("int8", BINARY),
        "timestamp": Column(checks=IS_DATETIME),
        "verified_purchase": Column(bool),
        "helpful_vote": Column(int, Check.ge(0)),
    },
    checks=[Check(lambda df: df["liked"] == (df["rating"] >= 4).astype("int8"), name="liked_means_4_or_5_stars")],
    unique=["user_id", "parent_asin"],
    strict=False,
    name="interactions",
)

ITEMS = DataFrameSchema(
    {
        "parent_asin": Column(str, NON_EMPTY, unique=True),
        "title": Column(str, nullable=True),
        "price": Column(float, Check.ge(0)),
        "price_missing": Column("int8", BINARY),
        "store": Column(str, NON_EMPTY),
        "platform": Column(str, NON_EMPTY),
        "item_text": Column(str),
        "train_rating_count": Column("int32", Check.ge(0)),
        "train_avg_rating": Column(float, Check.in_range(1, 5)),
        "train_liked_rate": Column(float, Check.in_range(0, 1)),
    },
    strict=False,
    name="items",
)

RANKER_TABLE = DataFrameSchema(
    {
        "user_id": Column(str, NON_EMPTY),
        "parent_asin": Column(str, NON_EMPTY),
        "label": Column("int8", BINARY),
        **{f: Column(nullable=False) for f in ALL_FEATURES},
        "days_since_last": Column(checks=Check.ge(0)),
        "item_days_since_first": Column(checks=Check.ge(0)),
        "item_avg_rating": Column(checks=Check.in_range(1, 5)),
        "item_avg_sentiment": Column(checks=Check.in_range(0, 1)),
        "pop_rank": Column(checks=Check.ge(0)),
    },
    unique=["user_id", "parent_asin"],
    strict=False,
    name="ranker_table",
)


def validate(df: pd.DataFrame, schema: DataFrameSchema) -> pd.DataFrame:
    """`schema.validate` collecting every failure, so one error message lists all problems."""
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases[["column", "check", "failure_case"]].drop_duplicates().head(20)
        raise ValueError(f"{schema.name} failed validation:\n{cases.to_string(index=False)}") from None
