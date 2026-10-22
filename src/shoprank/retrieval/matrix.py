"""User × product sparse matrices with fixed row/column orders."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def make_index(values) -> dict:
    return {v: i for i, v in enumerate(pd.unique(pd.Series(values)))}


def interaction_matrix(df: pd.DataFrame, user_index: dict, item_index: dict) -> csr_matrix:
    """1 where the user interacted with the product; rows of `df` with unknown ids are dropped."""
    rows = df["user_id"].map(user_index)
    cols = df["parent_asin"].map(item_index)
    ok = rows.notna() & cols.notna()
    X = csr_matrix(
        (np.ones(ok.sum(), dtype=np.float32), (rows[ok].astype(int), cols[ok].astype(int))),
        shape=(len(user_index), len(item_index)),
    )
    X.data[:] = 1.0
    return X
