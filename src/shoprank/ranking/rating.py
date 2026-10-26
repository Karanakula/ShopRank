"""Expected-rating model: linear regression by gradient descent on four standardized features."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from shoprank.ranking.linear_gd import LinearRegressionGD

FEATURES = ["user_avg_rating", "item_avg_rating", "price", "avg_sentiment"]


def loo_mean(df: pd.DataFrame, key: str, col: str, default: float) -> pd.Series:
    """Mean of `col` within each `key` group excluding the row itself; `default` for singletons."""
    g = df.groupby(key)[col]
    total, n = g.transform("sum"), g.transform("size")
    return ((total - df[col]) / (n - 1)).where(n > 1, default)


def train_features(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Features for train reviews with leave-one-out averages, so no review sees its own rating
    or sentiment. `train` needs `p_positive`; `items` needs `price`."""
    global_rating, global_sentiment = train["rating"].mean(), train["p_positive"].mean()
    return pd.DataFrame({
        "user_avg_rating": loo_mean(train, "user_id", "rating", global_rating),
        "item_avg_rating": loo_mean(train, "parent_asin", "rating", global_rating),
        "price": train[["parent_asin"]].merge(items[["parent_asin", "price"]], how="left")["price"].to_numpy(),
        "avg_sentiment": loo_mean(train, "parent_asin", "p_positive", global_sentiment),
    }, index=train.index)


def history_features(pairs: pd.DataFrame, history: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Features for (user_id, parent_asin) `pairs` from plain averages over `history`.

    `items` needs `price` and `avg_sentiment`; unknown users/products get the history mean.
    """
    global_rating = history["rating"].mean()
    p = pairs[["user_id", "parent_asin"]].merge(items[["parent_asin", "price", "avg_sentiment"]],
                                                on="parent_asin", how="left")
    return pd.DataFrame({
        "user_avg_rating": p["user_id"].map(history.groupby("user_id")["rating"].mean()).fillna(global_rating),
        "item_avg_rating": p["parent_asin"].map(history.groupby("parent_asin")["rating"].mean()).fillna(global_rating),
        "price": p["price"].to_numpy(),
        "avg_sentiment": p["avg_sentiment"].fillna(history["p_positive"].mean()).to_numpy(),
    })


def fit_rating_model(train: pd.DataFrame, items: pd.DataFrame, lr: float = 0.1, n_steps: int = 1000) -> dict:
    """Standardize on train, fit `LinearRegressionGD`; returns the artifact used by `predict_rating`."""
    X = train_features(train, items)
    scaler = StandardScaler().fit(X)
    gd = LinearRegressionGD(lr=lr, n_steps=n_steps).fit(scaler.transform(X), train["rating"].to_numpy(float))
    return {"features": FEATURES, "scaler": scaler, "coef": gd.coef_, "intercept": gd.intercept_,
            "global_rating": float(train["rating"].mean()), "clip": (1, 5)}


def predict_rating(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Expected stars, clipped to 1-5."""
    z = artifact["scaler"].transform(X[artifact["features"]])
    return np.clip(z @ artifact["coef"] + artifact["intercept"], *artifact["clip"])


def rating_metrics(artifact: dict, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    pred = predict_rating(artifact, X)
    baseline = np.full(len(y), artifact["global_rating"])
    return {"rmse": root_mean_squared_error(y, pred), "mae": mean_absolute_error(y, pred),
            "rmse_global_mean": root_mean_squared_error(y, baseline)}
