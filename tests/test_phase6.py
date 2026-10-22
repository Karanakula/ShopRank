import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from shoprank.features.user_features import user_features
from shoprank.nlp.sentiment import product_sentiment, sentiment_labels
from shoprank.ranking.linear_gd import LinearRegressionGD


def test_gradient_descent_matches_closed_form():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(scale=0.1, size=500)
    gd = LinearRegressionGD(lr=0.1, n_steps=2000).fit(X, y)
    ols = LinearRegression().fit(X, y)
    assert gd.coef_ == pytest.approx(ols.coef_, abs=1e-4)
    assert gd.intercept_ == pytest.approx(ols.intercept_, abs=1e-4)
    assert gd.losses_[-1] < gd.losses_[0]


def test_gradient_descent_diverges_with_huge_learning_rate():
    X = np.random.default_rng(0).normal(size=(100, 2))
    gd = LinearRegressionGD(lr=5.0, n_steps=50).fit(X, X.sum(axis=1))
    assert not gd.losses_[-1] < gd.losses_[0]


def test_sentiment_labels_drop_three_stars():
    labels = sentiment_labels(pd.DataFrame({"rating": [1, 2, 3, 4, 5]}))
    assert labels.iloc[[0, 1, 3, 4]].tolist() == [0, 0, 1, 1]
    assert np.isnan(labels.iloc[2])


def test_product_sentiment_average():
    out = product_sentiment(pd.DataFrame({"parent_asin": ["a", "a", "b"]}), np.array([0.2, 0.6, 1.0]))
    assert out.set_index("parent_asin")["avg_sentiment"].to_dict() == pytest.approx({"a": 0.4, "b": 1.0})


def test_user_features():
    interactions = pd.DataFrame({
        "user_id": ["u", "u"], "parent_asin": ["a", "b"], "rating": [5, 3],
        "verified_purchase": [True, False],
        "timestamp": pd.to_datetime(["2022-01-01", "2022-06-21"]),
    })
    items = pd.DataFrame({"parent_asin": ["a", "b"], "price": [10.0, 27.97], "price_missing": [0, 1],
                          "store": ["X", "Y"]})
    f = user_features(interactions, items, pd.Timestamp("2022-07-01")).loc["u"]
    assert f["n_purchases"] == 2 and f["days_since_last"] == 10 and f["n_stores"] == 2
    assert f["avg_price_paid"] == 10.0 and f["avg_rating_given"] == 4.0 and f["verified_share"] == 0.5
