"""Review sentiment: labels, text, classifier training, out-of-fold scores and per-product averages."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DEFAULT_TFIDF = {"ngram_range": [1, 2], "min_df": 5, "max_features": 200_000, "sublinear_tf": True}


def review_text(df: pd.DataFrame) -> pd.Series:
    return df["title"].fillna("") + ". " + df["text"].fillna("")


def sentiment_labels(df: pd.DataFrame) -> pd.Series:
    """1 for 4-5 stars, 0 for 1-2 stars, NaN for 3 stars (mixed, dropped from training)."""
    return pd.Series(np.select([df["rating"] >= 4, df["rating"] <= 2], [1.0, 0.0], np.nan), index=df.index)


def make_vectorizer(params: dict | None = None) -> TfidfVectorizer:
    """TF-IDF without a stop-word list: the English list would delete "not", "no", "never"."""
    p = DEFAULT_TFIDF | (params or {})
    return TfidfVectorizer(**dict(p, ngram_range=tuple(p["ngram_range"])))


def make_classifier(C: float = 4.0) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", C=C, max_iter=2000)


def fit_sentiment(reviews: pd.DataFrame, tfidf: dict | None = None, C: float = 4.0) -> dict:
    """Fit TF-IDF + logistic regression on the 1-2★ / 4-5★ reviews of `reviews`.

    Returns {"vectorizer", "model"}; score new reviews with `predict_positive`.
    """
    y = sentiment_labels(reviews)
    labeled = reviews[y.notna()]
    vec = make_vectorizer(tfidf)
    X = vec.fit_transform(review_text(labeled))
    model = make_classifier(C).fit(X, y[y.notna()].astype(int).to_numpy())
    return {"vectorizer": vec, "model": model}


def predict_positive(sentiment: dict, reviews: pd.DataFrame) -> np.ndarray:
    """Probability that each review is positive."""
    return sentiment["model"].predict_proba(sentiment["vectorizer"].transform(review_text(reviews)))[:, 1]


def out_of_fold_positive(reviews: pd.DataFrame, sentiment: dict, n_splits: int = 5,
                         random_state: int = 42) -> np.ndarray:
    """Positive probability for every review of the training set without in-sample optimism.

    Labeled reviews get out-of-fold scores (vectorizer and classifier refit per fold); 3★ reviews
    were never trained on and are scored by the full `sentiment` model.
    """
    y = sentiment_labels(reviews)
    labeled = y.notna().to_numpy()
    text = review_text(reviews[labeled]).to_numpy()
    y_lab = y[labeled].astype(int).to_numpy()

    oof = np.zeros(labeled.sum())
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fit_idx, pred_idx in folds.split(text, y_lab):
        vec = clone(sentiment["vectorizer"]).fit(text[fit_idx])
        model = clone(sentiment["model"]).fit(vec.transform(text[fit_idx]), y_lab[fit_idx])
        oof[pred_idx] = model.predict_proba(vec.transform(text[pred_idx]))[:, 1]

    p = np.empty(len(reviews))
    p[labeled] = oof
    if (~labeled).any():
        p[~labeled] = predict_positive(sentiment, reviews[~labeled])
    return p


def fit_stars_from_sentiment(p_positive: np.ndarray, ratings: np.ndarray) -> IsotonicRegression:
    """Monotone map from positive probability to the average star rating of reviews like it.

    Fit on out-of-fold probabilities so the map reflects how the classifier behaves on unseen text.
    """
    return IsotonicRegression(y_min=1, y_max=5, out_of_bounds="clip").fit(p_positive, ratings)


def product_sentiment(reviews: pd.DataFrame, p_positive: np.ndarray) -> pd.DataFrame:
    """Per product: `avg_sentiment` (mean positive probability), plus the sum and count used for leave-one-out."""
    df = pd.DataFrame({"parent_asin": reviews["parent_asin"].to_numpy(), "p": p_positive})
    out = df.groupby("parent_asin")["p"].agg(sentiment_sum="sum", sentiment_n="size")
    out["avg_sentiment"] = out["sentiment_sum"] / out["sentiment_n"]
    return out.reset_index()


def item_sentiment_table(items: pd.DataFrame, reviews: pd.DataFrame, p_positive: np.ndarray) -> pd.DataFrame:
    """`product_sentiment` for every product in `items`; products without reviews get the overall mean."""
    out = items[["parent_asin"]].merge(product_sentiment(reviews, p_positive), how="left")
    out["avg_sentiment"] = out["avg_sentiment"].fillna(float(np.mean(p_positive)))
    out[["sentiment_sum", "sentiment_n"]] = out[["sentiment_sum", "sentiment_n"]].fillna(0)
    return out
