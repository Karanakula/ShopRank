"""Rebuild ShopRank from the raw files with one command.

    python -m shoprank.pipeline --config config.yaml [--stages clean,split,...]

Stages run in order; each reads its inputs from disk, so any suffix of the list can be rerun.
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shoprank.config import Config, load_config
from shoprank.data.clean import add_train_item_stats, clean_items, clean_reviews, k_core
from shoprank.data.load import load_items, load_reviews, sample_users
from shoprank.data.schemas import INTERACTIONS, ITEMS, RANKER_TABLE, validate
from shoprank.data.split import eval_users, time_split
from shoprank.evaluation.gate import quality_gate
from shoprank.features.segments import SegmentModel, segment_input
from shoprank.features.user_features import user_features
from shoprank.nlp.sentiment import (
    fit_sentiment,
    fit_stars_from_sentiment,
    item_sentiment_table,
    out_of_fold_positive,
    predict_positive,
)
from shoprank.ranking.dataset import build_snapshot, fit_sources
from shoprank.ranking.preprocess import DROPPED, FEATURES
from shoprank.ranking.rating import fit_rating_model, history_features, rating_metrics
from shoprank.ranking.train import baseline_metrics, score_ranker, train_ranker
from shoprank.retrieval.candidates import run_retrieval
from shoprank.retrieval.embeddings import load_or_encode
from shoprank.retrieval.matrix import interaction_matrix

log = logging.getLogger("shoprank.pipeline")

STAGES = ["load", "clean", "split", "retrieval", "features", "train", "evaluate", "export"]


# ---------- helpers ----------

def _read(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed") / f"{name}.parquet")


def _write(df: pd.DataFrame, cfg: Config, name: str) -> None:
    path = cfg.path("processed") / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s (%s rows)", path.name, f"{len(df):,}")


def _models(cfg: Config):
    path = cfg.path("models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _embeddings(cfg: Config, items: pd.DataFrame) -> np.ndarray:
    r = cfg["retrieval"]
    return load_or_encode(items["parent_asin"].to_numpy(), items["item_text"].tolist(),
                          _models(cfg) / "item_embeddings.npy", r["embedding_model"], r["embedding_batch_size"])


def _scored_interactions(cfg: Config) -> pd.DataFrame:
    """train + val + test with each review's `p_positive`."""
    parts = pd.concat([_read(cfg, s) for s in ("train", "val", "test")], ignore_index=True)
    sent = _read(cfg, "review_sentiment")
    return parts.merge(sent, on=["user_id", "parent_asin"], how="left", validate="1:1")


def _truth(window: pd.DataFrame) -> dict:
    return window[window["liked"] == 1].groupby("user_id")["parent_asin"].agg(set).to_dict()


def _git_commit(cfg: Config) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cfg.root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------- stages ----------

def load(cfg: Config) -> None:
    """Raw JSONL → `interim/reviews.parquet`, `interim/items.parquet` and a 10% user sample."""
    raw, interim = cfg.path("raw"), cfg.path("interim")
    interim.mkdir(parents=True, exist_ok=True)
    reviews = load_reviews(raw / f"{cfg['category']}.jsonl.gz")
    items = load_items(raw / f"meta_{cfg['category']}.jsonl.gz")
    reviews.to_parquet(interim / "reviews.parquet", index=False)
    items.to_parquet(interim / "items.parquet", index=False)
    sample = sample_users(reviews, 0.10, cfg.seed)
    sample.to_parquet(interim / "reviews_sample.parquet", index=False)
    items[items["parent_asin"].isin(sample["parent_asin"])].to_parquet(interim / "items_sample.parquet", index=False)
    log.info("loaded %s reviews, %s products", f"{len(reviews):,}", f"{len(items):,}")


def clean(cfg: Config) -> None:
    """Clean reviews, keep the recent window, apply the k-core filter → `interactions.parquet`."""
    reviews = pd.read_parquet(cfg.path("interim") / "reviews.parquet")
    reviews, counts = clean_reviews(reviews, pd.Timestamp(cfg["min_date"]), cfg["liked_min_rating"])
    reviews = k_core(reviews, cfg["k_core"])
    counts[f"{cfg['k_core']}-core"] = len(reviews)
    log.info("cleaning: %s", counts)
    _write(reviews, cfg, "interactions")


def split(cfg: Config) -> None:
    """Time split → train/val/test, users to score, and cleaned product metadata with train-only stats."""
    interactions = _read(cfg, "interactions")
    parts = time_split(interactions, cfg.t1, cfg.t2)
    for name, df in parts.items():
        _write(validate(df, INTERACTIONS), cfg, name)

    train_users = set(parts["train"]["user_id"])
    _write(pd.concat([eval_users(parts["val"], train_users, "val"), eval_users(parts["test"], train_users, "test")],
                     ignore_index=True), cfg, "eval_users")

    raw_items = pd.read_parquet(cfg.path("interim") / "items.parquet")
    items, median_price = clean_items(raw_items, interactions["parent_asin"])
    _write(validate(add_train_item_stats(items, parts["train"]), ITEMS), cfg, "items")
    log.info("median price used for gaps: $%.2f", median_price)


def retrieval(cfg: Config) -> dict:
    """Encode products, fit the four sources on train and measure candidate recall on validation."""
    train, val, items = _read(cfg, "train"), _read(cfg, "val"), _read(cfg, "items")
    users = _read(cfg, "eval_users").query("split == 'val' and group == 'warm'")["user_id"]
    E = _embeddings(cfg, items)

    src = fit_sources(train, items, E, cfg.t1, cfg["retrieval"])
    truth = interaction_matrix(val[val["liked"] == 1], src.user_index, src.item_index)
    q = cfg["retrieval"]["quotas"]
    table, candidates = run_retrieval(src.scorers, users.map(src.user_index).to_numpy(), src.X, truth,
                                      np.array(list(src.user_index)), np.array(list(src.item_index)),
                                      unions={"candidates": q}, keep="candidates")
    candidates["pop_rank"] = src.scorers["pop"].rank[candidates["parent_asin"].map(src.item_index).to_numpy()]
    _write(candidates.drop(columns="pop_score"), cfg, "candidates_val")
    recall = table["recall"].round(4).to_dict()
    log.info("validation recall: %s", recall)
    return recall


def features(cfg: Config) -> None:
    """Sentiment and segment models (fit on train only), then the ranker train/test copies."""
    train, val, test, items = (_read(cfg, n) for n in ("train", "val", "test", "items"))
    models = _models(cfg)
    s = cfg["sentiment"]

    sentiment = fit_sentiment(train, s["tfidf"], s["C"])
    p_train = out_of_fold_positive(train, sentiment, s["oof_folds"], cfg.seed)
    sentiment["stars"] = fit_stars_from_sentiment(p_train, train["rating"].to_numpy(float))
    joblib.dump(sentiment, models / "sentiment.joblib")
    scored = pd.concat([
        train[["user_id", "parent_asin"]].assign(p_positive=p_train),
        val[["user_id", "parent_asin"]].assign(p_positive=predict_positive(sentiment, val)),
        test[["user_id", "parent_asin"]].assign(p_positive=predict_positive(sentiment, test)),
    ], ignore_index=True)
    _write(scored, cfg, "review_sentiment")
    _write(scored.iloc[:len(train)], cfg, "train_sentiment")
    _write(item_sentiment_table(items, train, p_train), cfg, "item_sentiment")

    feats = user_features(train, items, as_of=cfg.t1)
    segments = SegmentModel(k=cfg["segments"]["k"], random_state=cfg.seed).fit(feats)
    joblib.dump(segments, models / "segments.joblib")
    feats["segment_id"] = segments.predict(feats)
    feats["segment_name"] = feats["segment_id"].map(segments.names)
    _write(feats.reset_index(), cfg, "user_segments")

    interactions = _scored_interactions(cfg)
    E = _embeddings(cfg, items)
    for name, cutoff, label_end in [("ranker_train", cfg.t1, cfg.t2), ("ranker_test", cfg.t2, None)]:
        table, recall = build_snapshot(interactions, items, E, segments, cutoff, label_end, cfg["retrieval"])
        if name == "ranker_train":
            table = table[table.groupby("user_id")["label"].transform("max") == 1].reset_index(drop=True)
        log.info("%s: candidate recall %.4f, positive rate %.4f", name, recall["recall"], table["label"].mean())
        _write(validate(table, RANKER_TABLE), cfg, name)


def train(cfg: Config) -> None:
    """Fit the ranker on the training copy and the expected-rating model on train reviews."""
    r = cfg["ranker"]
    model = train_ranker(_read(cfg, "ranker_train"), r["params"], cfg.seed, r["top_stores"])
    joblib.dump(model, _models(cfg) / "ranker.joblib")

    items = _read(cfg, "items").merge(_read(cfg, "item_sentiment"), on="parent_asin")
    train_reviews = _read(cfg, "train").merge(_read(cfg, "train_sentiment"), on=["user_id", "parent_asin"])
    rating = fit_rating_model(train_reviews, items, cfg["rating"]["learning_rate"], cfg["rating"]["n_steps"])
    joblib.dump(rating, _models(cfg) / "rating_gd.joblib")
    log.info("saved ranker.joblib and rating_gd.joblib")


def evaluate(cfg: Config) -> dict:
    """Test-window metrics for the ranker and both comparisons; validation metrics for the rating model."""
    k = cfg["evaluation"]["k"]
    items = _read(cfg, "items")
    ranker_test, test = _read(cfg, "ranker_test"), _read(cfg, "test")
    history = pd.concat([_read(cfg, "train"), _read(cfg, "val")])
    truth = _truth(test)

    ranker = joblib.load(_models(cfg) / "ranker.joblib")
    metrics = {"ranker": score_ranker(ranker, ranker_test, truth, len(items), k),
               "baselines": baseline_metrics(history, ranker_test, truth, cfg.t2, len(items), k,
                                             cfg["retrieval"]["trending_days"])}

    train_reviews = _read(cfg, "train").merge(_read(cfg, "train_sentiment"), on=["user_id", "parent_asin"])
    val = _read(cfg, "val")
    items_s = items.merge(_read(cfg, "item_sentiment"), on="parent_asin")
    rating = joblib.load(_models(cfg) / "rating_gd.joblib")
    metrics["rating_val"] = rating_metrics(rating, history_features(val, train_reviews, items_s),
                                           val["rating"].to_numpy(float))

    ndcg = f"ndcg@{k}"
    best_baseline = max(m[ndcg] for m in metrics["baselines"].values())
    log.info("test %s: ranker %.4f vs best comparison %.4f", ndcg, metrics["ranker"][ndcg], best_baseline)
    gate = cfg["quality_gate"]
    for failure in quality_gate({"test_metrics": metrics["ranker"], "baselines": metrics["baselines"]},
                                gate["min_ndcg"], gate["min_roc_auc"], k):
        log.warning("quality gate: %s", failure)
    with open(_models(cfg) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    return metrics


def export(cfg: Config) -> None:
    """Everything the API needs, fit on all data up to the last interaction."""
    models = _models(cfg)
    items = _read(cfg, "items")
    interactions = _scored_interactions(cfg)
    segments = joblib.load(models / "segments.joblib")
    E = _embeddings(cfg, items)

    serve_cutoff = interactions["timestamp"].max().normalize() + pd.Timedelta(days=1)
    candidates, _, src = build_snapshot(interactions, items, E, segments, serve_cutoff, None, cfg["retrieval"],
                                        all_users=True, return_sources=True)
    candidates = candidates.drop(columns="label")
    float_cols = candidates.select_dtypes("float64").columns
    candidates[float_cols] = candidates[float_cols].astype("float32")
    candidates.to_parquet(models / "candidates.parquet", index=False)

    joblib.dump(src.scorers["cf"], models / "item_knn.joblib")
    joblib.dump({"vectorizer": src.vectorizer, "item_ids": list(src.item_index)}, models / "tfidf.joblib")
    with open(models / "popular_items.json", "w") as f:
        json.dump(src.trending[:100], f)

    users = user_features(interactions, items, as_of=serve_cutoff)
    users["segment_id"] = segments.predict(users)
    users["segment_name"] = users["segment_id"].map(segments.names)
    Z = segments.scaler.transform(segment_input(users))
    users[["pc1", "pc2"]] = PCA(n_components=2, random_state=cfg.seed).fit_transform(Z)
    users.reset_index().to_parquet(models / "segments.parquet", index=False)

    interactions[["user_id", "parent_asin", "rating", "timestamp"]].sort_values(
        ["user_id", "timestamp"], kind="stable").to_parquet(models / "user_history.parquet", index=False)

    g = interactions.groupby("parent_asin")
    item_info = items.set_index("parent_asin")[["title", "price", "price_missing", "store", "platform"]].copy()
    item_info["review_count"] = g.size().reindex(item_info.index, fill_value=0)
    item_info["avg_rating"] = g["rating"].mean().reindex(item_info.index).fillna(interactions["rating"].mean())
    item_info["avg_sentiment"] = (g["p_positive"].mean().reindex(item_info.index)
                                  .fillna(interactions["p_positive"].mean()))
    with open(models / "items.json", "w") as f:
        json.dump(item_info.to_dict(orient="index"), f)

    metrics = json.loads((models / "metrics.json").read_text()) if (models / "metrics.json").exists() else {}
    now = datetime.now(UTC)
    metadata = {
        "model_name": "shoprank-ranker",
        "model_version": now.strftime("%Y%m%d%H%M%S"),
        "trained_at": now.isoformat(timespec="seconds"),
        "git_commit": _git_commit(cfg),
        "split": {"t1": str(cfg.t1.date()), "t2": str(cfg.t2.date()), "min_date": cfg["min_date"],
                  "k_core": cfg["k_core"]},
        "serving_cutoff": str(serve_cutoff.date()),
        "data": {"interactions": len(interactions), "users": int(interactions["user_id"].nunique()),
                 "products": len(items), "serving_users": int(candidates["user_id"].nunique()),
                 "serving_candidates": len(candidates)},
        "ranker": {"model": cfg["ranker"]["model"], "params": cfg["ranker"]["params"], "features": FEATURES,
                   "dropped_features": DROPPED},
        "retrieval": cfg["retrieval"],
        "test_metrics": metrics.get("ranker"),
        "baselines": metrics.get("baselines"),
        "rating_val": metrics.get("rating_val"),
        "random_state": cfg.seed,
    }
    with open(models / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    log.info("exported %s candidates for %s users (serving cutoff %s)", f"{len(candidates):,}",
             f"{candidates['user_id'].nunique():,}", serve_cutoff.date())


def run(cfg: Config, stages: list[str] = STAGES) -> None:
    """Run `stages` in pipeline order."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    for name in [s for s in STAGES if s in stages]:
        t = time.time()
        log.info("== %s ==", name)
        globals()[name](cfg)
        log.info("== %s done in %.0fs ==", name, time.time() - t)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(load_config(args.config), [s.strip() for s in args.stages.split(",") if s.strip()])


if __name__ == "__main__":
    main()
