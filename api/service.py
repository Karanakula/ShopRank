"""Everything the API serves, loaded once from `models/`."""

import json
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shoprank.ranking.models import MODEL_INPUT
from shoprank.ranking.rating import predict_rating


def _contiguous_slices(keys: np.ndarray) -> dict[str, tuple[int, int]]:
    """{key: (start, end)} for an array where equal keys sit next to each other."""
    if len(keys) == 0:
        return {}
    change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(keys)]
    return {k: (int(s), int(e)) for k, s, e in zip(keys[starts], starts, ends)}


class UnknownItem(KeyError):
    pass


class UnknownUser(KeyError):
    pass


class ShopRankService:
    """Holds the ranker, precomputed candidates and lookup tables; methods return plain dicts."""

    def __init__(self, models_dir: str | Path):
        d = Path(models_dir)
        t = time.perf_counter()
        self.metadata = json.loads((d / "metadata.json").read_text())
        self.model_version = self.metadata["model_version"]
        self.ranker = joblib.load(d / "ranker.joblib")
        self.sentiment = joblib.load(d / "sentiment.joblib")
        self.rating = joblib.load(d / "rating_gd.joblib")
        self.similarity = joblib.load(d / "item_knn.joblib").S.tocsr()

        self.items = json.loads((d / "items.json").read_text())
        self.item_ids = np.array(list(self.items))
        self.item_index = {a: i for i, a in enumerate(self.item_ids)}
        self.titles_lower = [(it["title"] or "").lower() for it in self.items.values()]
        self.review_counts = np.array([it["review_count"] for it in self.items.values()])
        self.embeddings = np.load(d / "item_embeddings.npy")
        emb_ids = np.load(d / "item_embeddings_ids.npy", allow_pickle=True)
        assert np.array_equal(emb_ids, self.item_ids), "embeddings and items.json are in different orders"
        self.popular_ids = [a for a in json.loads((d / "popular_items.json").read_text()) if a in self.item_index]

        candidates = pd.read_parquet(d / "candidates.parquet")
        self.cand_X = candidates[MODEL_INPUT]
        self.cand_items = candidates["parent_asin"].to_numpy()
        self.cand_slices = _contiguous_slices(candidates["user_id"].to_numpy())

        history = pd.read_parquet(d / "user_history.parquet")
        self.hist_items = history["parent_asin"].map(self.item_index).to_numpy()
        self.hist_ratings = history["rating"].to_numpy()
        self.hist_dates = history["timestamp"].dt.strftime("%Y-%m-%d").to_numpy()
        self.hist_slices = _contiguous_slices(history["user_id"].to_numpy())

        users = pd.read_parquet(d / "segments.parquet", columns=["user_id", "avg_rating_given", "segment_name"])
        self.user_avg_rating = dict(zip(users["user_id"], users["avg_rating_given"]))
        self.user_segment = dict(zip(users["user_id"], users["segment_name"]))
        self._lock = threading.Lock()
        self.load_seconds = time.perf_counter() - t

    # ---------- helpers ----------

    def product(self, asin: str) -> dict:
        it = self.items[asin]
        price = None if it["price_missing"] else it["price"]
        return {"parent_asin": asin, "title": it["title"] or "(untitled)", "price": price, "store": it["store"]}

    def _require_item(self, asin: str) -> int:
        if asin not in self.item_index:
            raise UnknownItem(asin)
        return self.item_index[asin]

    def is_known(self, user_id: str) -> bool:
        return user_id in self.cand_slices

    def _history(self, user_id: str) -> np.ndarray:
        if user_id not in self.hist_slices:
            return np.array([], dtype=int)
        h0, h1 = self.hist_slices[user_id]
        return self.hist_items[h0:h1]

    def _expected_rating(self, user_id: str, asin: str) -> tuple[float, dict]:
        it = self.items[asin]
        features = {
            "user_avg_rating": float(self.user_avg_rating.get(user_id, self.rating["global_rating"])),
            "item_avg_rating": float(it["avg_rating"]),
            "price": float(it["price"]),
            "avg_sentiment": float(it["avg_sentiment"]),
        }
        return float(predict_rating(self.rating, pd.DataFrame([features]))[0]), features

    # ---------- endpoints ----------

    def health(self) -> dict:
        m = self.metadata
        return {"status": "ok", "model_version": self.model_version, "trained_at": m["trained_at"],
                "serving_cutoff": m["serving_cutoff"], "users": len(self.cand_slices), "products": len(self.items)}

    def popular(self, user_id: str | None, k: int) -> dict:
        """Trending products (the popularity baseline), skipping anything `user_id` already bought."""
        seen = set(self.item_ids[self._history(user_id)]) if user_id else set()
        asins = [a for a in self.popular_ids if a not in seen][:k]
        items = [self.product(a) | {"rank": r, "score": None, "reason": None,
                                    "expected_rating": round(self._expected_rating(user_id or "", a)[0], 2)}
                 for r, a in enumerate(asins, 1)]
        return {"user_id": user_id or "", "strategy": "popular", "k": k, "model_version": self.model_version,
                "items": items}

    def recommend(self, user_id: str, k: int) -> dict:
        """Known user: score their precomputed candidates with the ranker. New user: trending products."""
        if not self.is_known(user_id):
            return self.popular(user_id, k)

        start, end = self.cand_slices[user_id]
        with self._lock:
            scores = self.ranker.predict_proba(self.cand_X.iloc[start:end])[:, 1]
        top = np.argsort(-scores, kind="stable")[:k]
        asins = self.cand_items[start:end][top]

        history = self._history(user_id)
        top_idx = np.array([self.item_index[a] for a in asins])
        sims = self.embeddings[top_idx] @ self.embeddings[history].T
        best = sims.argmax(axis=1)

        items = []
        for r, (asin, s, b) in enumerate(zip(asins, scores[top], best), 1):
            because = self.item_ids[history[b]]
            reason = {"because_you_bought": because, "title": self.product(because)["title"],
                      "similarity": round(float(sims[r - 1, b]), 4)}
            items.append(self.product(asin) | {"rank": r, "score": round(float(s), 6), "reason": reason,
                                               "expected_rating": round(self._expected_rating(user_id, asin)[0], 2)})
        return {"user_id": user_id, "strategy": "personalized", "k": k, "model_version": self.model_version,
                "items": items}

    def history(self, user_id: str) -> dict:
        """Products the user reviewed, newest first."""
        if user_id not in self.hist_slices:
            raise UnknownUser(user_id)
        h0, h1 = self.hist_slices[user_id]
        items = [self.product(self.item_ids[i]) | {"rating": int(r), "date": d}
                 for i, r, d in zip(self.hist_items[h0:h1], self.hist_ratings[h0:h1], self.hist_dates[h0:h1])]
        return {"user_id": user_id, "segment": self.user_segment.get(user_id), "items": items[::-1]}

    def search(self, query: str, limit: int) -> dict:
        """Products whose title contains every word of `query`, most-reviewed first."""
        words = query.lower().split()
        hits = [i for i, t in enumerate(self.titles_lower) if all(w in t for w in words)]
        hits = sorted(hits, key=lambda i: -self.review_counts[i])[:limit]
        return {"query": query, "items": [self.product(self.item_ids[i]) | {"review_count": int(self.review_counts[i])}
                                          for i in hits]}

    def similar(self, item_id: str, k: int) -> dict:
        """Co-purchase neighbours from item-KNN first, topped up with text-embedding neighbours."""
        i = self._require_item(item_id)
        row = self.similarity[i]
        order = np.argsort(-row.data, kind="stable")
        picked = [(int(row.indices[j]), float(row.data[j]), "co_purchase") for j in order[:k]]

        if len(picked) < k:
            content = self.embeddings @ self.embeddings[i]
            content[[i] + [p[0] for p in picked]] = -np.inf
            for j in np.argsort(-content)[:k - len(picked)]:
                picked.append((int(j), float(content[j]), "content"))

        items = [self.product(self.item_ids[j]) | {"similarity": round(s, 4), "source": src} for j, s, src in picked]
        return {"item": self.product(item_id), "items": items}

    def predict_sentiment(self, text: str) -> dict:
        p = float(self.sentiment["model"].predict_proba(self.sentiment["vectorizer"].transform([text]))[0, 1])
        label = "positive" if p >= 0.5 else "negative"
        stars = self.sentiment.get("stars")
        predicted = round(float(stars.predict([p])[0]), 2) if stars is not None else None
        return {"label": label, "probability": round(max(p, 1 - p), 6), "p_positive": round(p, 6),
                "predicted_rating": predicted}

    def predict_rating(self, user_id: str, parent_asin: str) -> dict:
        """Expected stars from the user's average rating and the product's rating, price and sentiment."""
        self._require_item(parent_asin)
        stars, features = self._expected_rating(user_id, parent_asin)
        return {"user_id": user_id, "parent_asin": parent_asin, "expected_rating": round(stars, 3),
                "known_user": user_id in self.user_avg_rating,
                "features": {k: round(v, 4) for k, v in features.items()}}
