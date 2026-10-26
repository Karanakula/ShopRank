"""ShopRank API.

    uvicorn api.main:app --reload        # then open http://localhost:8000/docs

Model files are read once at startup from $SHOPRANK_MODELS (default: models/); every prediction
request is appended to $SHOPRANK_PREDICTION_LOG (default: logs/predictions.jsonl).
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam

from api.schemas import (
    Health,
    History,
    RatingRequest,
    RatingResponse,
    Recommendations,
    SearchResults,
    SentimentRequest,
    SentimentResponse,
    SimilarItems,
)
from api.service import ShopRankService, UnknownItem, UnknownUser

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("shoprank.api")


class PredictionLog:
    """Append-only JSON lines, one per request; read by monitoring (Phase 17)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps({"time": datetime.now(UTC).isoformat(timespec="milliseconds")} | record)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ShopRankService(os.environ.get("SHOPRANK_MODELS", ROOT / "models"))
    app.state.predictions = PredictionLog(os.environ.get("SHOPRANK_PREDICTION_LOG", ROOT / "logs/predictions.jsonl"))
    log.info("loaded models version %s in %.1fs", app.state.service.model_version, app.state.service.load_seconds)
    yield


app = FastAPI(title="ShopRank", version="1.0",
              description="Video game recommendations from a two-stage (retrieval + LightGBM) ranker.",
              lifespan=lifespan)

K = Query(10, ge=1, le=50, description="number of products to return")


def _log(request: Request, endpoint: str, started: float, **fields) -> float:
    latency_ms = (time.perf_counter() - started) * 1000
    request.app.state.predictions.write({"endpoint": endpoint, **fields, "latency_ms": round(latency_ms, 2),
                                         "model_version": request.app.state.service.model_version})
    return latency_ms


@app.get("/health", response_model=Health)
def health(request: Request):
    return request.app.state.service.health()


@app.get("/recommend/{user_id}", response_model=Recommendations)
def recommend(request: Request, user_id: str = PathParam(min_length=1), k: int = K):
    started = time.perf_counter()
    out = request.app.state.service.recommend(user_id, k)
    _log(request, "recommend", started, user_id=user_id, strategy=out["strategy"], k=k,
         items=[i["parent_asin"] for i in out["items"]], scores=[i["score"] for i in out["items"]])
    return out


@app.get("/popular", response_model=Recommendations)
def popular(request: Request, k: int = K, user_id: str | None = Query(None, description="skip what they bought")):
    """The popularity baseline: trending products, optionally minus a user's purchases."""
    started = time.perf_counter()
    out = request.app.state.service.popular(user_id, k)
    _log(request, "popular", started, user_id=user_id, k=k, items=[i["parent_asin"] for i in out["items"]],
         scores=[])
    return out


@app.get("/users/{user_id}/history", response_model=History)
def history(request: Request, user_id: str = PathParam(min_length=1)):
    try:
        return request.app.state.service.history(user_id)
    except UnknownUser:
        raise HTTPException(404, f"unknown user {user_id!r}") from None


@app.get("/search", response_model=SearchResults)
def search(request: Request, q: str = Query(min_length=2, max_length=100), limit: int = Query(20, ge=1, le=100)):
    """Products whose title contains every word of `q`."""
    return request.app.state.service.search(q, limit)


@app.get("/similar/{item_id}", response_model=SimilarItems)
def similar(request: Request, item_id: str = PathParam(min_length=1), k: int = K):
    started = time.perf_counter()
    try:
        out = request.app.state.service.similar(item_id, k)
    except UnknownItem:
        raise HTTPException(404, f"unknown product {item_id!r}") from None
    _log(request, "similar", started, item_id=item_id, k=k, items=[i["parent_asin"] for i in out["items"]],
         scores=[i["similarity"] for i in out["items"]])
    return out


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(request: Request, body: SentimentRequest):
    started = time.perf_counter()
    out = request.app.state.service.predict_sentiment(body.text)
    _log(request, "sentiment", started, text_length=len(body.text), label=out["label"], scores=[out["p_positive"]])
    return out


@app.post("/predict-rating", response_model=RatingResponse)
def predict_rating(request: Request, body: RatingRequest):
    started = time.perf_counter()
    try:
        out = request.app.state.service.predict_rating(body.user_id, body.parent_asin)
    except UnknownItem:
        raise HTTPException(404, f"unknown product {body.parent_asin!r}") from None
    _log(request, "predict-rating", started, user_id=body.user_id, items=[body.parent_asin],
         scores=[out["expected_rating"]])
    return out
