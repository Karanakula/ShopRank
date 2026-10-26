"""API: every endpoint answers, bad input gets 422, a new user gets popular products.

Runs against models built from the fixture data (USER_1 ... USER_6, products P01 ... P08), and
again against the real `models/` when they exist.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from conftest import REAL_MODELS

K = 3


@pytest.fixture(scope="module", params=["tiny", "real"])
def client(request, tiny_models, tmp_path_factory):
    models = tiny_models if request.param == "tiny" else REAL_MODELS
    if not (models / "metadata.json").exists():
        pytest.skip("run `make train` first")
    mp = pytest.MonkeyPatch()
    mp.setenv("SHOPRANK_MODELS", str(models))
    mp.setenv("SHOPRANK_PREDICTION_LOG", str(tmp_path_factory.mktemp("logs") / "predictions.jsonl"))
    with TestClient(app) as c:
        c.models = models
        yield c
    mp.undo()


@pytest.fixture(scope="module")
def known_user(client):
    return pd.read_parquet(client.models / "segments.parquet", columns=["user_id"])["user_id"].sort_values().iloc[0]


@pytest.fixture(scope="module")
def some_item(client):
    return client.get("/recommend/new-user", params={"k": 1}).json()["items"][0]["parent_asin"]


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and response.json()["model_version"]


def test_recommend_returns_k_items(client, known_user):
    response = client.get(f"/recommend/{known_user}", params={"k": K})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "personalized" and len(body["items"]) == K
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(i["reason"]["because_you_bought"] for i in body["items"])

    history = pd.read_parquet(client.models / "user_history.parquet").query("user_id == @known_user")
    assert {i["reason"]["because_you_bought"] for i in body["items"]} <= set(history["parent_asin"])
    assert not set(history["parent_asin"]) & {i["parent_asin"] for i in body["items"]}


def test_new_user_gets_popular_products(client):
    body = client.get("/recommend/someone-we-have-never-seen", params={"k": K}).json()
    assert body["strategy"] == "popular" and len(body["items"]) == K
    assert all(i["score"] is None for i in body["items"])


@pytest.mark.parametrize("k", [0, 51, 500, "ten"])
def test_bad_k_returns_422(client, known_user, k):
    assert client.get(f"/recommend/{known_user}", params={"k": k}).status_code == 422


def test_popular_baseline_skips_what_the_user_bought(client, known_user):
    body = client.get("/popular", params={"k": K, "user_id": known_user}).json()
    assert body["strategy"] == "popular" and len(body["items"]) == K
    bought = {h["parent_asin"] for h in client.get(f"/users/{known_user}/history").json()["items"]}
    assert not bought & {i["parent_asin"] for i in body["items"]}


def test_history(client, known_user):
    body = client.get(f"/users/{known_user}/history").json()
    assert body["items"] and all(1 <= i["rating"] <= 5 for i in body["items"])
    dates = [i["date"] for i in body["items"]]
    assert dates == sorted(dates, reverse=True)
    assert client.get("/users/nobody/history").status_code == 404


def test_search(client, some_item):
    title = client.get(f"/similar/{some_item}").json()["item"]["title"]
    query = " ".join(title.split()[:4])[:100]
    hits = client.get("/search", params={"q": query, "limit": 100}).json()["items"]
    assert some_item in {h["parent_asin"] for h in hits}
    assert client.get("/search", params={"q": "x"}).status_code == 422


def test_similar(client, some_item):
    body = client.get(f"/similar/{some_item}", params={"k": K}).json()
    assert len(body["items"]) == K and some_item not in {i["parent_asin"] for i in body["items"]}
    assert client.get("/similar/NOT-A-PRODUCT").status_code == 404


def test_sentiment(client):
    good = client.post("/sentiment", json={"text": "Great game, really fun"}).json()
    bad = client.post("/sentiment", json={"text": "Broken, waste of money"}).json()
    assert good["label"] == "positive" and bad["label"] == "negative"
    assert 1 <= bad["predicted_rating"] <= good["predicted_rating"] <= 5
    assert client.post("/sentiment", json={"text": ""}).status_code == 422


def test_predict_rating(client, known_user, some_item):
    body = client.post("/predict-rating", json={"user_id": known_user, "parent_asin": some_item}).json()
    assert 1 <= body["expected_rating"] <= 5 and body["known_user"]
    assert not client.post("/predict-rating", json={"user_id": "new", "parent_asin": some_item}).json()["known_user"]
    assert client.post("/predict-rating", json={"user_id": "x", "parent_asin": "NOPE"}).status_code == 404
    assert client.post("/predict-rating", json={"user_id": "x"}).status_code == 422


def test_requests_are_logged(client, known_user):
    import os

    client.get(f"/recommend/{known_user}", params={"k": K})
    last = json.loads(Path(os.environ["SHOPRANK_PREDICTION_LOG"]).read_text().splitlines()[-1])
    assert {"time", "user_id", "items", "scores", "latency_ms", "model_version"} <= set(last)
    assert last["user_id"] == known_user and len(last["items"]) == K
