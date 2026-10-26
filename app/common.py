"""Shared helpers for the dashboard pages: API calls, cached file loaders, the sidebar status."""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL", "http://localhost:8000")
MODELS = Path(os.getenv("SHOPRANK_MODELS", ROOT / "models"))
REPORTS = Path(os.getenv("SHOPRANK_REPORTS", ROOT / "reports"))
PREDICTION_LOG = Path(os.getenv("SHOPRANK_PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))


class APIError(RuntimeError):
    pass


def api(method: str, path: str, **kwargs) -> dict:
    """Call the ShopRank API and return JSON; raises APIError with a readable message."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=10, **kwargs)
    except requests.ConnectionError:
        raise APIError(f"Cannot reach the API at {API_URL}. Start it with `make api`.") from None
    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") \
            else r.text
        raise APIError(f"API returned {r.status_code}: {detail}")
    return r.json()


def get(path: str, **params) -> dict:
    return api("GET", path, params=params)


def post(path: str, body: dict) -> dict:
    return api("POST", path, json=body)


def show_api_error(err: APIError) -> None:
    st.error(str(err))
    st.stop()


@st.cache_data(ttl=60)
def health() -> dict | None:
    try:
        return get("/health")
    except APIError:
        return None


def sidebar_status() -> None:
    h = health()
    with st.sidebar:
        if h:
            st.success(f"API online · model {h['model_version']}")
        else:
            st.error(f"API offline ({API_URL})")


@st.cache_data
def demo_users() -> list[dict]:
    return json.loads((Path(__file__).parent / "demo_users.json").read_text())


@st.cache_data
def metadata() -> dict:
    return json.loads((MODELS / "metadata.json").read_text())


@st.cache_data
def segments() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "segments.parquet")


@st.cache_data
def model_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "model_comparison.csv")


@st.cache_data(ttl=15)
def prediction_log() -> pd.DataFrame:
    if not PREDICTION_LOG.exists():
        return pd.DataFrame()
    df = pd.read_json(PREDICTION_LOG, lines=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df


def short(title: str, n: int = 55) -> str:
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def price(value) -> str:
    return "—" if value is None or pd.isna(value) else f"${value:,.2f}"


def stars(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1f} ★"
