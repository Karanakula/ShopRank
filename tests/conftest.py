"""Shared fixtures. `tiny_project` runs the real pipeline on tests/fixtures/ (26 reviews) in seconds,
so data, API and export tests work without the full dataset or a trained model."""

import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shoprank.config import load_config
from shoprank.pipeline import STAGES, run

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
REAL_MODELS = ROOT / "models"


@pytest.fixture(scope="session")
def tiny_project(tmp_path_factory):
    """Config for a finished pipeline run on the fixture data (models in `cfg.path('models')`)."""
    root = tmp_path_factory.mktemp("tiny_project")
    shutil.copy(FIXTURES / "config.yaml", root / "config.yaml")
    (root / "raw").mkdir()
    for src, dst in [("reviews.jsonl", "Fixture.jsonl.gz"), ("meta.jsonl", "meta_Fixture.jsonl.gz")]:
        with open(FIXTURES / src, "rb") as f, gzip.open(root / "raw" / dst, "wb") as g:
            g.write(f.read())
    cfg = load_config(root / "config.yaml")

    run(cfg, ["load", "clean", "split"])
    # Random unit vectors stand in for sentence embeddings so tests never download a model.
    items = pd.read_parquet(cfg.path("processed") / "items.parquet")
    E = np.random.default_rng(0).normal(size=(len(items), 16)).astype(np.float32)
    cfg.path("models").mkdir(parents=True, exist_ok=True)
    np.save(cfg.path("models") / "item_embeddings.npy", E / np.linalg.norm(E, axis=1, keepdims=True))
    np.save(cfg.path("models") / "item_embeddings_ids.npy", items["parent_asin"].to_numpy(dtype=object))
    run(cfg, STAGES[STAGES.index("retrieval"):])
    return cfg


@pytest.fixture(scope="session")
def tiny_models(tiny_project) -> Path:
    return tiny_project.path("models")
