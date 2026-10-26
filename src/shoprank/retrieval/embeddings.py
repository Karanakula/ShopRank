"""Sentence-transformer product embeddings, cached on disk next to the product ids they belong to."""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def encode(texts: list[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> np.ndarray:
    """L2-normalized float32 embeddings, one row per text."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    log.info("encoding %d texts with %s on %s", len(texts), model_name, encoder.device)
    return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)


def load_or_encode(item_ids: np.ndarray, texts: list[str], path: str | Path, model_name: str = "all-MiniLM-L6-v2",
                   batch_size: int = 64) -> np.ndarray:
    """Load `path` if it was built for exactly `item_ids` (same order), otherwise encode and save.

    The ids are stored beside it as `<name>_ids.npy`.
    """
    path = Path(path)
    ids_path = path.with_name(path.stem + "_ids.npy")
    if path.exists() and ids_path.exists() and np.array_equal(np.load(ids_path, allow_pickle=True), item_ids):
        log.info("using cached embeddings %s", path)
        return np.load(path)
    E = encode(list(texts), model_name, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    np.save(ids_path, np.asarray(item_ids, dtype=object))
    return E
