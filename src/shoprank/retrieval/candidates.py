"""Per-source top-N, Recall@N and the merged candidate table."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def top_n_mask(scores: np.ndarray, seen: csr_matrix, n: int) -> np.ndarray:
    """Boolean (users, items) mask of each row's top `n` unseen items."""
    s = np.array(scores, dtype=np.float32, copy=True)
    s[seen.nonzero()] = -np.inf
    top = np.argpartition(-s, n - 1, axis=1)[:, :n]
    mask = np.zeros_like(s, dtype=bool)
    np.put_along_axis(mask, top, True, axis=1)
    return mask & np.isfinite(s)


def recall(mask: np.ndarray, truth: csr_matrix) -> np.ndarray:
    """Per-user share of liked products that the mask contains."""
    hits = truth.multiply(mask).sum(axis=1).A1
    return hits / np.maximum(truth.sum(axis=1).A1, 1)


def union_mask(masks: dict, quotas: dict) -> np.ndarray:
    return np.logical_or.reduce([masks[name][n] for name, n in quotas.items()])


def run_retrieval(sources: dict, rows: np.ndarray, seen: csr_matrix, truth: csr_matrix,
                  user_ids: np.ndarray, item_ids: np.ndarray, unions: dict, keep: str,
                  eval_n: int = 100, batch: int = 1000):
    """Score `rows` (user row numbers) with every source.

    `unions` maps a label to per-source quotas, e.g. {"cf": 30, "pop": 75}; each union is the
    merged top-quota lists. Returns (table of mean recall and mean list size, candidate
    DataFrame for the `keep` union with every source's score per candidate).
    """
    sizes_needed = {name: {eval_n} for name in sources}
    for quotas in unions.values():
        for name, n in quotas.items():
            sizes_needed[name].add(n)

    stats = {f"{name}@{eval_n}": ([], []) for name in sources} | {label: ([], []) for label in unions}
    frames = []
    for start in range(0, len(rows), batch):
        r = rows[start:start + batch]
        seen_b, truth_b = seen[r], truth[r]
        scores = {name: src.score(r) for name, src in sources.items()}
        masks = {name: {n: top_n_mask(scores[name], seen_b, n) for n in sizes_needed[name]} for name in sources}

        lists = {f"{name}@{eval_n}": masks[name][eval_n] for name in sources}
        lists |= {label: union_mask(masks, quotas) for label, quotas in unions.items()}
        for label, m in lists.items():
            stats[label][0].append(recall(m, truth_b))
            stats[label][1].append(m.sum(axis=1))

        u, i = np.nonzero(lists[keep])
        frame = {"user_id": user_ids[r][u], "parent_asin": item_ids[i]}
        for name, s in scores.items():
            frame[f"{name}_score"] = np.asarray(s)[u, i].astype(np.float32)
        frame["label"] = np.asarray(truth_b[u, i]).ravel().astype(np.int8)
        frames.append(pd.DataFrame(frame))

    table = pd.DataFrame({
        label: {"recall": np.concatenate(rec).mean(), "avg size": np.concatenate(size).mean()}
        for label, (rec, size) in stats.items()
    }).T
    return table, pd.concat(frames, ignore_index=True)
