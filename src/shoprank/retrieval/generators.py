"""Candidate generators. Each `score(rows)` returns a dense (len(rows), n_items) array."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class ItemKNN:
    """"People who bought this also bought": sum of cosine similarities to the user's products."""

    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors = n_neighbors

    def fit(self, X: csr_matrix) -> "ItemKNN":
        n_items = X.shape[1]
        active = np.flatnonzero(X.getnnz(axis=0))
        items_by_users = X[:, active].T.tocsr()
        k = min(self.n_neighbors + 1, len(active))
        nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k).fit(items_by_users)
        dist, idx = nn.kneighbors(items_by_users)

        rows = np.repeat(active, k)
        cols = active[idx.ravel()]
        sims = 1.0 - dist.ravel()
        not_self = rows != cols
        # S[i, j] = similarity of j to i, for the top neighbours j of each product i
        self.S = csr_matrix((sims[not_self], (rows[not_self], cols[not_self])), shape=(n_items, n_items))
        self.X = X
        return self

    def score(self, rows: np.ndarray) -> np.ndarray:
        return (self.X[rows] @ self.S).toarray()


class ProfileScorer:
    """Content scorer: cosine between each product and the user's average liked-product vector.

    `item_vectors` rows follow the item index; `profile_X` marks which products build each profile.
    """

    def __init__(self, item_vectors, profile_X: csr_matrix):
        self.V = normalize(item_vectors)
        self.profile_X = profile_X

    def score(self, rows: np.ndarray) -> np.ndarray:
        profile = normalize(self.profile_X[rows] @ self.V)
        scores = profile @ self.V.T
        return scores.toarray() if issparse(scores) else np.asarray(scores)


class PopularityScorer:
    """Same score for every user: higher = more popular. `rank` is 1-based, unranked = n_items + 1."""

    def __init__(self, ranking: list, item_index: dict):
        n_items = len(item_index)
        self.rank = np.full(n_items, n_items + 1, dtype=np.int32)
        for r, item in enumerate(ranking, start=1):
            if item in item_index:
                self.rank[item_index[item]] = r
        self._score = (n_items + 1 - self.rank).astype(np.float32)

    def score(self, rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._score, (len(rows), len(self._score)))
