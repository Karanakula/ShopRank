import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from shoprank.retrieval.candidates import recall, top_n_mask
from shoprank.retrieval.generators import ItemKNN, ProfileScorer
from shoprank.retrieval.matrix import interaction_matrix, make_index


def test_interaction_matrix_is_binary_and_drops_unknown_ids():
    df = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u3"], "parent_asin": ["a", "b", "a", "zzz"]})
    X = interaction_matrix(df, make_index(["u1", "u2"]), make_index(["a", "b"]))
    assert X.toarray().tolist() == [[1, 1], [1, 0]]


def test_top_n_mask_skips_seen_items():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    seen = csr_matrix(np.array([[1, 0, 0, 0]]))
    assert top_n_mask(scores, seen, 2).tolist() == [[False, True, False, True]]


def test_recall_per_user():
    mask = np.array([[True, False, True], [False, False, False]])
    truth = csr_matrix(np.array([[1, 1, 0], [0, 1, 0]]))
    assert recall(mask, truth).tolist() == [0.5, 0.0]


def test_item_knn_recommends_co_bought_product():
    # users 0 and 1 bought a+b; user 2 bought only a -> b should score highest for user 2
    X = csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
    scores = ItemKNN(n_neighbors=2).fit(X).score(np.array([2]))
    assert scores[0].argmax() == 1


def test_profile_scorer_prefers_similar_vectors():
    V = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    profile = csr_matrix(np.array([[1, 0, 0]]))
    scores = ProfileScorer(V, profile).score(np.array([0]))[0]
    assert scores[1] > scores[2]
