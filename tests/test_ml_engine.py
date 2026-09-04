"""
test_ml_engine.py

Tests app/ml_engine/* using hand-constructed synthetic vectors rather than
a real Sentence-Transformers model. This is deliberate: it lets us verify
the RANKING LOGIC, EXCLUSION LOGIC, and STORAGE round-trip are all correct
independent of whether any particular embedding model is available (e.g.
in network-restricted CI environments) -- what we're testing here is "does
argpartition + exclusion + centroid math behave correctly", not "does
all-MiniLM-L6-v2 produce good embeddings" (that's a model-quality question,
evaluated instead by evaluate_recommendations.py against real data).
"""

import numpy as np
import pytest

from app.ml_engine.vector_store import VectorStore
from app.ml_engine.recommender import get_similar_papers, personalized_recommendations
from app.ml_engine.similarity import describe_score, extract_shared_terms, explain_similarity


def make_unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    """Deterministic pseudo-random unit vector for reproducible tests."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---- VectorStore ----

def test_vector_store_search_returns_most_similar_first():
    # Construct 4 vectors where we know the similarity ordering by design:
    # vector 0 is the query; vector 1 is identical to it (max similarity);
    # vector 2 is orthogonal-ish; vector 3 is the negation (min similarity).
    v0 = make_unit_vector(1)
    v1 = v0.copy()  # identical -> similarity 1.0
    v2 = make_unit_vector(2)  # unrelated
    v3 = -v0  # opposite -> similarity -1.0

    embeddings = np.stack([v0, v1, v2, v3])
    paper_ids = np.array([100, 101, 102, 103])
    store = VectorStore(embeddings, paper_ids)

    results = store.search(v0, top_k=3, exclude_ids={100})  # exclude the query paper itself
    result_ids = [pid for pid, _ in results]

    assert result_ids[0] == 101  # identical vector should rank first
    assert result_ids[-1] == 103  # opposite vector should rank last among returned
    assert 100 not in result_ids  # excluded id must never appear


def test_vector_store_save_and_load_round_trip(tmp_path):
    embeddings = np.stack([make_unit_vector(i) for i in range(5)])
    paper_ids = np.array([10, 20, 30, 40, 50])
    store = VectorStore(embeddings, paper_ids)

    emb_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "paper_ids.npy"
    store.save(emb_path, ids_path)

    loaded = VectorStore.load(emb_path, ids_path)
    assert loaded.size == 5
    assert loaded.dim == embeddings.shape[1]
    np.testing.assert_array_almost_equal(loaded.get_vector(30), embeddings[2])


def test_vector_store_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        VectorStore(np.zeros((3, 8)), np.array([1, 2]))  # 3 vectors, only 2 ids


def test_vector_store_empty_search_returns_empty_list():
    store = VectorStore(np.zeros((0, 8)), np.array([], dtype=np.int64))
    assert store.search(make_unit_vector(1), top_k=5) == []


def test_vector_store_top_k_larger_than_corpus():
    embeddings = np.stack([make_unit_vector(i) for i in range(3)])
    store = VectorStore(embeddings, np.array([1, 2, 3]))
    results = store.search(embeddings[0], top_k=100)
    assert len(results) == 3  # can't return more than exist


# ---- recommender.py ----

def test_get_similar_papers_excludes_self():
    embeddings = np.stack([make_unit_vector(i) for i in range(4)])
    store = VectorStore(embeddings, np.array([1, 2, 3, 4]))

    results = get_similar_papers(store, paper_id=1, top_k=3)
    result_ids = [pid for pid, _ in results]
    assert 1 not in result_ids


def test_get_similar_papers_missing_paper_returns_empty():
    embeddings = np.stack([make_unit_vector(i) for i in range(3)])
    store = VectorStore(embeddings, np.array([1, 2, 3]))
    assert get_similar_papers(store, paper_id=999, top_k=3) == []


def test_personalized_recommendations_excludes_bookmarked_papers():
    embeddings = np.stack([make_unit_vector(i) for i in range(6)])
    paper_ids = np.array([1, 2, 3, 4, 5, 6])
    store = VectorStore(embeddings, paper_ids)

    bookmarked = [1, 2]
    results = personalized_recommendations(store, bookmarked, top_k=4)
    result_ids = [pid for pid, _ in results]

    assert 1 not in result_ids
    assert 2 not in result_ids
    assert len(result_ids) <= 4


def test_personalized_recommendations_cold_start_returns_empty():
    embeddings = np.stack([make_unit_vector(i) for i in range(3)])
    store = VectorStore(embeddings, np.array([1, 2, 3]))
    # bookmarked ids that don't exist in the vector store at all
    assert personalized_recommendations(store, [999, 998], top_k=3) == []


# ---- similarity.py ----

def test_describe_score_bands():
    assert describe_score(0.9) == "Very strongly related"
    assert describe_score(0.6) == "Strongly related"
    assert describe_score(0.4) == "Moderately related"
    assert describe_score(0.1) == "Loosely related"


def test_extract_shared_terms_finds_overlap():
    text_a = "Graph neural networks for fraud detection in financial transactions."
    text_b = "A graph neural network approach to detecting fraudulent transactions."
    shared = extract_shared_terms(text_a, text_b, top_n=5)
    assert len(shared) > 0
    # at least one of these concepts should surface as a shared term
    assert any("graph" in term or "transaction" in term or "fraud" in term for term in shared)


def test_extract_shared_terms_no_overlap():
    text_a = "Graph neural networks for fraud detection."
    text_b = "Culinary techniques in traditional French pastry baking."
    shared = extract_shared_terms(text_a, text_b, top_n=5)
    assert shared == []


def test_explain_similarity_builds_full_payload():
    explanation = explain_similarity(
        "Graph neural networks for fraud detection in transactions.",
        "A graph neural network method for detecting transaction fraud.",
        score=0.82,
    )
    assert explanation["score_percent"] == 82
    assert explanation["label"] == "Very strongly related"
    assert "82%" in explanation["summary"]
